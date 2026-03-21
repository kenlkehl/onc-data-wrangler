"""GRPO trainer for clinical summary fine-tuning.

Uses the TRL library's GRPOTrainer to fine-tune a language model to produce
clinical summaries that contain the information needed for accurate downstream
structured extraction.

The reward signal comes from extracting structured data from generated
summaries and comparing against silver-standard labels.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import torch

from ..config import ProjectConfig

logger = logging.getLogger(__name__)


def run_grpo_training(config: ProjectConfig, notes_df: pd.DataFrame):
    """Run GRPO fine-tuning for the clinical summary model.

    Args:
        config: Project configuration with training section populated.
        notes_df: DataFrame with patient notes.
    """
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from .dataset import build_training_dataset
    from .reward import RewardFunction
    from .silver_labels import generate_silver_labels

    train_config = config.training
    ext_config = config.extraction
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate or load silver labels
    logger.info("Step 1: Generating silver labels...")
    silver_labels = _generate_silver_labels_with_servers(config, notes_df)

    # Step 2: Load tokenizer and build dataset
    logger.info("Step 2: Building training dataset...")
    tokenizer = AutoTokenizer.from_pretrained(train_config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = build_training_dataset(
        notes_df=notes_df,
        silver_labels=silver_labels,
        tokenizer=tokenizer,
        patient_id_column=ext_config.patient_id_column,
        text_column=ext_config.notes_text_column,
        date_column=ext_config.notes_date_column,
        type_column=ext_config.notes_type_column,
        max_prompt_tokens=ext_config.chunk_tokens,
        max_patients=train_config.max_patients,
    )

    if not samples:
        logger.error("No training samples generated. Check notes and silver labels.")
        return

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list([
        {"prompt": s["prompt"], "patient_id": s["patient_id"]}
        for s in samples
    ])

    # Build a lookup from patient_id -> reference extraction
    reference_map = {s["patient_id"]: s["reference_extraction"] for s in samples}

    # Step 3: Build reward function
    logger.info("Step 3: Setting up reward function...")
    reward_llm_client = _build_reward_client(config)
    reward_fn = RewardFunction(
        llm_client=reward_llm_client,
        ontology_ids=train_config.target_ontology_ids,
        silver_labels={pid: ref for pid, ref in reference_map.items()},
        cancer_type=ext_config.cancer_type,
    )

    # Build the reward callable for TRL
    # TRL's GRPOTrainer expects a reward function that takes:
    #   completions: list[str] (generated texts)
    # and returns list[float] (rewards)
    # We also need to track which patient_id each prompt corresponds to.
    prompt_to_pid = {}
    for s in samples:
        prompt_to_pid[s["prompt"][:200]] = s["patient_id"]  # use prefix as key

    def reward_function(completions: list[str], **kwargs) -> list[float]:
        """Compute rewards for a batch of generated summaries."""
        prompts = kwargs.get("prompts", [])
        rewards = []
        for i, completion in enumerate(completions):
            # Try to find patient_id from the prompt
            pid = None
            if i < len(prompts):
                prompt_prefix = prompts[i][:200]
                pid = prompt_to_pid.get(prompt_prefix)

            if pid is None:
                rewards.append(0.0)
                continue

            score = reward_fn.compute_reward(pid, completion)
            rewards.append(score)

        return rewards

    # Step 4: Configure and run GRPO training
    logger.info("Step 4: Starting GRPO training...")
    logger.info("  Model: %s", train_config.model)
    logger.info("  LoRA: %s (rank=%d)", train_config.use_lora, train_config.lora_rank)
    logger.info("  Learning rate: %s", train_config.learning_rate)
    logger.info("  Epochs: %d", train_config.num_epochs)
    logger.info("  Batch size: %d", train_config.batch_size)
    logger.info("  Num generations (G): %d", train_config.num_generations)
    logger.info("  Training samples: %d", len(samples))

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size,
        learning_rate=train_config.learning_rate,
        num_generations=train_config.num_generations,
        max_completion_length=train_config.max_summary_tokens,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        train_config.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # LoRA configuration
    peft_config = None
    if train_config.use_lora:
        peft_config = LoraConfig(
            r=train_config.lora_rank,
            lora_alpha=train_config.lora_rank * 2,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_function,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Save final model
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info("Training complete. Model saved to %s", output_dir / "final")


def _generate_silver_labels_with_servers(
    config: ProjectConfig, notes_df: pd.DataFrame,
) -> dict[str, list]:
    """Generate silver labels, managing vLLM servers if needed."""
    from .silver_labels import generate_silver_labels

    train_config = config.training
    vs_config = train_config.reward_vllm_servers
    use_managed = vs_config.gpus and train_config.reward_llm.provider == "openai"

    if use_managed:
        from ..llm.multi_client import MultiVLLMClient
        from ..llm.vllm_server import VLLMServerManager

        server_mgr = VLLMServerManager(
            model=train_config.reward_llm.model,
            gpus=vs_config.gpus,
            gpus_per_server=vs_config.gpus_per_server,
            base_port=vs_config.base_port,
            extra_args=vs_config.extra_args,
            log_dir=Path(config.output_dir) / "logs",
        )
        server_mgr.start()
        try:
            llm_client = MultiVLLMClient.from_base_urls(
                base_urls=server_mgr.base_urls,
                model=train_config.reward_llm.model,
                api_key=train_config.reward_llm.resolve_api_key(),
            )
            return generate_silver_labels(
                config, notes_df, Path(config.output_dir), llm_client,
            )
        finally:
            server_mgr.shutdown()
    else:
        return generate_silver_labels(
            config, notes_df, Path(config.output_dir),
        )


def _build_reward_client(config: ProjectConfig):
    """Build an LLM client for reward computation."""
    from ..agents.pipeline import _create_llm_client
    return _create_llm_client(config.training.reward_llm)
