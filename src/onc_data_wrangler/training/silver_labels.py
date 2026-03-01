"""Silver label generation for GRPO training.

Runs structured extraction on full patient notes using the target ontologies
to create ground-truth ("silver standard") labels. These are used as the
reference for computing rewards during GRPO fine-tuning.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from ..config import ProjectConfig
from ..extraction.chunked import (
    ChunkedExtractor,
    CheckpointManager,
    concatenate_patient_notes,
)
from ..extraction.extractor import Extractor

logger = logging.getLogger(__name__)


def generate_silver_labels(
    config: ProjectConfig,
    notes_df: pd.DataFrame,
    output_dir: Path,
    llm_client=None,
) -> dict[str, list]:
    """Generate silver-standard structured extractions from full notes.

    Args:
        config: Project configuration.
        notes_df: DataFrame with patient notes.
        output_dir: Directory to save silver extractions.
        llm_client: Optional pre-built LLM client for reward extraction.

    Returns:
        Dict mapping patient_id -> extraction list.
    """
    output_dir = Path(output_dir)
    silver_dir = output_dir / "silver_extractions"
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing silver labels
    silver_path = silver_dir / "extractions.parquet"
    silver_json_path = silver_dir / "extractions.json"
    if silver_json_path.exists():
        logger.info("Loading existing silver labels from %s", silver_json_path)
        with open(silver_json_path) as f:
            return json.load(f)

    train_config = config.training
    ext_config = config.extraction

    # Build LLM client for silver label extraction
    if llm_client is None:
        from ..agents.pipeline import _create_llm_client
        llm_client = _create_llm_client(train_config.reward_llm)

    # Create extractor with target structured ontologies
    extractor = Extractor(
        llm_client=llm_client,
        ontology_ids=train_config.target_ontology_ids,
        cancer_type=ext_config.cancer_type,
    )

    # Load tokenizer for chunking
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(train_config.reward_llm.model)
        logger.info("Loaded tokenizer for silver label extraction: %s", train_config.reward_llm.model)
    except Exception:
        logger.info("No tokenizer available; using single-chunk silver extraction")

    chunked = ChunkedExtractor(
        extractor=extractor,
        tokenizer=tokenizer,
        chunk_size=ext_config.chunk_tokens,
        overlap=ext_config.overlap_tokens,
        max_retries=ext_config.max_retries,
        patient_workers=ext_config.patient_workers,
    )

    # Run extraction
    logger.info("Generating silver labels for %d notes using ontologies %s",
                len(notes_df), train_config.target_ontology_ids)

    chunked.extract_cohort(
        notes_df=notes_df,
        output_dir=silver_dir,
        patient_id_column=ext_config.patient_id_column,
        text_column=ext_config.notes_text_column,
        date_column=ext_config.notes_date_column,
        type_column=ext_config.notes_type_column,
        resume=True,
    )

    # Collect final extractions per patient from round files
    ckpt = CheckpointManager(silver_dir)
    final_extractions: dict[str, list] = {}
    for round_idx in ckpt.find_round_files():
        round_data = ckpt.load_round(round_idx)
        for pid, record in round_data.items():
            final_extractions[pid] = record["extraction"]

    # Save as JSON for easy loading during training
    with open(silver_json_path, "w") as f:
        json.dump(final_extractions, f)
    logger.info("Saved silver labels for %d patients to %s",
                len(final_extractions), silver_json_path)

    return final_extractions
