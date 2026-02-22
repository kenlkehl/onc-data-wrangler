"""Interactive project setup agent using Claude Agent SDK.

Walks users through configuring a Talk-to-Data project by exploring
their source data, asking questions, and writing the YAML config file.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .prompts import SETUP_SYSTEM_PROMPT
from .discovery import DEFAULT_MAX_BUDGET_USD

logger = logging.getLogger(__name__)

ONTOLOGY_DESCRIPTIONS = {
    'naaccr': 'NAACCR v25 -- North American cancer registry fields',
    'pan_top': 'Pan-TOP -- Thoracic oncology (lung, mesothelioma, thymus)',
    'prissmm': 'PRISSMM -- GENIE BPC clinical data model',
    'omop': 'OMOP CDM -- Common Data Model oncology extension',
    'matchminer_ai': 'MatchMiner-AI -- Clinical trial matching concepts',
    'msk_chord': 'MSK-CHORD -- MSK oncology data model',
}


def _print_message(message):
    """Print assistant text blocks to stdout.

    Returns the text that was printed (empty string if nothing printed).
    """
    from claude_agent_sdk.types import AssistantMessage, TextBlock
    if not isinstance(message, AssistantMessage):
        return ""
    printed = ""
    for block in message.content:
        if isinstance(block, TextBlock):
            print(block.text, flush=True)
            printed += block.text
    return printed


def run_setup_agent(
    data_paths: list[str] | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
    max_turns: int = 80,
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
) -> Optional[str]:
    """Run the interactive setup agent.

    Uses Claude Agent SDK's ClaudeSDKClient for bidirectional, interactive
    conversation. The agent explores data files, asks the user questions,
    waits for responses, and writes the YAML config file.

    All parameters are optional. When omitted, the agent will ask the user
    for them interactively at the start of the session.

    Args:
        data_paths: Files and/or directories containing source data.
            If None, the agent asks the user interactively.
        output_dir: Directory for pipeline outputs.
            If None, the agent asks the user interactively.
        config_path: Path for the generated config YAML.
            If None, derived from project name or asked interactively.
        max_turns: Maximum agent interaction turns.
        max_budget_usd: Maximum budget in USD for the agent session.

    Returns:
        Agent result text.
    """
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock
    from .discovery import _log_usage

    ontology_list = "\n".join(f"  - {oid}: {desc}" for oid, desc in ONTOLOGY_DESCRIPTIONS.items())

    if data_paths:
        resolved_paths = [str(Path(p).resolve()) for p in data_paths]
        paths_list = "\n".join(f"  - {p}" for p in resolved_paths)

        # Determine config path from data paths if not provided
        if config_path is None:
            first = Path(resolved_paths[0])
            dir_name = first.name if first.is_dir() else first.parent.name
            config_path = str(Path("configs") / f"{dir_name}.yaml")

        # Determine working directory
        cwd = str(Path(os.path.commonpath(
            [str(Path(p).parent) if Path(p).is_file() else str(p) for p in resolved_paths]
        )))

        # Build prompt with known parameters
        prompt_parts = ["I need help setting up a new Talk-to-Data project.\n"]
        prompt_parts.append(f"**Source data paths**:\n{paths_list}\n")
        if output_dir:
            prompt_parts.append(f"**Output directory**: {output_dir}\n")
        else:
            prompt_parts.append("**Output directory**: _(please ask me)_\n")
        prompt_parts.append(f"**Config file path**: {config_path}\n")
        prompt_parts.append(f"\n**Available ontologies**:\n{ontology_list}\n\n")
        prompt_parts.append(
            "Please walk me through the setup process step by step, "
            "exploring my data files and writing the config YAML as we go."
        )
        initial_prompt = "\n".join(prompt_parts)
    else:
        # Fully interactive mode -- agent asks for everything
        cwd = str(Path.cwd())
        initial_prompt = (
            "I need help setting up a new Talk-to-Data project.\n\n"
            "I haven't specified any paths yet. Please start by asking me for:\n"
            "1. The paths to my source data files and/or directories\n"
            "2. Where I'd like the pipeline output directory to be\n"
            "3. Where to save the generated config YAML file\n\n"
            f"**Available ontologies**:\n{ontology_list}\n\n"
            "Then walk me through the rest of the setup process step by step."
        )

    env = {**os.environ, "CLAUDECODE": ""}

    def _log_stderr(line):
        print(f"[claude-cli] {line.rstrip()}", file=sys.stderr)

    logger.info("Starting setup agent (max_budget_usd=$%.2f)", max_budget_usd)

    options = ClaudeAgentOptions(
        model="claude-opus-4-6",
        cwd=cwd,
        allowed_tools=("Read", "Write", "Edit", "Glob", "Grep", "Bash"),
        system_prompt=SETUP_SYSTEM_PROMPT,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        permission_mode="acceptEdits",
        env=env,
        stderr=_log_stderr,
    )

    result_text = None
    last_result_message = None

    client = ClaudeSDKClient(options=options)
    all_streamed_texts = []

    try:
        # Send initial prompt and receive streaming response
        for message in client.connect(initial_prompt):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        all_streamed_texts.append(block.text)
            elif isinstance(message, ResultMessage):
                last_result_message = message
                result_text = message.result
                logger.debug(
                    "ResultMessage: subtype=%s, is_error=%s, result_len=%d, num_turns=%d",
                    message.subtype, message.is_error, len(message.result or ""), message.num_turns,
                )
                break

        if last_result_message is None:
            # Bidirectional conversation loop
            all_streamed = "".join(all_streamed_texts)
            while True:
                try:
                    user_input = input("> ")
                except (EOFError, KeyboardInterrupt):
                    break

                all_streamed_texts = []
                for message in client.query(user_input):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(block.text, flush=True)
                                all_streamed_texts.append(block.text)
                    elif isinstance(message, ResultMessage):
                        last_result_message = message
                        result_text = message.result

                        # Print any part of result not already streamed
                        extra = result_text or ""
                        all_streamed = "".join(all_streamed_texts)
                        if extra and all_streamed:
                            idx = extra.find(all_streamed)
                            if idx >= 0:
                                before = extra[:idx]
                                after = extra[idx + len(all_streamed):]
                                if after.strip():
                                    logger.debug("Printing %d extra trailing chars from ResultMessage.result", len(after))
                                    print(after, flush=True)
                                if before.strip():
                                    logger.debug("Printing %d extra leading chars from ResultMessage.result", len(before))
                            else:
                                logger.debug(
                                    "ResultMessage.result (%d chars) could not be matched against streamed text (%d chars); printing in full",
                                    len(extra), len(all_streamed),
                                )
                                print(extra, flush=True)
                        break

                if last_result_message is not None:
                    break
    finally:
        client.disconnect()

    if last_result_message:
        _log_usage(last_result_message)

    return result_text
