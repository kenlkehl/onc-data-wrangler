"""Interactive project setup agent using Claude Agent SDK.

Walks users through configuring a Talk-to-Data project by exploring
their source data, asking questions, and writing the YAML config file.
"""
import asyncio
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


def run_setup_agent(
    data_paths: list[str] | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
    max_turns: int = 80,
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
) -> Optional[str]:
    """Run the interactive setup agent (sync wrapper).

    See _run_setup_agent_async for details.
    """
    return asyncio.run(_run_setup_agent_async(
        data_paths=data_paths,
        output_dir=output_dir,
        config_path=config_path,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    ))


async def _run_setup_agent_async(
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

        # Determine config path: default to <output_dir>/<dir_name>.yaml
        if config_path is None and output_dir is not None:
            first = Path(resolved_paths[0])
            dir_name = first.name if first.is_dir() else first.parent.name
            config_path = str(Path(output_dir) / f"{dir_name}.yaml")

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
        if config_path:
            prompt_parts.append(f"**Config file path**: {config_path}\n")
        else:
            prompt_parts.append("**Config file path**: _(default to `<output_dir>/<project_name>.yaml`)_\n")
        prompt_parts.append(f"\n**Available ontologies**:\n{ontology_list}\n\n")
        prompt_parts.append(
            "IMPORTANT: Before exploring any data files, first complete Stage 1 "
            "(Project Basics). Ask me for any missing information — project name, "
            "output directory, etc. — and confirm everything with me before "
            "proceeding to data exploration. Do NOT use any tools until you have "
            "asked me your Stage 1 questions."
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

    async def _receive_and_print() -> ResultMessage | None:
        """Receive messages from the agent, print text blocks, return ResultMessage."""
        result_msg = None
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, flush=True)
            elif isinstance(message, ResultMessage):
                result_msg = message
        return result_msg

    try:
        # Connect, then send initial prompt via query().
        # Note: connect() with a string prompt does NOT actually send it
        # when using stream-json input mode. We must use query() instead.
        await client.connect()
        await client.query(initial_prompt)

        # Receive response to initial prompt
        last_result_message = await _receive_and_print()

        # Bidirectional conversation loop: prompt user for input, send to
        # agent, print response, repeat. A ResultMessage after each turn
        # just means the agent finished its current response -- it does NOT
        # mean the session is over. We keep looping until the agent hits
        # max_turns, the budget is exhausted, or the user interrupts.
        while True:
            try:
                user_input = await asyncio.to_thread(input, "> ")
            except (EOFError, KeyboardInterrupt):
                break

            await client.query(user_input)
            last_result_message = await _receive_and_print()

    finally:
        await client.disconnect()

    if last_result_message:
        result_text = last_result_message.result
        _log_usage(last_result_message)

    return result_text
