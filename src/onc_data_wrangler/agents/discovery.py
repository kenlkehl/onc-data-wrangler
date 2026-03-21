"""Field discovery agent using Claude Agent SDK.

Helps users explore source data files and identify fields relevant
to their clinical dataset project using interactive agent exploration.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from .prompts import DISCOVERY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MAX_BUDGET_USD = 10.0


def _log_usage(message):
    """Log token usage and cost from a ResultMessage."""
    if message.usage:
        logger.info("Token usage: %s", message.usage)
    if message.total_cost_usd:
        logger.info("Total cost: $%.4f", message.total_cost_usd)
    logger.info("Turns: %d, Duration: %dms (API: %dms)",
                message.num_turns, message.duration_ms, message.duration_api_ms)


async def run_discovery_agent(
    data_paths: list[str],
    ontology_ids: list[str],
    output_config_path: str = None,
    max_turns: int = 50,
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
) -> str:
    """Run the field discovery agent interactively.

    Uses Claude Agent SDK to explore source data files, match columns
    to ontology fields, and generate field mapping configurations.

    Args:
        data_paths: Files and/or directories containing source data.
        ontology_ids: Ontology IDs to match against.
        output_config_path: Path to save discovered field mappings.
        max_turns: Maximum agent turns.
        max_budget_usd: Maximum budget in USD for the agent session.

    Returns:
        Agent result text with discovered mappings.
    """
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk.types import ResultMessage
    import os

    resolved_paths = [str(Path(p).resolve()) for p in data_paths]

    # Find a common working directory
    if len(resolved_paths) == 1 and Path(resolved_paths[0]).is_file():
        cwd = str(Path(resolved_paths[0]).parent)
    else:
        cwd = str(Path(os.path.commonpath(resolved_paths)))

    ontology_list = ", ".join(ontology_ids) if ontology_ids else "naaccr"
    paths_desc = "\n".join(f"  - {p}" for p in resolved_paths)

    prompt = (
        f"Explore the following data paths and help me identify fields that map to the following ontologies: {ontology_list}.\n\n"
        f"**Data paths**:\n{paths_desc}\n\n"
        "For each source file, examine the columns, data types, and sample values (without showing raw patient data). "
        "Then suggest field mappings in YAML format."
    )
    if output_config_path:
        prompt += f"\n\nSave the resulting field_mappings YAML section to {output_config_path}"

    env = {**os.environ, "CLAUDECODE": ""}

    logger.info("Starting discovery agent (max_budget_usd=$%.2f)", max_budget_usd)

    message = await query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model="claude-opus-4-6",
            cwd=cwd,
            allowed_tools=("Read", "Glob", "Grep", "Bash"),
            system_prompt=DISCOVERY_SYSTEM_PROMPT,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            permission_mode="acceptEdits",
            env=env,
        ),
    )

    result_text = message.result if isinstance(message, ResultMessage) else str(message)
    _log_usage(message)
    return result_text


async def run_cohort_discovery_agent(
    data_paths: list[str],
    max_turns: int = 30,
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
) -> str:
    """Run an agent to help discover cohort definition criteria.

    Explores source tables and helps users define inclusion/exclusion
    criteria for their cohort.

    Args:
        data_paths: Files and/or directories containing source data.
        max_turns: Maximum agent turns.
        max_budget_usd: Maximum budget in USD for the agent session.

    Returns:
        Agent result with cohort criteria suggestions.
    """
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk.types import ResultMessage
    import os

    resolved_paths = [str(Path(p).resolve()) for p in data_paths]

    if len(resolved_paths) == 1 and Path(resolved_paths[0]).is_file():
        cwd = str(Path(resolved_paths[0]).parent)
    else:
        cwd = str(Path(os.path.commonpath(resolved_paths)))

    paths_desc = "\n".join(f"  - {p}" for p in resolved_paths)

    prompt = (
        f"Explore the following data paths to help define a patient cohort.\n\n"
        f"**Data paths**:\n{paths_desc}\n\n"
        "1. List available data files and their schemas\n"
        "2. Identify columns that could be used for cohort filtering (diagnosis codes, dates, demographics)\n"
        "3. Suggest cohort definition criteria\n"
        "4. Identify the patient ID column\n\n"
        "Do NOT show raw patient data. Only describe column names, types, and value distributions."
    )

    env = {**os.environ, "CLAUDECODE": ""}

    logger.info("Starting cohort discovery agent (max_budget_usd=$%.2f)", max_budget_usd)

    message = await query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model="claude-opus-4-6",
            cwd=cwd,
            allowed_tools=("Read", "Glob", "Grep", "Bash"),
            system_prompt=DISCOVERY_SYSTEM_PROMPT,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            env=env,
        ),
    )

    result_text = message.result if isinstance(message, ResultMessage) else str(message)
    _log_usage(message)
    return result_text
