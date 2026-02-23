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

def _get_ontology_descriptions() -> dict[str, str]:
    """Dynamically discover available ontologies from the registry."""
    from ..ontologies.registry import OntologyRegistry
    descriptions = {}
    for ont in OntologyRegistry.get_all():
        descriptions[ont.ontology_id] = f"{ont.display_name} -- {ont.description}"
    return descriptions


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

    ontology_descriptions = _get_ontology_descriptions()
    ontology_list = "\n".join(f"  - {oid}: {desc}" for oid, desc in ontology_descriptions.items())

    # Build ontology table for the system prompt
    ontology_table_lines = ["| ID | Name | Description |", "|---|---|---|"]
    from ..ontologies.registry import OntologyRegistry
    for ont in OntologyRegistry.get_all():
        ontology_table_lines.append(f"| {ont.ontology_id} | {ont.display_name} | {ont.description} |")
    ontology_table = "\n".join(ontology_table_lines)

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
        system_prompt=SETUP_SYSTEM_PROMPT.format(ontology_table=ontology_table),
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        permission_mode="acceptEdits",
        env=env,
        stderr=_log_stderr,
    )

    result_text = None
    last_result_message = None
    transcript: list[str] = []
    transcript_dir: str | None = output_dir  # may be None until user provides it

    def _transcript_path() -> Path | None:
        if transcript_dir is None:
            return None
        p = Path(transcript_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / "agent_conversation_record.txt"

    def _flush_transcript():
        path = _transcript_path()
        if path is not None:
            path.write_text("\n".join(transcript) + "\n")

    def _record_user(text: str):
        transcript.append(f"USER:\n{text}\n")
        _flush_transcript()

    def _record_agent(text: str):
        transcript.append(f"AGENT:\n{text}\n")
        _flush_transcript()

    def _try_detect_output_dir():
        """Try to read output_dir from the config YAML the agent wrote."""
        nonlocal transcript_dir
        if transcript_dir is not None:
            return
        # Check the config_path we know about, or scan for recently written YAMLs
        candidates = [config_path] if config_path else []
        for cp in candidates:
            if cp and Path(cp).exists():
                try:
                    import yaml
                    with open(cp) as f:
                        cfg = yaml.safe_load(f)
                    od = (cfg or {}).get("project", {}).get("output_dir")
                    if od:
                        transcript_dir = od
                        _flush_transcript()
                        return
                except Exception:
                    pass

    client = ClaudeSDKClient(options=options)

    # All messages flow through a single asyncio.Queue so we can drain
    # stale messages (e.g. reminders the CLI sends while the user is
    # typing) before processing the response to a new query.
    msg_queue: asyncio.Queue = asyncio.Queue()
    _STREAM_END = object()

    async def _message_reader():
        """Background task: read messages from the SDK into our queue."""
        try:
            async for message in client.receive_messages():
                await msg_queue.put(message)
        except Exception as exc:
            logger.debug("Message reader ended: %s", exc)
        await msg_queue.put(_STREAM_END)

    async def _drain_stale():
        """Consume any messages that accumulated while waiting for input.

        The CLI may send unsolicited messages (reminders, notifications)
        while the user is idle. These sit in the queue and would otherwise
        be mistaken for the response to the next query. We silently
        discard them so that _receive_and_print only sees the real
        response to the user's new input.
        """
        if msg_queue.empty():
            return
        discarded = 0
        while True:
            try:
                msg = await asyncio.wait_for(msg_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                break
            if msg is _STREAM_END:
                await msg_queue.put(_STREAM_END)  # re-queue for others
                break
            discarded += 1
            # All stale messages (AssistantMessage, ResultMessage, etc.)
            # are silently consumed — they are no longer relevant since
            # the user has provided new input.
        if discarded:
            logger.debug("Drained %d stale messages from queue", discarded)

    async def _receive_and_print() -> ResultMessage | None:
        """Read from the queue until a ResultMessage signals end of turn."""
        result_msg = None
        turn_texts: list[str] = []
        while True:
            msg = await msg_queue.get()
            if msg is _STREAM_END:
                break
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text, flush=True)
                        turn_texts.append(block.text)
            elif isinstance(msg, ResultMessage):
                result_msg = msg
                break
        if turn_texts:
            _record_agent("\n".join(turn_texts))
            _try_detect_output_dir()
        return result_msg

    reader_task = None
    try:
        # Connect, then send initial prompt via query().
        # Note: connect() with a string prompt does NOT actually send it
        # when using stream-json input mode. We must use query() instead.
        await client.connect()
        reader_task = asyncio.create_task(_message_reader())

        _record_user(initial_prompt)
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

            # Drain any messages the CLI sent while the user was typing
            # (reminders, notifications, etc.) so they don't get confused
            # with the response to the user's new input.
            await _drain_stale()

            _record_user(user_input)
            await client.query(user_input)
            last_result_message = await _receive_and_print()

    finally:
        if reader_task:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
        await client.disconnect()

    if last_result_message:
        result_text = last_result_message.result
        _log_usage(last_result_message)

    # Final flush — if output_dir was discovered late, ensure transcript is written
    _try_detect_output_dir()
    _flush_transcript()

    return result_text
