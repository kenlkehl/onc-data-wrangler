"""Setup agent HTTP bridge API router."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/setup", tags=["setup"])

# In-memory store of active setup sessions
_sessions: dict[str, "SetupAgentSession"] = {}

MAX_CONCURRENT_SESSIONS = 3


_KEEPALIVE_INTERVAL = 15  # seconds between heartbeats


def _sse_event(event: str, data: str) -> str:
    escaped = data.replace("\n", "\ndata: ")
    return f"event: {event}\ndata: {escaped}\n\n"


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _summarize_tool_input(inp: dict[str, Any]) -> str:
    """Return a short summary of tool input for the frontend."""
    # For common tools, show the most useful field
    if "command" in inp:
        return _truncate(inp["command"], 120)
    if "pattern" in inp:
        return _truncate(inp["pattern"], 120)
    if "file_path" in inp:
        return inp["file_path"]
    return _truncate(json.dumps(inp, default=str), 120)


class StartSetupRequest(BaseModel):
    data_paths: list[str] | None = None
    output_dir: str | None = None
    config_path: str | None = None
    max_budget: float = 10.0
    provider: str = "claude"
    model: str | None = None
    ollama_base_url: str = "http://localhost:11434"


class SetupAgentSession:
    """Manages a single setup agent session over HTTP."""

    def __init__(
        self,
        session_id: str,
        data_paths: list[str] | None,
        output_dir: str | None,
        config_path: str | None,
        max_budget_usd: float,
        provider: str = "claude",
        model: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.session_id = session_id
        self.data_paths = data_paths
        self.output_dir = output_dir
        self.config_path = config_path
        self.max_budget_usd = max_budget_usd
        self.provider = provider
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.client = None
        self._reader_task: asyncio.Task | None = None
        self._msg_queue: asyncio.Queue = asyncio.Queue()
        self._STREAM_END = object()
        self._connected = False
        self._last_activity = 0.0
        self._first_message = True

    def _build_initial_prompt(self, ontology_list: str) -> str:
        """Build the initial prompt for the agent, mirroring CLI behavior."""
        if self.data_paths:
            resolved_paths = [str(Path(p).resolve()) for p in self.data_paths]
            paths_list = "\n".join(f"  - {p}" for p in resolved_paths)

            # Derive config_path if not provided
            if self.config_path is None and self.output_dir is not None:
                first = Path(resolved_paths[0])
                dir_name = first.name if first.is_dir() else first.parent.name
                self.config_path = str(Path(self.output_dir) / f"{dir_name}.yaml")

            prompt_parts = ["I need help setting up a new Talk-to-Data project.\n"]
            prompt_parts.append(f"**Source data paths**:\n{paths_list}\n")
            if self.output_dir:
                prompt_parts.append(f"**Output directory**: {self.output_dir}\n")
            else:
                prompt_parts.append("**Output directory**: _(please ask me)_\n")
            if self.config_path:
                prompt_parts.append(f"**Config file path**: {self.config_path}\n")
            else:
                prompt_parts.append(
                    "**Config file path**: _(default to `<output_dir>/<project_name>.yaml`)_\n"
                )
            prompt_parts.append(f"\n**Available ontologies**:\n{ontology_list}\n\n")
            prompt_parts.append(
                "IMPORTANT: Before exploring any data files, first complete Stage 1 "
                "(Project Basics). Ask me for any missing information — project name, "
                "output directory, etc. — and confirm everything with me before "
                "proceeding to data exploration. Do NOT use any tools until you have "
                "asked me your Stage 1 questions."
            )
            return "\n".join(prompt_parts)
        else:
            return (
                "I need help setting up a new Talk-to-Data project.\n\n"
                "I haven't specified any paths yet. Please start by asking me for:\n"
                "1. The paths to my source data files and/or directories\n"
                "2. Where I'd like the pipeline output directory to be\n"
                "3. Where to save the generated config YAML file\n\n"
                f"**Available ontologies**:\n{ontology_list}\n\n"
                "Then walk me through the rest of the setup process step by step."
            )

    async def connect(self) -> None:
        """Connect to the agent backend (Claude SDK or Ollama)."""
        import time

        from ...agents.prompts import SETUP_SYSTEM_PROMPT
        from ...ontologies.registry import OntologyRegistry

        # Build ontology table for the system prompt
        ontology_table_lines = ["| ID | Name | Description |", "|---|---|---|"]
        for ont in OntologyRegistry.get_all():
            ontology_table_lines.append(
                f"| {ont.ontology_id} | {ont.display_name} | {ont.description} |"
            )
        ontology_table = "\n".join(ontology_table_lines)

        # Build ontology list for the initial prompt
        ontology_list_parts = []
        for ont in OntologyRegistry.get_all():
            ontology_list_parts.append(
                f"  - {ont.ontology_id}: {ont.display_name} -- {ont.description}"
            )
        self._ontology_list = "\n".join(ontology_list_parts)

        system_prompt = SETUP_SYSTEM_PROMPT.format(ontology_table=ontology_table)

        if self.provider == "ollama":
            from ...agents.ollama_client import OllamaAgentClient

            self.client = OllamaAgentClient(
                model=self.model or "llama3.1",
                base_url=f"{self.ollama_base_url.rstrip('/')}/v1",
                system_prompt=system_prompt,
                cwd=str(Path.cwd()),
                max_turns=80,
            )
        else:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            env = {**os.environ, "CLAUDECODE": ""}

            options = ClaudeAgentOptions(
                model="claude-opus-4-6",
                cwd=str(Path.cwd()),
                allowed_tools=("Read", "Write", "Edit", "Glob", "Grep", "Bash"),
                system_prompt=system_prompt,
                max_turns=80,
                max_budget_usd=self.max_budget_usd,
                permission_mode="acceptEdits",
                env=env,
            )

            self.client = ClaudeSDKClient(options=options)

        await self.client.connect()
        self._reader_task = asyncio.create_task(self._message_reader())
        self._connected = True
        self._last_activity = time.time()

    async def _message_reader(self) -> None:
        """Background: read messages from SDK into queue."""
        try:
            async for message in self.client.receive_messages():
                await self._msg_queue.put(message)
        except Exception as exc:
            logger.debug("Message reader ended: %s", exc)
        await self._msg_queue.put(self._STREAM_END)

    async def _drain_stale(self) -> None:
        """Consume any messages that accumulated while waiting for input."""
        if self._msg_queue.empty():
            return
        discarded = 0
        while True:
            try:
                msg = await asyncio.wait_for(self._msg_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                break
            if msg is self._STREAM_END:
                await self._msg_queue.put(self._STREAM_END)
                break
            discarded += 1
        if discarded:
            logger.debug("Drained %d stale messages", discarded)

    async def send_and_stream(self, user_message: str):
        """Send user input, yield SSE events until the agent pauses."""
        import time

        if self.provider == "ollama":
            from ...agents.ollama_client import (
                AssistantMessage,
                ResultMessage,
                TextBlock,
                ToolUseBlock,
                UserMessage,
            )
        else:
            from claude_agent_sdk.types import (
                AssistantMessage,
                ResultMessage,
                TextBlock,
                ToolUseBlock,
                UserMessage,
            )

        self._last_activity = time.time()
        await self._drain_stale()

        # On the first message, replace the generic trigger with a rich
        # initial prompt that includes any paths the user already provided.
        if self._first_message:
            self._first_message = False
            user_message = self._build_initial_prompt(self._ontology_list)

        await self.client.query(user_message)

        while True:
            # Use a timeout so we can send keepalive comments and prevent
            # the SSE connection from being dropped during long tool calls
            # (e.g. data exploration of many files).
            try:
                msg = await asyncio.wait_for(
                    self._msg_queue.get(), timeout=_KEEPALIVE_INTERVAL
                )
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            # Debug: print every message type and its structure
            print(f"[SETUP-DEBUG] msg type: {type(msg).__name__}", flush=True)
            try:
                if hasattr(msg, '__dict__'):
                    # Print field names and types (not full values to avoid huge output)
                    fields = {k: type(v).__name__ for k, v in msg.__dict__.items()}
                    print(f"[SETUP-DEBUG]   fields: {fields}", flush=True)
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, list):
                        for i, block in enumerate(msg.content):
                            print(f"[SETUP-DEBUG]   content[{i}]: {type(block).__name__}", flush=True)
                            if hasattr(block, '__dict__'):
                                bfields = {k: type(v).__name__ for k, v in block.__dict__.items()}
                                print(f"[SETUP-DEBUG]     block fields: {bfields}", flush=True)
                    elif isinstance(msg.content, str):
                        print(f"[SETUP-DEBUG]   content (str): {msg.content[:100]}", flush=True)
            except Exception as dbg_exc:
                print(f"[SETUP-DEBUG]   debug print error: {dbg_exc}", flush=True)

            if msg is self._STREAM_END:
                yield _sse_event("done", "{}")
                break

            try:
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            yield _sse_event(
                                "text", json.dumps({"text": block.text})
                            )
                        elif isinstance(block, ToolUseBlock):
                            print(f"[SETUP-DEBUG] ToolUseBlock: name={block.name}, input type={type(block.input).__name__}", flush=True)
                            yield _sse_event(
                                "tool_call",
                                json.dumps({
                                    "tool": block.name,
                                    "input": _summarize_tool_input(block.input) if isinstance(block.input, dict) else str(block.input)[:120],
                                }),
                            )
                elif isinstance(msg, UserMessage):
                    print(f"[SETUP-DEBUG] UserMessage: tool_use_result type={type(msg.tool_use_result).__name__ if msg.tool_use_result is not None else 'None'}", flush=True)
                    if msg.tool_use_result:
                        result = msg.tool_use_result
                        if isinstance(result, dict):
                            content = result.get("content", "")
                        else:
                            content = result
                        yield _sse_event(
                            "tool_result",
                            json.dumps({
                                "tool": "tool",
                                "result": _truncate(str(content), 200),
                            }),
                        )
                elif isinstance(msg, ResultMessage):
                    yield _sse_event("done", "{}")
                    break
            except Exception as exc:
                print(f"[SETUP-DEBUG] ERROR processing msg: {exc}", flush=True)
                print(f"[SETUP-DEBUG]   msg repr: {repr(msg)[:500]}", flush=True)
                import traceback
                traceback.print_exc()
                yield _sse_event("error", json.dumps({"error": str(exc)}))
                yield _sse_event("done", "{}")
                break

    async def disconnect(self) -> None:
        """Disconnect the agent."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self.client:
            await self.client.disconnect()
        self._connected = False

    def get_config_yaml(self) -> str:
        """Read the current config YAML from disk."""
        if self.config_path and Path(self.config_path).exists():
            return Path(self.config_path).read_text()
        # Try to find any YAML in the output directory
        if self.output_dir:
            for f in Path(self.output_dir).glob("*.yaml"):
                return f.read_text()
        return ""


@router.post("/start")
async def start_setup(req: StartSetupRequest) -> JSONResponse:
    """Start a new setup agent session."""
    if len(_sessions) >= MAX_CONCURRENT_SESSIONS:
        raise HTTPException(
            429, "Maximum concurrent setup sessions reached"
        )

    session_id = str(uuid.uuid4())[:8]
    session = SetupAgentSession(
        session_id=session_id,
        data_paths=req.data_paths,
        output_dir=req.output_dir,
        config_path=req.config_path,
        max_budget_usd=req.max_budget,
        provider=req.provider,
        model=req.model,
        ollama_base_url=req.ollama_base_url,
    )

    try:
        await session.connect()
    except Exception as exc:
        logger.error("Failed to start setup agent: %s", exc, exc_info=True)
        raise HTTPException(500, f"Failed to start setup agent: {exc}")

    _sessions[session_id] = session
    return JSONResponse({"session_id": session_id})


@router.post("/message")
async def send_message_by_body(request_body: dict[str, Any] = {}) -> StreamingResponse:
    """Send a message to the setup agent (session_id in body) and stream the response."""
    session_id = request_body.get("session_id", "")
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Setup session not found: {session_id}")

    message = request_body.get("message", "")

    async def event_generator():
        yield _sse_event("session", json.dumps({"session_id": session_id}))
        try:
            async for event in session.send_and_stream(message):
                yield event
        except Exception as exc:
            print(f"[SETUP-DEBUG] event_generator (by_body) exception: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            yield _sse_event("error", json.dumps({"error": str(exc)}))
            yield _sse_event("done", "{}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{session_id}/message")
async def send_message(session_id: str, request_body: dict[str, Any] = {}) -> StreamingResponse:
    """Send a message to the setup agent and stream the response."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Setup session not found")

    message = request_body.get("message", "")

    async def event_generator():
        try:
            async for event in session.send_and_stream(message):
                yield event
        except Exception as exc:
            print(f"[SETUP-DEBUG] event_generator (by_id) exception: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            yield _sse_event("error", json.dumps({"error": str(exc)}))
            yield _sse_event("done", "{}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{session_id}/config")
async def get_session_config(session_id: str) -> JSONResponse:
    """Get the current config being written by the setup agent."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Setup session not found")

    yaml_text = session.get_config_yaml()
    parsed = None
    if yaml_text:
        try:
            import yaml

            parsed = yaml.safe_load(yaml_text)
        except Exception:
            pass

    return JSONResponse({"yaml": yaml_text, "parsed": parsed})


@router.get("/ollama/models")
async def list_ollama_models(base_url: str = "http://localhost:11434") -> JSONResponse:
    """List available models on an Ollama server."""
    try:
        from ...agents.ollama_client import list_ollama_models as _list_models

        models = await _list_models(base_url)
        return JSONResponse({
            "available": True,
            "models": [
                {
                    "name": m.get("name", "unknown"),
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in models
            ],
        })
    except Exception as exc:
        return JSONResponse({
            "available": False,
            "models": [],
            "error": str(exc),
        })


@router.delete("/{session_id}")
async def stop_setup(session_id: str) -> JSONResponse:
    """Stop and clean up a setup agent session."""
    session = _sessions.pop(session_id, None)
    if session:
        await session.disconnect()
    return JSONResponse({"status": "ok"})
