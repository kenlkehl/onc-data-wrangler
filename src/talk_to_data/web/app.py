"""FastAPI web chatbot with SSE streaming and MCP tool proxy.

Provides a chat interface that uses Claude (Anthropic or Vertex) as the LLM,
proxies tool calls to the Talk-to-Data MCP server, and streams responses
back to the browser via Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from sse_starlette.sse import EventSourceResponse

from ..config import ProjectConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

ASK_USER_TOOL = {
    "name": "ask_user",
    "description": (
        "Ask the user a clarifying question before proceeding. "
        "Use this when you need more information to answer accurately."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user.",
            }
        },
        "required": ["question"],
    },
}


def _build_system_prompt(config: ProjectConfig) -> str:
    """Build the chatbot system prompt from template and config."""
    from ..agents.prompts import CHATBOT_SYSTEM_PROMPT_TEMPLATE

    schema_context = ""
    summary_context = ""

    schema_path = config.schema_path
    if schema_path.exists():
        schema_context = schema_path.read_text()

    summary_path = config.summary_path
    if summary_path.exists():
        summary_context = summary_path.read_text()

    return CHATBOT_SYSTEM_PROMPT_TEMPLATE.format(
        min_cell_size=config.query.min_cell_size,
        max_query_rows=config.query.max_query_rows,
        schema_context=schema_context,
        summary_context=summary_context,
    )


def _make_anthropic_client(config: ProjectConfig):
    """Create the appropriate Anthropic client based on provider config."""
    llm = config.chatbot.llm
    if llm.provider == "vertex":
        from anthropic import AnthropicVertex

        return AnthropicVertex(
            project_id=llm.resolve_vertex_project(),
            region=llm.vertex_region,
        )
    else:
        from anthropic import Anthropic

        api_key = llm.resolve_api_key()
        return Anthropic(api_key=api_key)


async def _list_mcp_tools(mcp_url: str, mcp_token: str) -> list[dict]:
    """Connect to the MCP server and retrieve the tool list."""
    headers = {}
    if mcp_token:
        headers["Authorization"] = f"Bearer {mcp_token}"

    async with streamablehttp_client(url=mcp_url, headers=headers) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for tool in result.tools:
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                    }
                )
            return tools


async def _call_mcp_tool(
    mcp_url: str, mcp_token: str, tool_name: str, tool_input: dict
) -> str:
    """Call a single tool on the MCP server and return the text result."""
    headers = {}
    if mcp_token:
        headers["Authorization"] = f"Bearer {mcp_token}"

    async with streamablehttp_client(url=mcp_url, headers=headers) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_input)
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            return "\n".join(parts)


# ---------------------------------------------------------------------------
# Transcript logging
# ---------------------------------------------------------------------------


class TranscriptLogger:
    """Logs each session's conversation to a JSON file in a transcripts folder."""

    def __init__(self, transcripts_dir: Path) -> None:
        self.transcripts_dir = transcripts_dir
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str, created_at: str) -> Path:
        """Return the transcript file path for a session."""
        # Use creation timestamp in the filename for easy sorting
        safe_ts = created_at.replace(":", "-")
        return self.transcripts_dir / f"{safe_ts}_{session_id}.json"

    def save(self, session_id: str, transcript: dict) -> None:
        """Write the full transcript dict to disk."""
        path = self._path_for(session_id, transcript["created_at"])
        try:
            path.write_text(json.dumps(transcript, indent=2, default=str))
        except Exception:
            logger.error("Failed to save transcript %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class ChatSession:
    """Per-session conversation state."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.pending_ask_user_id: str | None = None
        self.lock = asyncio.Lock()
        self.created_at: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self.transcript: list[dict[str, Any]] = []

    def log(self, entry: dict[str, Any]) -> None:
        """Append a timestamped entry to the session transcript."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.transcript.append(entry)


# ---------------------------------------------------------------------------
# Agent loop (shared between /chat and /answer)
# ---------------------------------------------------------------------------


async def _run_agent_loop(
    session: ChatSession,
    config: ProjectConfig,
    system_prompt: str,
    all_tools: list[dict],
) -> AsyncGenerator[dict, None]:
    """Run the agentic tool loop, yielding SSE event dicts.

    This is an async generator that calls the LLM, executes MCP tool calls,
    feeds results back, and repeats until the LLM produces a final text
    response or uses ask_user.
    """
    client = _make_anthropic_client(config)
    model = config.chatbot.llm.model
    max_tokens = config.chatbot.llm.max_tokens
    max_turns = config.chatbot.max_agent_turns

    for _turn in range(max_turns):
        # Call the LLM
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=session.messages,
                tools=all_tools,
            )
        except Exception as exc:
            logger.error("Anthropic API error: %s", exc, exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({"error": f"LLM API error: {exc}"}),
            }
            return

        # Separate text blocks and tool_use blocks
        assistant_content: list[dict[str, Any]] = []
        tool_use_blocks: list[Any] = []

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                session.log({"role": "assistant", "type": "text", "text": block.text})
                yield {
                    "event": "text",
                    "data": json.dumps({"text": block.text}),
                }
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
                tool_use_blocks.append(block)

        # Check for ask_user first (takes priority — pause agent loop)
        ask_user_block = None
        for block in tool_use_blocks:
            if block.name == "ask_user":
                ask_user_block = block
                break

        if ask_user_block is not None:
            question = ask_user_block.input.get("question", "")
            session.log({"role": "assistant", "type": "ask_user", "question": question})
            yield {
                "event": "ask_user",
                "data": json.dumps(
                    {
                        "tool_use_id": ask_user_block.id,
                        "question": question,
                    }
                ),
            }
            # Save assistant message and pause
            session.messages.append(
                {"role": "assistant", "content": assistant_content}
            )
            session.pending_ask_user_id = ask_user_block.id
            yield {"event": "done", "data": "{}"}
            return

        # No ask_user — append assistant message
        session.messages.append(
            {"role": "assistant", "content": assistant_content}
        )

        # If no tool calls at all, we are done
        if not tool_use_blocks:
            break

        # Execute MCP tool calls and build tool_result messages
        tool_result_contents: list[dict[str, Any]] = []
        for block in tool_use_blocks:
            session.log({"role": "assistant", "type": "tool_call", "tool": block.name, "input": block.input})
            yield {
                "event": "tool_call",
                "data": json.dumps(
                    {"tool": block.name, "input": block.input}
                ),
            }

            try:
                result_text = await _call_mcp_tool(
                    config.chatbot.mcp_url,
                    config.chatbot.mcp_token,
                    block.name,
                    block.input,
                )
            except Exception as exc:
                result_text = f"Tool error: {exc}"
                logger.error(
                    "MCP tool %s failed: %s", block.name, exc, exc_info=True
                )

            session.log({"role": "tool", "type": "tool_result", "tool": block.name, "result": result_text[:2000]})
            yield {
                "event": "tool_result",
                "data": json.dumps(
                    {"tool": block.name, "result": result_text[:2000]}
                ),
            }

            tool_result_contents.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )

        # Feed tool results back as a user message
        session.messages.append(
            {"role": "user", "content": tool_result_contents}
        )

        # Continue loop for next LLM turn

    yield {"event": "done", "data": "{}"}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app_from_config(config: ProjectConfig) -> FastAPI:
    """Create a FastAPI application wired to the project configuration.

    Args:
        config: The project configuration containing chatbot, query, and
            output directory settings.

    Returns:
        A FastAPI app ready to be served with uvicorn.
    """

    sessions: dict[str, ChatSession] = {}
    system_prompt: str = ""
    mcp_tools: list[dict] = []
    transcripts_dir = Path(config.output_dir) / "transcripts"
    transcript_logger = TranscriptLogger(transcripts_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal system_prompt, mcp_tools

        # Build the system prompt once at startup
        system_prompt = _build_system_prompt(config)
        logger.info("System prompt built (%d chars)", len(system_prompt))

        # Discover MCP tools
        try:
            mcp_tools = await _list_mcp_tools(
                config.chatbot.mcp_url, config.chatbot.mcp_token
            )
            logger.info("Discovered %d MCP tools", len(mcp_tools))
        except Exception:
            logger.warning(
                "Could not connect to MCP server at %s -- tool calls will fail",
                config.chatbot.mcp_url,
                exc_info=True,
            )
            mcp_tools = []

        yield

    app = FastAPI(title="Talk-to-Data Chatbot", lifespan=lifespan)

    # ---- Static files / UI --------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=html_path.read_text(), status_code=200)

    # Serve other static assets (CSS, JS, images) if any
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ---- Transcript persistence helper ---------------------------------------

    def _save_transcript(session_id: str, session: ChatSession) -> None:
        """Persist the current session transcript to disk."""
        transcript_logger.save(
            session_id,
            {
                "session_id": session_id,
                "created_at": session.created_at,
                "entries": session.transcript,
            },
        )

    # ---- Chat endpoint (SSE) ------------------------------------------------

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        user_message: str = body.get("message", "").strip()
        session_id: str = body.get("session_id", "")

        if not session_id or session_id not in sessions:
            session_id = str(uuid.uuid4())
            sessions[session_id] = ChatSession()

        session = sessions[session_id]

        async def event_generator():
            async with session.lock:
                # Log and append user message
                session.log({"role": "user", "type": "message", "text": user_message})
                session.messages.append(
                    {"role": "user", "content": user_message}
                )

                # Send session_id so client can persist it
                yield {
                    "event": "session",
                    "data": json.dumps({"session_id": session_id}),
                }

                all_tools = mcp_tools + [ASK_USER_TOOL]
                async for event in _run_agent_loop(
                    session, config, system_prompt, all_tools
                ):
                    yield event

                # Save transcript after each interaction completes
                _save_transcript(session_id, session)

        return EventSourceResponse(event_generator())

    # ---- Answer to ask_user -------------------------------------------------

    @app.post("/answer")
    async def answer(request: Request):
        """Handle the user's response to an ask_user tool call.

        The client sends the answer text and session_id.  We inject the
        tool_result into the conversation and re-enter the agent loop.
        """
        body = await request.json()
        answer_text: str = body.get("answer", "").strip()
        session_id: str = body.get("session_id", "")

        if not session_id or session_id not in sessions:
            return JSONResponse(
                {"error": "Invalid session"}, status_code=400
            )

        session = sessions[session_id]

        pending_id = session.pending_ask_user_id
        if not pending_id:
            return JSONResponse(
                {"error": "No pending ask_user"}, status_code=400
            )

        async def event_generator():
            async with session.lock:
                # Log and inject the tool result for the ask_user call
                session.log({"role": "user", "type": "answer", "text": answer_text})
                session.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": pending_id,
                                "content": answer_text,
                            }
                        ],
                    }
                )
                session.pending_ask_user_id = None

                all_tools = mcp_tools + [ASK_USER_TOOL]
                async for event in _run_agent_loop(
                    session, config, system_prompt, all_tools
                ):
                    yield event

                # Save transcript after each interaction completes
                _save_transcript(session_id, session)

        return EventSourceResponse(event_generator())

    # ---- Reset endpoint -----------------------------------------------------

    @app.post("/reset")
    async def reset(request: Request):
        body = await request.json()
        session_id = body.get("session_id", "")
        if session_id in sessions:
            session = sessions[session_id]
            if session.transcript:
                _save_transcript(session_id, session)
            del sessions[session_id]
        return JSONResponse({"status": "ok", "message": "Session reset."})

    # ---- Health check -------------------------------------------------------

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    return app
