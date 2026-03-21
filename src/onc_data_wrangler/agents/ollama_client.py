"""Ollama-based agent client using OpenAI-compatible API with tool calling.

Provides an alternative to the Claude Agent SDK for the setup agent,
allowing users to run the agentic setup process against a local LLM
served by Ollama.
"""

import asyncio
import glob as glob_module
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message types (matching claude_agent_sdk.types attribute interface)
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    name: str
    input: dict


@dataclass
class AssistantMessage:
    content: list  # list[TextBlock | ToolUseBlock]


@dataclass
class UserMessage:
    tool_use_result: str | None = None


@dataclass
class ResultMessage:
    result: str | None = None
    num_turns: int = 0
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file from the filesystem. Returns numbered lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-based). Default: 1",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Default: 2000",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Edit a file by replacing an exact string match with new text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find and replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files matching a glob pattern. Returns matching paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.csv', '*.parquet')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to current working directory.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search file contents for a regex pattern. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in. Defaults to current working directory.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py', '*.csv')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command and return stdout + stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds. Default: 120",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

# Max characters to return from a single tool call
_MAX_TOOL_RESULT_CHARS = 50_000


def _truncate_result(text: str) -> str:
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text
    return text[:_MAX_TOOL_RESULT_CHARS] + f"\n... (truncated, {len(text)} total chars)"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _tool_read(file_path: str, offset: int = 1, limit: int = 2000, **_) -> str:
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    if path.is_dir():
        return f"Error: {file_path} is a directory, not a file"
    try:
        text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {file_path}: {e}"
    lines = text.splitlines()
    start = max(0, offset - 1)
    end = start + limit
    numbered = []
    for i, line in enumerate(lines[start:end], start=start + 1):
        numbered.append(f"{i:>6}\t{line}")
    return "\n".join(numbered) if numbered else "(empty file)"


async def _tool_write(file_path: str, content: str, **_) -> str:
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, content)
        return f"File written: {file_path} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing {file_path}: {e}"


async def _tool_edit(file_path: str, old_string: str, new_string: str, **_) -> str:
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    try:
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except Exception as e:
        return f"Error reading {file_path}: {e}"
    if old_string not in text:
        return f"Error: old_string not found in {file_path}"
    count = text.count(old_string)
    if count > 1:
        return f"Error: old_string found {count} times in {file_path}. Provide more context to make it unique."
    new_text = text.replace(old_string, new_string, 1)
    try:
        await asyncio.to_thread(path.write_text, new_text)
        return f"File edited: {file_path}"
    except Exception as e:
        return f"Error writing {file_path}: {e}"


async def _tool_glob(pattern: str, path: str | None = None, **_) -> str:
    base = Path(path) if path else Path.cwd()
    try:
        matches = sorted(str(p) for p in base.glob(pattern))
    except Exception as e:
        return f"Error: {e}"
    if not matches:
        return f"No files matching '{pattern}' in {base}"
    return "\n".join(matches[:500])


async def _tool_grep(
    pattern: str, path: str | None = None, glob: str | None = None, **_
) -> str:
    cmd = ["grep", "-rn", "-E"]
    if glob:
        cmd.extend(["--include", glob])
    cmd.append(pattern)
    cmd.append(path or ".")
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if not output:
            return f"No matches for pattern '{pattern}'"
        return output
    except subprocess.TimeoutExpired:
        return "Error: grep timed out"
    except Exception as e:
        return f"Error: {e}"


async def _tool_bash(command: str, timeout: int = 120, cwd: str = ".", **_) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: command timed out after {timeout}s"
        output = ""
        if stdout:
            output += stdout.decode(errors="replace")
        if stderr:
            output += ("\n--- stderr ---\n" if output else "") + stderr.decode(
                errors="replace"
            )
        if proc.returncode != 0:
            output += f"\n(exit code: {proc.returncode})"
        return output or "(no output)"
    except Exception as e:
        return f"Error executing command: {e}"


_TOOL_DISPATCH = {
    "Read": _tool_read,
    "Write": _tool_write,
    "Edit": _tool_edit,
    "Glob": _tool_glob,
    "Grep": _tool_grep,
    "Bash": _tool_bash,
}


# ---------------------------------------------------------------------------
# OllamaAgentClient
# ---------------------------------------------------------------------------


class OllamaAgentClient:
    """Agent client that uses Ollama's OpenAI-compatible API with tool calling.

    Implements the same external interface as ClaudeSDKClient:
    - connect() / disconnect()
    - query(msg) to send a user message
    - receive_messages() async generator yielding message objects
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        system_prompt: str = "",
        cwd: str = ".",
        allowed_tools: tuple[str, ...] = ("Read", "Write", "Edit", "Glob", "Grep", "Bash"),
        max_turns: int = 80,
    ):
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.cwd = cwd
        self.allowed_tools = allowed_tools
        self.max_turns = max_turns

        self._client: AsyncOpenAI | None = None
        self._conversation: list[dict[str, Any]] = []
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._STREAM_END = object()
        self._turn_count = 0
        self._start_time = 0.0
        self._run_task: asyncio.Task | None = None

        # Filter tool schemas to allowed tools only
        self._tool_schemas = [
            t for t in TOOL_SCHEMAS if t["function"]["name"] in allowed_tools
        ]

    async def connect(self) -> None:
        """Initialize the client and verify Ollama is reachable."""
        self._client = AsyncOpenAI(base_url=self.base_url, api_key="ollama")
        self._start_time = time.time()

        # Verify connectivity by listing models
        try:
            models_response = await self._client.models.list()
            available = [m.id for m in models_response.data]
            if self.model not in available:
                # Ollama may list models with/without tags -- try a fuzzy match
                base_name = self.model.split(":")[0]
                fuzzy = [m for m in available if m.startswith(base_name)]
                if not fuzzy:
                    logger.warning(
                        "Model '%s' not found in Ollama. Available: %s. "
                        "Proceeding anyway (Ollama may pull it on first use).",
                        self.model,
                        available,
                    )
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running (`ollama serve`). Error: {exc}"
            ) from exc

        logger.info("Connected to Ollama at %s, model=%s", self.base_url, self.model)

    async def disconnect(self) -> None:
        """Clean up."""
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
        self._client = None

    async def query(self, message: str) -> None:
        """Send a user message and start processing the response.

        The response (including any tool-calling loops) runs as a background
        task. Messages are emitted to the outbox queue and consumed via
        receive_messages().
        """
        self._conversation.append({"role": "user", "content": message})
        self._run_task = asyncio.create_task(self._run_turn())

    async def receive_messages(self) -> AsyncIterator:
        """Yield messages from the outbox queue."""
        while True:
            msg = await self._outbox.get()
            if msg is self._STREAM_END:
                return
            yield msg

    async def _run_turn(self) -> None:
        """Execute one agent turn: call the model, handle tool calls, repeat."""
        self._turn_count += 1
        tool_call_iterations = 0
        max_tool_iterations = 30  # safety limit per turn

        try:
            while tool_call_iterations < max_tool_iterations:
                tool_call_iterations += 1

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    *self._conversation,
                ]

                try:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self._tool_schemas if self._tool_schemas else None,
                        temperature=0.0,
                    )
                except Exception as exc:
                    error_text = f"Error calling Ollama: {exc}"
                    logger.error(error_text)
                    await self._outbox.put(
                        AssistantMessage(content=[TextBlock(text=error_text)])
                    )
                    await self._outbox.put(
                        ResultMessage(
                            result=error_text,
                            num_turns=self._turn_count,
                            duration_ms=int((time.time() - self._start_time) * 1000),
                        )
                    )
                    return

                choice = response.choices[0]
                assistant_msg = choice.message

                # Emit any text content
                if assistant_msg.content:
                    await self._outbox.put(
                        AssistantMessage(content=[TextBlock(text=assistant_msg.content)])
                    )

                # Check for tool calls
                if assistant_msg.tool_calls:
                    # Add assistant message to conversation history (with tool calls)
                    self._conversation.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in assistant_msg.tool_calls
                            ],
                        }
                    )

                    # Execute each tool call
                    for tc in assistant_msg.tool_calls:
                        tool_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                            logger.warning(
                                "Failed to parse tool arguments: %s",
                                tc.function.arguments,
                            )

                        # Emit tool use block
                        await self._outbox.put(
                            AssistantMessage(
                                content=[ToolUseBlock(name=tool_name, input=args)]
                            )
                        )

                        # Execute tool
                        tool_fn = _TOOL_DISPATCH.get(tool_name)
                        if tool_fn is None:
                            result_text = f"Error: Unknown tool '{tool_name}'"
                        else:
                            try:
                                if tool_name == "Bash":
                                    args.setdefault("cwd", self.cwd)
                                result_text = await tool_fn(**args)
                                result_text = _truncate_result(result_text)
                            except Exception as exc:
                                result_text = f"Error executing {tool_name}: {exc}"

                        # Emit tool result
                        await self._outbox.put(
                            UserMessage(tool_use_result=result_text)
                        )

                        # Add tool result to conversation history
                        self._conversation.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_text,
                            }
                        )

                    # Continue loop — model needs to respond to tool results

                else:
                    # No tool calls — add response to history and finish turn
                    self._conversation.append(
                        {"role": "assistant", "content": assistant_msg.content or ""}
                    )
                    await self._outbox.put(
                        ResultMessage(
                            result=assistant_msg.content,
                            num_turns=self._turn_count,
                            duration_ms=int(
                                (time.time() - self._start_time) * 1000
                            ),
                        )
                    )
                    return

            # Exceeded max tool iterations
            warning = (
                "Reached maximum tool call iterations for this turn. "
                "Please continue with your next instruction."
            )
            await self._outbox.put(
                AssistantMessage(content=[TextBlock(text=warning)])
            )
            await self._outbox.put(
                ResultMessage(
                    result=warning,
                    num_turns=self._turn_count,
                    duration_ms=int((time.time() - self._start_time) * 1000),
                )
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Unexpected error in _run_turn: %s", exc, exc_info=True)
            error_text = f"Agent error: {exc}"
            await self._outbox.put(
                AssistantMessage(content=[TextBlock(text=error_text)])
            )
            await self._outbox.put(
                ResultMessage(
                    result=error_text,
                    num_turns=self._turn_count,
                    duration_ms=int((time.time() - self._start_time) * 1000),
                )
            )


# ---------------------------------------------------------------------------
# Utility: list available Ollama models
# ---------------------------------------------------------------------------


async def list_ollama_models(
    base_url: str = "http://localhost:11434",
) -> list[dict[str, Any]]:
    """Query Ollama for available models.

    Uses Ollama's native /api/tags endpoint (not the OpenAI compat layer)
    to get richer model metadata including size.

    Returns a list of dicts with keys: name, size, modified_at, details.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
    except Exception as exc:
        logger.debug("Failed to list Ollama models at %s: %s", url, exc)
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. "
            f"Ensure Ollama is running (`ollama serve`). Error: {exc}"
        ) from exc
