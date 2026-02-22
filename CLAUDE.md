# Talk-to-Data Developer Guide

## Project Structure

- `src/talk_to_data/` — Main package
  - `config.py` — YAML configuration dataclasses
  - `cli.py` — CLI entry point with subcommands
  - `llm/` — LLM backend abstraction (vLLM, Claude)
  - `ontologies/` — Pluggable ontology system with builtins
  - `extraction/` — Ontology-driven LLM extraction from notes
  - `harmonization/` — Structured data column mapping
  - `cohort/` — Cohort definition from structured tables
  - `database/` — DuckDB creation and metadata generation
  - `query/` — SQL validation, privacy, MCP server
  - `web/` — FastAPI chatbot with SSE streaming
  - `agents/` — Claude Agent SDK orchestration
- `configs/` — Example YAML project configs
- `tests/` — Test suite

## Commands

```bash
uv sync                              # Install dependencies
uv run pytest tests/                 # Run tests
uv run talk-to-data --help           # CLI help
uv run talk-to-data pipeline <cfg>   # Run pipeline
uv run talk-to-data serve <cfg>      # Start MCP server
uv run talk-to-data chat <cfg>       # Start chatbot
```

## Key Patterns

- Config is always a `ProjectConfig` loaded from YAML
- LLM calls go through `LLMClient` ABC (vLLM or Claude backends)
- Ontologies self-register via `@register_ontology` decorator
- Privacy enforced via SQL validation + cell suppression in MCP server
- All patient IDs are de-identified before database creation
