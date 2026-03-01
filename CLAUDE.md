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
  - `training/` — GRPO fine-tuning for clinical summary models
    - `grpo_trainer.py` — TRL GRPOTrainer integration
    - `reward.py` — Reward function (structured extraction F1)
    - `silver_labels.py` — Silver-standard label generation
    - `dataset.py` — Training dataset builder
  - `agents/` — Claude Agent SDK orchestration
    - `setup.py` — Interactive setup agent (creates project config)
    - `pipeline.py` — Pipeline orchestration (runs stages)
    - `discovery.py` — Field discovery agent (maps columns to ontologies)
    - `prompts.py` — System prompts for agents
- `configs/` — Example YAML project configs
- `tests/` — Test suite

## Commands

```bash
uv sync                              # Install dependencies
uv run pytest tests/                 # Run tests
uv run talk-to-data --help           # CLI help
uv run talk-to-data setup /path/to/data  # Interactive project setup
uv run talk-to-data pipeline <cfg>   # Run pipeline
uv run talk-to-data pipeline <cfg> --stages cohort extract  # Run specific stages
uv run talk-to-data serve <cfg>      # Start MCP server
uv run talk-to-data chat <cfg>       # Start chatbot
uv run talk-to-data discover /path/to/data  # Interactive field discovery
uv run talk-to-data finetune <cfg>       # GRPO fine-tune summary model
```

## Key Patterns

- Config is always a `ProjectConfig` loaded from YAML
- LLM calls go through `LLMClient` ABC (vLLM or Claude backends)
- Ontologies self-register via `@register_ontology` decorator
- Privacy enforced via SQL validation + cell suppression in MCP server
- All patient IDs are de-identified before database creation
- Dates are de-identified by conversion to intervals since birth (`*_years_since_birth` float, `*_calendar_year` integer); raw dates and birth_date are excluded from the final database

## Data Flow

1. **Setup agent** interacts with user to discover source files, identify columns, configure cohort, and propose a database schema
2. **Cohort stage** builds patient roster from patient file + optional diagnosis file + optional demographics file (can be a separate file from the patient roster)
3. **Extraction stage** uses ontology-driven LLM prompts to extract structured data from clinical notes
4. **Harmonization stage** maps structured data columns to ontology fields using field mappings
5. **Database stage** builds DuckDB with de-identified IDs and de-identified dates; birth_date is preserved in cohort.parquet for downstream date conversion but excluded from the final DB
6. **MCP server** exposes the database with SQL validation and privacy enforcement
7. **Chatbot** provides an agentic web interface that queries via the MCP server

## ID De-identification Flow

The `CohortBuilder` de-identifies patient IDs when building the cohort and saves the original IDs to `cohort_ids.json`. The `DatabaseBuilder` reconstructs this same mapping from `cohort_ids.json` to apply consistent de-identification to extraction and harmonized data. The cohort table is already de-identified so it is not re-mapped.

## Pipeline Stages

Valid stages (for `--stages` flag): `cohort`, `prepare_notes`, `extract`, `harmonize`, `propose_tables`, `database`, `metadata`

## Available Ontologies

naaccr, prissmm, omop, matchminer_ai, msk_chord, pan_top, generic_cancer, clinical_summary
