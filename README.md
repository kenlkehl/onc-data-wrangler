# Talk-to-Data

A general-purpose framework for building agentic clinical dataset query systems.
Takes raw clinical data (structured tables + unstructured notes), extracts
structured information using ontology-driven LLM extraction, builds a
privacy-safe analytical database, and exposes it through an interactive chatbot.

## Features

- **Ontology-driven extraction**: Define what to extract using pluggable ontology schemas (NAACCR, PRISSMM, OMOP, MatchMiner-AI, MSK-CHORD)
- **Chunked extraction**: Process long patient note histories with iterative LLM extraction and checkpointing
- **Structured harmonization**: Map existing structured dataset columns to ontology fields
- **Privacy-safe queries**: SQL validation, cell suppression, and output sanitization via MCP server
- **Interactive chatbot**: Web-based chat interface with agentic analysis (SSE streaming)
- **Agent-assisted discovery**: Use Claude Agent SDK to interactively explore data and map fields

## Installation

```bash
# Clone and install with uv
git clone <repo-url> talk_to_data
cd talk_to_data
uv sync
```

## Quick Start

### 1. Create a project config

```bash
cp configs/example_project.yaml configs/my_project.yaml
# Edit paths and settings
```

### 2. Run the pipeline

```bash
# Run all stages
uv run talk-to-data pipeline configs/my_project.yaml

# Or run specific stages
uv run talk-to-data pipeline configs/my_project.yaml --stages database metadata
```

### 3. Start the query server

```bash
uv run talk-to-data serve configs/my_project.yaml
```

### 4. Start the chatbot

```bash
uv run talk-to-data chat configs/my_project.yaml
# Open http://localhost:8080 in your browser
```

## Pipeline Stages

| Stage | Description | Input | Output |
|---|---|---|---|
| `cohort` | Define patient cohort | Source CSVs/parquets | `cohort.parquet` |
| `extract` | Extract from clinical notes | Notes file | Extraction shards |
| `harmonize` | Map structured columns | Source files + field mappings | Harmonized parquets |
| `database` | Build DuckDB | Cohort + extractions + harmonized | `.duckdb` file |
| `metadata` | Generate schema/summary | DuckDB database | `schema.md`, `summary.md` |

## Field Discovery

Use the discovery agent to interactively explore your data and create field mappings:

```bash
uv run talk-to-data discover /path/to/data --ontologies naaccr prissmm
```

The agent will explore your CSV/parquet files, identify relevant columns, and
suggest ontology field mappings you can add to your project config.

## Available Ontologies

- **naaccr**: NAACCR v25 cancer registry fields
- **prissmm**: PRISSMM/GENIE BPC clinical data model
- **omop**: OMOP Common Data Model (oncology extension)
- **matchminer_ai**: MatchMiner-AI clinical trial matching
- **msk_chord**: MSK-CHORD oncology data model

## Project Configuration

See `configs/example_project.yaml` for a complete configuration reference.

Key sections:
- `project`: Name, input/output directories
- `extraction`: LLM backend, ontologies, chunking parameters
- `database`: De-identification, column filtering
- `query`: MCP server privacy settings
- `chatbot`: Web interface and LLM settings
- `field_mappings`: Structured data column mappings

## Architecture

```
Source Data -> [Extraction] -> [Harmonization] -> [Database] -> [MCP Server] -> [Chatbot]
     |              |                |                              |
     |         Uses ontology    Uses field           SQL validation +
     |         schemas for      mappings for         cell suppression
     |         LLM prompts      column renaming      for privacy
     |
     +-- [Discovery Agent] -- Interactive field exploration
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/

# Run a single module
uv run python -m talk_to_data.query.mcp_server configs/my_project.yaml
```
