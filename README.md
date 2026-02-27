# Talk-to-Data

A general-purpose framework for building agentic clinical dataset query systems.
Takes raw clinical data (structured tables + unstructured notes), extracts
structured information using ontology-driven LLM extraction, builds a
privacy-safe analytical database, and exposes it through an interactive chatbot.

## Features

- **Interactive setup**: Agent-guided project configuration that explores your data, identifies columns, finds demographics, and proposes a database schema
- **Ontology-driven extraction**: Define what to extract using pluggable ontology schemas (NAACCR, PRISSMM, OMOP, MatchMiner-AI, MSK-CHORD, Pan-TOP, Generic Cancer)
- **Chunked extraction**: Process long patient note histories with iterative LLM extraction and checkpointing
- **Structured harmonization**: Map existing structured dataset columns to ontology fields
- **Date de-identification**: All dates are converted to intervals since birth (years) and calendar years; raw dates are removed from the final database
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

## Getting Started

The recommended way to start a new project is with the **setup agent**. It
walks you through an interactive conversation to configure everything:

### 1. Run the setup agent

```bash
uv run talk-to-data setup /path/to/your/data
```

You can point it at one or more files or directories:

```bash
uv run talk-to-data setup /path/to/patients.csv /path/to/notes_dir/ /path/to/labs.parquet
```

Optional flags:

```bash
uv run talk-to-data setup /path/to/data \
    --output-dir ./my_output \          # Where pipeline outputs go (default: ./output)
    --config configs/my_project.yaml \  # Where to save the YAML config
    --max-budget 10.0                   # Max agent budget in USD (default: 10.0)
```

The setup agent will:

1. **Explore your source files** — scan all CSVs and parquets to understand what data you have
2. **Identify data types** — distinguish patient rosters, clinical notes, structured data, etc.
3. **Collect cohort criteria** — ask about patient ID columns, diagnosis code filters, and inclusion criteria
4. **Find demographics** — search *all* source files for demographic columns (sex, race, ethnicity, birth date, death date), even if they're in a different file than the patient roster
5. **Configure extraction** — set up the LLM backend, select ontologies, and configure chunking parameters
6. **Discover field mappings** — identify which structured data columns map to ontology fields
7. **Propose a database schema** — show you the tables that will be created, including how dates will be de-identified (converted to `*_years_since_birth` intervals and `*_calendar_year` integers)
8. **Write the config** — generate a complete YAML config file ready for the pipeline

### 2. Run the pipeline

Once you have a config file (from setup or by editing `configs/example_project.yaml`):

```bash
# Run all stages
uv run talk-to-data pipeline configs/my_project.yaml

# Or run specific stages
uv run talk-to-data pipeline configs/my_project.yaml --stages cohort prepare_notes extract harmonize propose_tables database metadata

# Resume extraction from checkpoint (if previously interrupted)
uv run talk-to-data pipeline configs/my_project.yaml --stages extract --resume
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
| `cohort` | Define patient cohort from roster + optional diagnosis filter + optional demographics file | Source CSVs/parquets | `cohort.parquet`, `cohort_ids.json` |
| `prepare_notes` | Prepare clinical notes for extraction | Notes files | Filtered notes |
| `extract` | Extract structured data from clinical notes via LLM | Notes + ontology schemas | Extraction shards (parquet) |
| `harmonize` | Map structured data columns to ontology fields | Source files + field mappings | Harmonized parquets |
| `propose_tables` | Preview the database schema that will be created | Cohort + extractions + harmonized | Schema preview (displayed to user) |
| `database` | Build DuckDB with de-identified IDs and dates | All of the above | `.duckdb` file |
| `metadata` | Generate schema and summary docs from database | DuckDB | `schema.md`, `summary.md` |

## Field Discovery

Use the discovery agent to interactively explore your data and create field mappings:

```bash
uv run talk-to-data discover /path/to/data --ontologies naaccr pan_top
```

The agent will explore your CSV/parquet files, identify relevant columns, and
suggest ontology field mappings you can add to your project config.

## Available Ontologies

| ID | Description |
|---|---|
| `naaccr` | NAACCR v25 cancer registry fields |
| `prissmm` | PRISSMM/GENIE BPC clinical data model |
| `omop` | OMOP Common Data Model (oncology extension) |
| `matchminer_ai` | MatchMiner-AI clinical trial matching concepts |
| `msk_chord` | MSK-CHORD oncology data model |
| `pan_top` | Pan-TOP thoracic oncology data extraction (cancer diagnosis, biomarkers, systemic therapy, surgery, radiation, burden, smoking history) |
| `generic_cancer` | Cancer-type-agnostic extraction covering diagnosis, biomarkers, systemic therapy, surgery, radiation, burden, and social history across all cancer types |

## Project Configuration

See `configs/example_project.yaml` for a complete configuration reference.

Key sections:

- `project` — Name, input paths (list of files/directories), output directory
- `cohort` — Patient roster file, optional diagnosis file, optional demographics file, ID and demographic column names, diagnosis code filters, followup date
- `extraction` — LLM backend (vLLM/Claude/Vertex), ontology IDs, notes paths, chunking parameters
- `database` — Date de-identification (`deidentify_dates: true`), column filtering, ID prefix
- `query` — MCP server host/port, cell suppression threshold, output size limits
- `chatbot` — Web interface LLM settings, MCP connection
- `field_mappings` — Structured data column-to-ontology mappings (can be generated by the discovery agent)

### Demographics handling

The cohort builder supports demographics from three sources:

1. **In the patient roster file** — set `sex_column`, `race_column`, `birth_date_column`, etc. in the `cohort` section
2. **In a separate demographics file** — set `cohort.demographics_file` to the filename; the pipeline will merge it with the patient roster by matching on `patient_id_column`
3. **Extracted from notes** — the extraction stage can extract demographic information from clinical notes if an appropriate ontology is configured

The setup agent automatically searches all source files for demographic columns and configures the correct option.

### Date de-identification

When `database.deidentify_dates` is `true` (the default), all date columns in the database are converted to:

- `{column}_years_since_birth` — float, years between patient's birth date and the event date
- `{column}_calendar_year` — integer, just the year component

The original date columns and `birth_date` are removed from the final database. This requires `birth_date_column` to be configured in the cohort section.

## Architecture

```
Source Data -> [Setup Agent] -> YAML Config
                                    |
                                    v
Source Data -> [Cohort] -> [Prepare Notes] -> [Extract] -> [Harmonize]
                  |                                            |
                  v                                            v
            cohort.parquet                           harmonized/*.parquet
            cohort_ids.json                                    |
                  |               [Propose Tables]             |
                  |                (schema preview)            |
                  +--------------------+----------------------+
                                       |
                                       v
                               [Database Builder]
                           (de-identify IDs + dates)
                                       |
                                       v
                                 project.duckdb
                                       |
                                       v
                              [MCP Server] (SQL validation + cell suppression)
                                       |
                                       v
                              [Chatbot] (web UI with SSE streaming)
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
