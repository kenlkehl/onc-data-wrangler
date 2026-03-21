# Onc-Data-Wrangler

A general-purpose framework for building agentic clinical dataset query systems.
Takes raw clinical data (structured tables + unstructured notes), extracts
structured information using dictionary-driven LLM extraction, builds a
privacy-safe analytical database, and exposes it through an interactive chatbot.

## Features

- **Web UI**: Full-featured React frontend with agentic setup wizard, pipeline dashboard, visual config editor, data explorer, and chatbot — all accessible from `onc-data-wrangler ui`
- **Interactive setup**: Agent-guided project configuration that explores your data, identifies columns, finds demographics, and proposes a database schema (available via CLI or web UI)
- **Dictionary-driven extraction**: Each ontology loads field definitions from authoritative data dictionaries (NAACCR CSVs, OMOP vocabularies, MSK-CHORD CDM codebooks, PRISSMM BPC variable synopses). The LLM receives valid codes inline in prompts and returns per-field `{value, confidence, evidence}`. Extracted values are resolved against valid code tables using a 6-tier strategy (exact match, case-insensitive, description, fuzzy, numeric range, fallback).
- **Domain-group processing**: Extraction is organized into sequential domain groups with data dependencies (demographics before staging, since staging items depend on primary site and histology). For NAACCR, there are 7 hand-curated groups; for other ontologies, groups are auto-generated from their data category definitions.
- **Site-specific schemas**: After extracting demographics, the SchemaRegistry determines the cancer type from ICD-O-3 site+histology codes and dynamically adds site-specific data items (e.g., ER/PR/HER2 for breast, Gleason/PSA for prostate, ALK/EGFR/PD-L1 for lung). 22 cancer schemas supported.
- **NAACCR output**: Produces registry-submission-ready NAACCR XML, flat-file, and CSV output with proper NaaccrData>Patient>Tumor hierarchy
- **Audit trail and review queue**: Every extracted field retains its confidence score, supporting evidence quote, and source provenance. Items below configurable confidence thresholds are flagged for human review at four priority levels (CRITICAL, HIGH, MEDIUM, LOW).
- **Free-text clinical summaries**: Extract concise narrative summaries instead of (or alongside) structured JSON using the `clinical_summary` ontology
- **GRPO fine-tuning**: Fine-tune summary models with reinforcement learning (GRPO) so that summaries contain the information needed for accurate downstream structured extraction
- **Chunked extraction**: Process long patient note histories with round-based parallel extraction, iterative merging (higher confidence wins), and crash-safe checkpointing
- **Structured harmonization**: Map existing structured dataset columns to ontology fields
- **Date de-identification**: All dates are converted to intervals since birth (years) and calendar years; raw dates are removed from the final database
- **Privacy-safe queries**: SQL validation, cell suppression, and output sanitization via MCP server
- **Interactive chatbot**: Web-based chat interface with agentic analysis (SSE streaming)
- **Agent-assisted discovery**: Use Claude Agent SDK to interactively explore data and map fields

## Installation

```bash
# Clone and install with uv
git clone <repo-url> onc_data_wrangler
cd onc_data_wrangler
uv sync
```

## Getting Started

### Option A: Web UI (recommended)

The web UI provides a visual interface for the entire workflow — setup, pipeline, config editing, data exploration, and chat.

```bash
# Build the frontend (first time only)
cd src/onc_data_wrangler/web/frontend
npm install && npm run build
cd ../../../..

# Start the UI without a config (setup mode)
uv run onc-data-wrangler ui

# Or start with an existing config
uv run onc-data-wrangler ui configs/my_project.yaml

# Open http://localhost:8080/ui/ in your browser
```

The UI has five pages:

| Page | URL | Description |
|------|-----|-------------|
| **Setup** | `/ui/setup` | Chat with the AI setup agent to create a project config. The left panel is the chat; the right panel shows a 9-stage progress indicator and a live YAML preview that updates as the agent writes the config. |
| **Pipeline** | `/ui/pipeline` | Launch pipeline runs with stage selection. Watch real-time stage progress with status indicators and a live log viewer. |
| **Config** | `/ui/config` | Visual form editor for YAML configs. Tabbed sections for Project, Cohort, Extraction, Database, Query, Chatbot, and Field Mappings. Includes an ontology multi-selector and a field mapping table editor. |
| **Data** | `/ui/data` | Browse source data files and pipeline outputs. Paginated data table preview with column statistics (click any column header for non-null count, unique values, top values, or numeric stats). |
| **Chat** | `/ui/chat` | Query your database in natural language. The AI agent writes and executes SQL via the MCP server, with privacy enforcement. |

### Option B: CLI

#### 1. Run the setup agent

```bash
uv run onc-data-wrangler setup /path/to/your/data
```

You can point it at one or more files or directories:

```bash
uv run onc-data-wrangler setup /path/to/patients.csv /path/to/notes_dir/ /path/to/labs.parquet
```

Optional flags:

```bash
uv run onc-data-wrangler setup /path/to/data \
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

#### 2. Run the pipeline

Once you have a config file (from setup or by editing `configs/example_project.yaml`):

```bash
# Run all stages
uv run onc-data-wrangler pipeline configs/my_project.yaml

# Or run specific stages
uv run onc-data-wrangler pipeline configs/my_project.yaml --stages cohort prepare_notes extract harmonize propose_tables database metadata

# Resume extraction from checkpoint (if previously interrupted)
uv run onc-data-wrangler pipeline configs/my_project.yaml --stages extract --resume
```

### Option C: Standalone Extraction (no config needed)

For quick extraction from a single notes file without setting up a full project:

```bash
# Extract NAACCR fields from a plain text note
uv run onc-data-wrangler extract note.txt --ontology naaccr \
    --vllm-url http://localhost:8000/v1 --model my-model

# Extract from a CSV/parquet of patient notes
uv run onc-data-wrangler extract notes.csv --ontology naaccr \
    --vllm-url http://localhost:8000/v1 --model my-model \
    --text-column note_text --patient-id-column mrn

# Use multiple ontologies
uv run onc-data-wrangler extract notes.parquet \
    --ontology naaccr generic_cancer \
    --vllm-url http://localhost:8000/v1

# Output as NAACCR XML
uv run onc-data-wrangler extract notes.csv --ontology naaccr \
    --format naaccr-xml -o registry_output.xml \
    --vllm-url http://localhost:8000/v1

# Use Claude instead of a local vLLM server
uv run onc-data-wrangler extract note.txt --ontology generic_cancer \
    --provider anthropic --model claude-sonnet-4-20250514

# Specify cancer type for site-specific items
uv run onc-data-wrangler extract notes.csv --ontology naaccr \
    --cancer-type breast --vllm-url http://localhost:8000/v1
```

**Supported input formats:**
- `.txt` — single clinical note (treated as one patient)
- `.csv` / `.tsv` — tabular with columns for patient ID, text, and optionally date
- `.parquet` — same as CSV but in Parquet format

**Output formats** (`--format`):
- `json` (default) — full extraction results with metadata per patient
- `csv` — flat table with patient_id, category, field, value columns
- `naaccr-xml` — registry-submission-ready NAACCR XML (requires `--ontology naaccr`)
- `naaccr-csv` — one column per NAACCR item (requires `--ontology naaccr`)

A `_metadata.csv` file is always written alongside the main output, containing per-field confidence scores and evidence quotes.

#### 3. Start the query server

```bash
uv run onc-data-wrangler serve configs/my_project.yaml
```

#### 4. Start the chatbot (standalone)

```bash
uv run onc-data-wrangler chat configs/my_project.yaml
# Open http://localhost:8080 in your browser
```

## How Extraction Works

The extraction engine uses a **dictionary-driven, domain-group-based** approach adapted from a purpose-built NAACCR cancer registry extraction pipeline. This approach works well because it feeds the LLM structured field definitions with valid codes, tracks confidence per field, and resolves LLM output to valid codes.

### Extraction Flow

```
Input (clinical notes per patient)
    |
    v
[Chunk by tokens] (default 40K tokens, 200 overlap)
    |
    v
For each chunk (round-based, all patients in parallel):
  For each ontology:
    For each domain group (sequential, respecting dependencies):
      |
      v
    [Resolve items] -> filter already-high-confidence items
      |
      v
    [Batch by items_per_call] (default 50)
      |
      v
    For each batch:
      [Build system prompt] (domain-specific, with site context)
      [Build user prompt] (prior state + chunk text + field descriptions with valid codes)
      [Call LLM] -> {field: {value, confidence, evidence}} per item
      [Resolve codes] -> 6-tier resolution against valid code tables
      [Create ExtractionResult] -> per-field confidence, evidence, provenance
      |
      v
    [Merge into state] (higher confidence wins)
      |
      v
    After demographics: [Resolve schema] -> add site-specific staging items
  |
  v
[Save checkpoint] (JSONL per round, crash-safe resume)
    |
    v
After all chunks:
  [Validate] -> cross-field edits, code checks
  [Score confidence] -> flag for human review (CRITICAL/HIGH/MEDIUM/LOW)
  [Output] -> list[dict] for database, NAACCR XML/flat/CSV, audit trail
```

### NAACCR Domain Groups

When using the NAACCR ontology, extraction proceeds through 7 hand-curated domain groups:

| Group | Items | Depends On | Description |
|-------|-------|------------|-------------|
| Demographics & Cancer ID | 23 | — | Primary site (C##.#), histology, behavior, diagnosis date, sex, race, age |
| Staging & Prognostic Factors | 39-85 | Demographics | TNM (clinical + pathological), Summary Stage, EOD, biomarkers, SSDIs (dynamic per schema) |
| Surgical Treatment | 15 | Demographics | Surgery date, primary site procedure, LN scope, margins |
| Radiation Treatment | 35 | Demographics | Radiation date, modality, up to 3 phases (dose, fractions, technique) |
| Systemic Therapy | 16 | Demographics | Chemo, hormone, BRM/immunotherapy, other systemic, neoadjuvant status |
| Follow-up & Outcomes | 6 | — | Last contact date, vital status, cancer status |
| Narrative Summaries | 17 | — | Running-update text summaries for registry text fields |

After the demographics group extracts primary site and histology, the **SchemaRegistry** determines the cancer type and dynamically adds site-specific data items (SSDIs) to the staging group. For example:
- **Breast**: ER/PR status (summary, percent positive, Allred score), HER2 (IHC, ISH, overall), Ki-67, Oncotype Dx, multigene signatures
- **Prostate**: Gleason patterns (clinical + pathological), PSA lab value, core counts
- **Lung**: ALK, EGFR, KRAS, BRAF mutations, PD-L1, STAS, pleural invasion
- **Colorectal**: CEA, CRM, MSI, KRAS, tumor deposits

### Code Resolution

When the LLM returns a value for a field, the code resolver maps it to a valid code:

| Tier | Method | Confidence |
|------|--------|------------|
| 1 | Exact code match | 1.0 |
| 2 | Case-insensitive code match | 0.95 |
| 3 | Exact description match | 0.9 |
| 4 | Fuzzy description match (rapidfuzz, score >85) | 0.9 * score/100 |
| 5 | Numeric range check (from allowable values) | 0.85 |
| 6 | No match (pass through) | 0.0 |

Final field confidence = min(LLM confidence, resolution confidence). Fields with no code table get resolution confidence 1.0 (pass-through).

## Pipeline Stages

| Stage | Description | Input | Output |
|---|---|---|---|
| `cohort` | Define patient cohort from roster + optional diagnosis filter + optional demographics file | Source CSVs/parquets | `cohort.parquet`, `cohort_ids.json` |
| `prepare_notes` | Prepare clinical notes for extraction | Notes files | Filtered notes |
| `extract` | Extract structured data from clinical notes via domain-group LLM extraction | Notes + ontology schemas | Extraction shards (parquet) + audit trail |
| `harmonize` | Map structured data columns to ontology fields | Source files + field mappings | Harmonized parquets |
| `propose_tables` | Preview the database schema that will be created | Cohort + extractions + harmonized | Schema preview (displayed to user) |
| `database` | Build DuckDB with de-identified IDs and dates | All of the above | `.duckdb` file |
| `metadata` | Generate schema and summary docs from database | DuckDB | `schema.md`, `summary.md` |

## Available Ontologies

| ID | Dictionary Source | Description |
|---|---|---|
| `naaccr` | NAACCR v26 CSV dictionary (771 items, 4372 codes, 22 cancer schemas) | North American cancer registry with site-specific data items |
| `prissmm` | GENIE BPC Excel variable synopsis (536 fields, 332 with coded values) | PRISSMM/GENIE BPC clinical data model (NSCLC) |
| `omop` | Pre-filtered OMOP vocabulary (544K oncology concepts from SNOMED, ICD10CM, RxNorm, LOINC) | OMOP CDM v5.4 with oncology extension |
| `msk_chord` | MSK CDM codebook metadata CSV (770 fields) | MSK-CHORD cBioPortal clinical data model |
| `pan_top` | Built-in JSON definitions | Pan-TOP thoracic oncology (lung, mesothelioma, thymus) |
| `generic_cancer` | Built-in JSON definitions | Cancer-type-agnostic structured extraction |
| `matchminer_ai` | Built-in JSON definitions | MatchMiner-AI clinical trial matching concepts |
| `clinical_summary` | N/A (free-text) | Free-text clinical narrative summary |

## Field Discovery

Use the discovery agent to interactively explore your data and create field mappings:

```bash
uv run onc-data-wrangler discover /path/to/data --ontologies naaccr pan_top
```

The agent will explore your CSV/parquet files, identify relevant columns, and
suggest ontology field mappings you can add to your project config.

## Fine-Tuning Summary Models (GRPO)

You can fine-tune a language model to produce clinical summaries optimized for
downstream structured extraction. This uses Group Relative Policy Optimization
(GRPO) from the HuggingFace TRL library.

The reward loop:
1. The model generates a free-text summary from patient notes
2. A configurable reward LLM extracts structured data from the summary using
   one or more target ontologies
3. The structured extraction is compared to "silver standard" labels
   (auto-extracted from the full original notes) to compute an F1 reward
4. The model is updated via GRPO to produce summaries that better preserve
   structured information

### Setup

Install the training dependencies:

```bash
uv sync --extra training
```

### Configuration

Add a `training` section to your project YAML config:

```yaml
training:
  model: "Qwen/Qwen3.5-35B-A3B"            # Model to fine-tune
  target_ontology_ids: ["generic_cancer"]     # Structured ontologies for reward
  reward_llm:                                 # LLM for reward extraction
    provider: openai
    model: "Qwen/Qwen3.5-35B-A3B"
    base_url: "http://localhost:8000/v1"
  use_lora: true                              # Use LoRA (recommended)
  lora_rank: 16
  learning_rate: 1.0e-6
  num_epochs: 1
  batch_size: 4
  num_generations: 4                          # GRPO group size
  max_summary_tokens: 2048
  gpus: [0, 1, 2, 3]
  output_dir: "./finetuned_model"
```

### Run fine-tuning

```bash
# Basic usage
uv run onc-data-wrangler finetune configs/my_project.yaml

# With CLI overrides
uv run onc-data-wrangler finetune configs/my_project.yaml \
    --gpus 0,1,2,3 \
    --epochs 2 \
    --batch-size 8 \
    --max-patients 100
```

The command will:
1. Generate silver-standard structured extractions from full notes (if not already done)
2. Build training prompts from patient notes using the `clinical_summary` ontology
3. Run GRPO training with the configured reward function
4. Save the fine-tuned model (LoRA adapter or full weights) to the output directory

## Project Configuration

See `configs/example_project.yaml` for a complete configuration reference.

Key sections:

- `project` — Name, input paths (list of files/directories), output directory
- `cohort` — Patient roster file, optional diagnosis file, optional demographics file, ID and demographic column names, diagnosis code filters, followup date
- `extraction` — LLM backend (vLLM/Claude/Vertex), ontology IDs, notes paths, chunking parameters, `items_per_call` (default 50)
- `database` — Date de-identification (`deidentify_dates: true`), column filtering, ID prefix
- `query` — MCP server host/port, cell suppression threshold, output size limits
- `chatbot` — Web interface LLM settings, MCP connection
- `training` — GRPO fine-tuning settings: model, reward LLM, target ontologies, LoRA config, hyperparameters
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
                              +-----------------------------------+
                              |        Web UI (React SPA)         |
                              |  Setup | Pipeline | Config | ...  |
                              +-----------------+-----------------+
                                                | /api/*
                              +-----------------v-----------------+
                              |     FastAPI (unified app)         |
                              |  setup_api | pipeline_api | ...   |
                              +-----------------+-----------------+
                                                |
           +------------------------------------+-----------------------------+
           |                                    |                             |
           v                                    v                             v
   [Setup Agent]                       [Pipeline Runner]              [MCP Server]
  (ClaudeSDKClient                     (background thread             (SQL validation
   over HTTP/SSE)                       + progress tracking)           + cell suppression)
           |                                    |                             |
           v                                    |                             v
      YAML Config                               |                       [Chatbot]
           |                                    |
           v                                    v
Source Data -> [Cohort] -> [Prepare Notes] -> [Extract] -> [Harmonize]
                  |                               |                |
                  v                               v                v
            cohort.parquet              extractions/*.parquet   harmonized/
            cohort_ids.json             audit_trail.csv
                  |                     review_queue.csv
                  |               [Propose Tables]                 |
                  |                (schema preview)                 |
                  +--------------------+---------------------------+
                                       |
                                       v
                               [Database Builder]
                           (de-identify IDs + dates)
                                       |
                                       v
                                 project.duckdb
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/

# Run a single module
uv run python -m onc_data_wrangler.query.mcp_server configs/my_project.yaml
```

### Frontend development

The web UI is a React + TypeScript SPA built with Vite and Tailwind CSS, located
at `src/onc_data_wrangler/web/frontend/`.

```bash
cd src/onc_data_wrangler/web/frontend

# Install dependencies
npm install

# Start the Vite dev server (hot reload, port 5173)
npm run dev

# Build for production (outputs to dist/)
npm run build
```

During development, run both the Vite dev server and the FastAPI backend:

```bash
# Terminal 1: backend
uv run onc-data-wrangler ui

# Terminal 2: frontend (with hot reload)
cd src/onc_data_wrangler/web/frontend && npm run dev
```

The Vite dev server proxies `/api/*`, `/chat`, `/answer`, and other backend
routes to `http://localhost:8080` so both servers work together seamlessly.
Open `http://localhost:5173/ui/` for the dev frontend.

For production, run `npm run build` first, then the built files are served
directly by FastAPI at `/ui/`.

#### Backend API endpoints

| Prefix | Router | Purpose |
|--------|--------|---------|
| `/api/setup/` | `setup_api.py` | Start/message/stop setup agent sessions (SSE streaming) |
| `/api/pipeline/` | `pipeline_api.py` | Launch/status/logs for pipeline runs (background thread) |
| `/api/config/` | `config_api.py` | Load/save/validate YAML configs, list ontologies |
| `/api/data/` | `data_api.py` | List files, preview data, column stats, list outputs |

#### Key frontend libraries

| Library | Purpose |
|---------|---------|
| React 18 + TypeScript | UI framework |
| Vite | Build tool |
| Tailwind CSS | Utility-first styling |
| `@tanstack/react-query` | Server state and polling |
| `zustand` | Lightweight global state |
| `react-markdown` + `remark-gfm` | Markdown rendering |
| `@tanstack/react-table` | Data tables |
| `lucide-react` | Icons |
