# Onc-Data-Wrangler Developer Guide

## Project Structure

- `src/onc_data_wrangler/` — Main package
  - `config.py` — YAML configuration dataclasses
  - `cli.py` — CLI entry point with subcommands
  - `llm/` — LLM backend abstraction (vLLM, Claude)
  - `ontologies/` — Pluggable ontology system with builtins
    - `base.py` — OntologyBase ABC, DataItem, DataCategory; optional hooks: `get_code_resolver()`, `get_schema_resolver()`, `get_domain_groups()`
    - `protocols.py` — DictionaryItemLike, CodeResolverLike, SchemaResolverLike, DomainGroup protocols
    - `registry.py` — OntologyRegistry with @register_ontology decorator
    - `builtins/naaccr/` — NAACCR v26 ontology (dictionary-driven from CSVs)
      - `data/` — DataItems.csv, CodeList.csv, AlternateNames.csv (771 items, 4372 codes)
      - `dictionary.py` — NAACCRDictionary CSV loader
      - `code_resolver.py` — 6-tier NAACCRCodeResolver (exact → case-insensitive → description → fuzzy → numeric range → fail)
      - `schema_registry.py` — 22-schema SchemaRegistry (ICD-O-3 site+histology → site-specific data items)
    - `builtins/msk_chord/` — MSK-CHORD ontology with CDM codebook dictionary (770 fields)
    - `builtins/prissmm/` — PRISSMM/GENIE BPC ontology with Excel variable synopsis dictionary (536 fields)
    - `builtins/omop/` — OMOP CDM v5.4 with pre-filtered oncology vocabulary (544K concepts)
    - `builtins/generic_cancer/`, `builtins/pan_top/`, `builtins/matchminer_ai/`, `builtins/clinical_summary/`
  - `extraction/` — Domain-group-based LLM extraction from notes
    - `extractor.py` — Extractor (domain groups, batching, code resolution, confidence), SummaryExtractor, create_extractor factory
    - `result.py` — ExtractionResult dataclass (field_id, confidence, evidence, resolved_code), merge_results, serialization
    - `code_resolver.py` — GenericCodeResolver for non-NAACCR ontologies (same 6-tier pattern using DataItem.valid_values)
    - `schema_builder.py` — Prompt-level JSON format instructions builder (tells LLM to return `{field: {value, confidence, evidence}}`)
    - `domain_groups.py` — 7 hand-curated NAACCR domain groups + auto-generation for generic ontologies; domain-specific system prompts
    - `validator.py` — EnhancedValidator with NAACCR cross-field edits (site/sex, site/laterality, treatment dates) and code validation
    - `audit.py` — AuditTrail (per-item provenance CSV), ReviewQueue (prioritized worklist), ConfidenceScorer (CRITICAL/HIGH/MEDIUM/LOW)
    - `chunked.py` — ChunkedExtractor with round-based processing and checkpointing
  - `output/` — Output format writers
    - `naaccr_writer.py` — NAACCR XML (NaaccrData>Patient>Tumor), flat-file, CSV writer
  - `harmonization/` — Structured data column mapping
  - `cohort/` — Cohort definition from structured tables
  - `database/` — DuckDB creation and metadata generation
  - `query/` — SQL validation, privacy, MCP server
  - `web/` — Web UI and chatbot
    - `app.py` — FastAPI app factories (`create_app_from_config` for chatbot, `create_ui_app` for full UI)
    - `static/index.html` — Legacy standalone chatbot UI
    - `routers/` — API routers for the React UI
      - `config_api.py` — Config CRUD, ontology listing
      - `data_api.py` — File listing, data preview, column stats
      - `pipeline_api.py` — Pipeline launch/status/logs
      - `setup_api.py` — Setup agent HTTP/SSE bridge
    - `frontend/` — React + TypeScript SPA (Vite + Tailwind)
  - `training/` — GRPO fine-tuning for clinical summary models
    - `grpo_trainer.py` — TRL GRPOTrainer integration
    - `reward.py` — Reward function (structured extraction F1)
    - `silver_labels.py` — Silver-standard label generation
    - `dataset.py` — Training dataset builder
  - `agents/` — Claude Agent SDK orchestration
    - `setup.py` — Interactive setup agent (creates project config)
    - `pipeline.py` — Pipeline orchestration (runs stages, accepts optional `progress` callback)
    - `discovery.py` — Field discovery agent (maps columns to ontologies)
    - `progress.py` — Pipeline progress tracking (`PipelineRun`, `StageProgress`, `ProgressCallback`)
    - `prompts.py` — System prompts for agents
- `configs/` — Example YAML project configs
- `tests/` — Test suite

## Commands

```bash
uv sync                              # Install dependencies
uv run pytest tests/                 # Run tests
uv run onc-data-wrangler --help           # CLI help
uv run onc-data-wrangler setup /path/to/data  # Interactive project setup (CLI)
uv run onc-data-wrangler ui                   # Start web UI (no config — setup mode)
uv run onc-data-wrangler ui <cfg>             # Start web UI with config
uv run onc-data-wrangler pipeline <cfg>   # Run pipeline
uv run onc-data-wrangler pipeline <cfg> --stages cohort extract  # Run specific stages
uv run onc-data-wrangler serve <cfg>      # Start MCP server
uv run onc-data-wrangler chat <cfg>       # Start chatbot (legacy standalone)
uv run onc-data-wrangler discover /path/to/data  # Interactive field discovery
uv run onc-data-wrangler extract notes.csv --ontology naaccr --vllm-url http://localhost:8000/v1  # Standalone extraction (no config needed)
uv run onc-data-wrangler finetune <cfg>       # GRPO fine-tune summary model
```

## Extraction Architecture

Extraction uses a **dictionary-driven, domain-group-based** approach ported from the NAACCR cancer registry extraction pipeline.

### How It Works

1. **Dictionary loading**: Each ontology loads its field definitions from an authoritative source — CSV files for NAACCR (771 items with 4,372 valid codes), CDM codebook for MSK-CHORD (770 fields), Excel variable synopsis for PRISSMM (536 fields), pre-filtered OMOP vocabulary (544K oncology concepts), or JSON files for generic ontologies.

2. **Domain groups**: Fields are organized into sequential domain groups with data dependencies. For NAACCR, there are 7 hand-curated groups: demographics (23 items) → staging (39-85 items, dynamic per cancer schema) → surgery (15) → radiation (35) → systemic (16) → followup (6) → narratives (17). For other ontologies, groups are auto-generated from DataCategory objects.

3. **Prompt construction**: For each domain group, the SchemaBuilder generates prompt-level JSON format instructions that embed valid codes inline (e.g., `"Valid codes: 0=No, 1=Yes, 9=Unknown"`). The LLM is instructed to return `{field: {value, confidence, evidence}}` per item.

4. **Batching**: Items within a domain group are batched by `items_per_call` (default 50) to keep individual LLM calls focused.

5. **Schema resolution**: After demographics extraction, the SchemaRegistry determines the cancer type from the extracted primary site (ICD-O-3 topography) and histology (ICD-O-3 morphology), then dynamically populates the staging group with site-specific data items (SSDIs). For example, breast cancer adds ER/PR/HER2/Ki-67/Oncotype Dx fields; prostate adds Gleason/PSA fields.

6. **Code resolution**: After each LLM call, extracted values are resolved against valid code tables using a 6-tier strategy: exact code match (1.0) → case-insensitive match (0.95) → description match (0.9) → fuzzy match via rapidfuzz (0.9 * score/100) → numeric range check (0.85) → no match (0.0). Final confidence = min(LLM confidence, resolution confidence).

7. **Merging across chunks**: When a patient has multiple text chunks, results are merged using higher-confidence-wins: if a later chunk provides a higher-confidence value for the same field, it replaces the earlier one.

8. **Validation**: After extraction, the EnhancedValidator runs NAACCR cross-field edits (site/sex consistency, site/laterality for paired organs, treatment date ordering) and the ConfidenceScorer flags items for human review at four priority levels (CRITICAL <0.9 for key fields, HIGH <0.7, MEDIUM for violations, LOW <0.5).

### Key Config Options

```yaml
extraction:
  ontology_ids: [naaccr]     # Which ontologies to extract with
  cancer_type: generic        # Or specific: lung, breast, prostate, etc.
  items_per_call: 50          # Fields per LLM call (batching)
  max_output_tokens: 16384    # Max tokens per LLM response
  chunk_tokens: 40000         # Tokens per text chunk
  patient_workers: 8          # Parallel patient processing threads
```

### Output Formats

- **list[dict]** — Standard pipeline output format `[{category: {field: value}}]` compatible with DatabaseBuilder and training modules
- **ExtractionResult metadata** — Per-field confidence, evidence, and provenance embedded in the output for audit trail
- **NAACCR XML/flat-file/CSV** — Registry-submission-ready output when using the NAACCR ontology (via `NAACCRWriter`)
- **Audit trail CSV** — Per-item provenance: patient, field, value, confidence, evidence, source chunk
- **Review queue CSV** — Prioritized worklist of items needing human review

### Adding a New Ontology

1. Create `ontologies/builtins/my_ontology/ontology.py` implementing `OntologyBase`
2. Define `DataCategory` objects with `DataItem` fields in `get_base_items()`
3. Optionally: load an external data dictionary and override `get_code_resolver()` to return a resolver with valid codes
4. Optionally: override `get_domain_groups()` for hand-curated domain groupings instead of auto-generated ones
5. Register with `@register_ontology` in `__init__.py`

## Web UI

The `ui` command launches a React frontend at `http://localhost:8080/ui/` with five pages:

- **Setup** (`/ui/setup`) — Chat with the setup agent in the browser
- **Pipeline** (`/ui/pipeline`) — Launch and monitor pipeline runs with real-time progress
- **Config** (`/ui/config`) — Visual editor for project YAML
- **Data** (`/ui/data`) — Browse and preview source data files
- **Chat** (`/ui/chat`) — Chatbot for querying the database

### Frontend development

```bash
cd src/onc_data_wrangler/web/frontend
npm install                          # Install JS dependencies
npm run build                        # Production build → dist/
npm run dev                          # Vite dev server (port 5173, proxies to backend)
```

When developing the frontend, run the Vite dev server (`npm run dev`) and the backend (`uv run onc-data-wrangler ui`) simultaneously. Vite proxies `/api/*` requests to the backend.

### Architecture

- Backend: FastAPI app with API routers under `/api/setup`, `/api/pipeline`, `/api/config`, `/api/data`
- Frontend: React 18 + TypeScript + Tailwind CSS + Vite. Uses `@tanstack/react-query` for server state, `zustand` for UI state, SSE for streaming.
- The setup agent is bridged over HTTP via `SetupAgentSession` (wraps `ClaudeSDKClient` with asyncio queue for bidirectional SSE).
- Pipeline progress is tracked via `PipelineRun` (in `agents/progress.py`) with a `PipelineLogHandler` that captures log records for the log stream endpoint.

## Key Patterns

- Config is always a `ProjectConfig` loaded from YAML
- LLM calls go through `LLMClient` ABC (vLLM or Claude backends)
- Ontologies self-register via `@register_ontology` decorator
- Extraction uses the domain-group pattern described above
- Privacy enforced via SQL validation + cell suppression in MCP server
- All patient IDs are de-identified before database creation
- Dates are de-identified by conversion to intervals since birth (`*_years_since_birth` float, `*_calendar_year` integer); raw dates and birth_date are excluded from the final database

## Data Flow

1. **Setup** (CLI agent or Web UI `/ui/setup`) interacts with user to discover source files, identify columns, configure cohort, and propose a database schema
2. **Cohort stage** builds patient roster from patient file + optional diagnosis file + optional demographics file (can be a separate file from the patient roster)
3. **Extraction stage** processes clinical notes through domain groups with code resolution and confidence tracking
4. **Harmonization stage** maps structured data columns to ontology fields using field mappings
5. **Database stage** builds DuckDB with de-identified IDs and de-identified dates; birth_date is preserved in cohort.parquet for downstream date conversion but excluded from the final DB
6. **MCP server** exposes the database with SQL validation and privacy enforcement
7. **Chatbot** (legacy standalone or Web UI `/ui/chat`) provides an agentic web interface that queries via the MCP server

## ID De-identification Flow

The `CohortBuilder` de-identifies patient IDs when building the cohort and saves the original IDs to `cohort_ids.json`. The `DatabaseBuilder` reconstructs this same mapping from `cohort_ids.json` to apply consistent de-identification to extraction and harmonized data. The cohort table is already de-identified so it is not re-mapped.

## Pipeline Stages

Valid stages (for `--stages` flag): `cohort`, `prepare_notes`, `extract`, `harmonize`, `propose_tables`, `database`, `metadata`

## Available Ontologies

| ID | Version | Dictionary Source | Code Resolver | Description |
|----|---------|-------------------|---------------|-------------|
| `naaccr` | v26 | CSV (DataItems/CodeList/AlternateNames) | NAACCRCodeResolver (6-tier) | NAACCR cancer registry with 22 site-specific schemas |
| `prissmm` | v2.0 | Excel (BPC NSCLC variable synopsis) | GenericCodeResolver (332 coded fields) | PRISSMM/GENIE BPC clinical data model |
| `omop` | v5.4 | CSV (pre-filtered OMOP vocabulary) | GenericCodeResolver (544K concepts) | OMOP CDM with oncology extension |
| `msk_chord` | v2023.12 | CSV (CDM codebook metadata) | GenericCodeResolver | MSK-CHORD cBioPortal clinical data model |
| `generic_cancer` | v1.0 | JSON (built-in) | GenericCodeResolver (from valid_values) | Cancer-type-agnostic extraction |
| `pan_top` | v1.0 | JSON (built-in) | GenericCodeResolver (from valid_values) | Pan-TOP thoracic oncology |
| `matchminer_ai` | v1.0 | JSON (built-in) | GenericCodeResolver (from valid_values) | MatchMiner-AI clinical trial matching |
| `clinical_summary` | v1.0 | N/A | N/A | Free-text clinical summary (not structured) |
