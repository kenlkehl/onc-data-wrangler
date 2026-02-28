"""Pipeline orchestration agent.

Guides users through the full Talk-to-Data pipeline:
cohort definition -> extraction -> harmonization -> database -> query.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from ..config import ProjectConfig, load_config

logger = logging.getLogger(__name__)


def _save_cohort_ids(output_dir: str, ids: list):
    """Save original cohort patient IDs for downstream filtering."""
    path = Path(output_dir) / "cohort_ids.json"
    with open(path, "w") as f:
        json.dump(ids, f)
    logger.info("Saved %d cohort IDs to %s", len(ids), path)


def _load_cohort_ids(output_dir: str) -> list | None:
    """Load original cohort patient IDs, or None if not available."""
    path = Path(output_dir) / "cohort_ids.json"
    if not path.exists():
        return None
    with open(path) as f:
        ids = json.load(f)
    logger.info("Loaded %d cohort IDs from %s", len(ids), path)
    return ids


def run_pipeline(config_path: str, stages: list[str] = None, resume: bool = False):
    """Run the Talk-to-Data pipeline stages.

    Args:
        config_path: Path to the project YAML config.
        stages: List of stages to run. If None, runs all stages.
            Options: "cohort", "extract", "harmonize", "database", "metadata".
        resume: Whether to resume extraction from checkpoint.
    """
    config = load_config(config_path)
    errors = config.validate()
    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        raise ValueError("Invalid config: " + "; ".join(errors))

    all_stages = ("cohort", "prepare_notes", "extract", "harmonize", "propose_tables", "database", "metadata")
    if stages is None:
        stages = all_stages

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        if stage not in all_stages:
            logger.warning("Unknown stage: %s (skipping)", stage)
            continue

        logger.info("=== Stage: %s ===", stage)

        if stage == "cohort":
            _run_cohort(config)
        elif stage == "prepare_notes":
            _run_prepare_notes(config)
        elif stage == "extract":
            _run_extraction(config, resume=resume)
        elif stage == "harmonize":
            _run_harmonization(config)
        elif stage == "propose_tables":
            _run_propose_tables(config)
        elif stage == "database":
            _run_database(config)
        elif stage == "metadata":
            _run_metadata(config)

    logger.info("Pipeline complete.")


def _run_cohort(config: ProjectConfig):
    """Run cohort definition stage using CohortBuilder."""
    from ..cohort.builder import CohortBuilder, CohortConfig
    import pandas as pd

    output_dir = Path(config.output_dir)
    coh = config.cohort

    cohort_config = CohortConfig(
        patient_id_column=coh.patient_id_column,
        diagnosis_code_column=coh.diagnosis_code_column,
        diagnosis_code_filter=coh.diagnosis_code_filter,
        sex_column=coh.sex_column,
        race_column=coh.race_column,
        ethnicity_column=coh.ethnicity_column,
        birth_date_column=coh.birth_date_column,
        death_date_column=coh.death_date_column,
        death_indicator_column=coh.death_indicator_column,
        followup_date=coh.followup_date,
        id_prefix=config.database.record_id_prefix,
    )

    # Find patient file
    patient_path = None
    if coh.patient_file:
        patient_path = config.find_file(coh.patient_file)
        if patient_path is None:
            logger.warning("Patient file '%s' not found in input paths. Check cohort.patient_file in config.", coh.patient_file)

    if patient_path is None:
        # Auto-detect: find a file containing the patient_id_column
        for f in config.resolve_input_files():
            if f.suffix == ".parquet":
                df_peek = pd.read_parquet(f)
            else:
                df_peek = pd.read_csv(f, nrows=0)
            if coh.patient_id_column in df_peek.columns:
                patient_path = f
                logger.info("Auto-detected patient file: %s", f)
                break

    if patient_path is None:
        logger.warning("No file with column '%s' found in input paths. Set cohort.patient_file in config.", coh.patient_id_column)
        return

    # Load patient file
    if patient_path.suffix == ".parquet":
        patient_df = pd.read_parquet(patient_path)
    else:
        patient_df = pd.read_csv(patient_path, low_memory=False)
    logger.info("Loaded patient file %s: %d rows", patient_path.name, len(patient_df))

    # Load diagnosis file if specified
    diagnosis_df = None
    if coh.diagnosis_file:
        diag_path = config.find_file(coh.diagnosis_file)
        if diag_path and diag_path.exists():
            if diag_path.suffix == ".parquet":
                diagnosis_df = pd.read_parquet(diag_path)
            else:
                diagnosis_df = pd.read_csv(diag_path, low_memory=False)
            logger.info("Loaded diagnosis file %s: %d rows", diag_path.name, len(diagnosis_df))
            # Rename patient ID column to match cohort patient_id_column
            diag_pid = config.get_patient_id_column(coh.diagnosis_file)
            if diag_pid != coh.patient_id_column and diag_pid in diagnosis_df.columns:
                diagnosis_df = diagnosis_df.rename(columns={diag_pid: coh.patient_id_column})
                logger.info("Renamed patient ID column '%s' -> '%s' in %s", diag_pid, coh.patient_id_column, coh.diagnosis_file)
        else:
            logger.warning("Diagnosis file not found: %s", coh.diagnosis_file)

    # Load demographics files
    demographics_dfs = []
    demo_file_list = list(coh.demographics_files) if coh.demographics_files else []
    # Legacy single-file fallback
    if not demo_file_list and coh.demographics_file:
        demo_file_list = [coh.demographics_file]

    for demo_filename in demo_file_list:
        demo_path = config.find_file(demo_filename)
        if demo_path and demo_path.exists():
            if demo_path.suffix == ".parquet":
                demo_df = pd.read_parquet(demo_path)
            else:
                demo_df = pd.read_csv(demo_path, low_memory=False)
            logger.info("Loaded demographics file %s: %d rows", demo_path.name, len(demo_df))
            # Rename patient ID column to match cohort patient_id_column
            demo_pid = config.get_patient_id_column(demo_filename)
            if demo_pid != coh.patient_id_column and demo_pid in demo_df.columns:
                demo_df = demo_df.rename(columns={demo_pid: coh.patient_id_column})
                logger.info("Renamed patient ID column '%s' -> '%s' in %s", demo_pid, coh.patient_id_column, demo_filename)
            demographics_dfs.append(demo_df)
        else:
            logger.warning("Demographics file not found: %s", demo_filename)

    # Build cohort
    builder = CohortBuilder(cohort_config)
    cohort_df = builder.build_from_dataframes(patient_df, diagnosis_df, demographics_dfs=demographics_dfs or None)

    # Save
    output_path = output_dir / "cohort.parquet"
    cohort_df.to_parquet(output_path, index=False)
    logger.info("Saved cohort (%d patients) to %s", len(cohort_df), output_path)

    _save_cohort_ids(output_dir, builder.original_ids)


def _run_prepare_notes(config: ProjectConfig):
    """Consolidate note files into a single sorted, cohort-filtered notes.parquet."""
    import pandas as pd

    ext_config = config.extraction
    output_dir = Path(config.output_dir)

    if not ext_config.notes_paths:
        logger.info("No notes_paths configured; skipping notes preparation")
        return

    notes_files = config.resolve_notes_files()
    if not notes_files:
        logger.warning("No CSV/parquet files found in notes_paths: %s", ext_config.notes_paths)
        return

    pid_col = ext_config.patient_id_column
    text_col = ext_config.notes_text_column

    # Load cohort IDs upfront so we can filter each file early
    cohort_ids = _load_cohort_ids(output_dir)
    cohort_set = set(str(x) for x in cohort_ids) if cohort_ids is not None else None
    if cohort_set is not None:
        logger.info("Loaded %d cohort IDs for filtering", len(cohort_set))
    else:
        logger.info("No cohort_ids.json found; keeping all patients")

    dfs = []

    for f in notes_files:
        logger.info("Loading notes file: %s", f)
        if f.suffix == ".parquet":
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f, low_memory=False)

        # Check for per-file patient ID column and rename if needed
        file_pid = config.get_patient_id_column(f.name)
        if file_pid != pid_col and file_pid in df.columns and pid_col not in df.columns:
            df = df.rename(columns={file_pid: pid_col})
            logger.info("  Renamed patient ID column '%s' -> '%s' in %s", file_pid, pid_col, f.name)

        if pid_col not in df.columns:
            logger.info("  Skipping %s: missing patient ID column '%s' (has: %s)", f.name, pid_col, list(df.columns))
            continue
        if text_col not in df.columns:
            logger.info("  Skipping %s: missing text column '%s' (has: %s)", f.name, text_col, list(df.columns))
            continue

        # Filter to cohort early, before any text processing
        if cohort_set is not None:
            before = len(df)
            df = df[df[pid_col].astype(str).isin(cohort_set)]
            logger.info("  Filtered to cohort: %d -> %d notes in %s", before, len(df), f.name)
            if df.empty:
                continue

        # Look for a date column: try the configured name, then common alternatives
        date_col = ext_config.notes_date_column
        if date_col not in df.columns:
            date_alternatives = [
                "note_date", "date", "service_date", "encounter_date",
                "date_of_service", "note_datetime", "datetime",
                "report_date", "document_date",
            ]
            # Try case-insensitive matching
            col_lower_map = {c.lower(): c for c in df.columns}
            matched = False
            for alt in date_alternatives:
                if alt.lower() in col_lower_map:
                    original_name = col_lower_map[alt.lower()]
                    df = df.rename(columns={original_name: date_col})
                    logger.info("  Mapped date column '%s' -> '%s' in %s", original_name, date_col, f.name)
                    matched = True
                    break
            if not matched:
                logger.info("  No date column found in %s (looked for '%s' and common alternatives)", f.name, date_col)

        # Prepend note date to the beginning of each note's text
        if date_col in df.columns:
            mask = df[date_col].notna() & (df[date_col].astype(str).str.strip() != "")
            df.loc[mask, text_col] = (
                "Note date: " + df.loc[mask, date_col].astype(str) + "\n" + df.loc[mask, text_col].astype(str)
            )
            logger.info("  Prepended note dates to %d/%d notes in %s", mask.sum(), len(df), f.name)

        logger.info("  %d notes from %s", len(df), f.name)
        dfs.append(df)

    if not dfs:
        logger.warning("No valid notes files found")
        return

    all_notes = pd.concat(dfs, ignore_index=True)
    logger.info("Combined %d notes from %d files (%d patients)", len(all_notes), len(dfs), all_notes[pid_col].nunique())

    # Sort
    date_col = ext_config.notes_date_column
    if date_col and date_col in all_notes.columns:
        all_notes = all_notes.sort_values(by=[pid_col, date_col])
    else:
        all_notes = all_notes.sort_values(by=[pid_col])
        if date_col:
            logger.info("Date column '%s' not found; sorted by patient ID only", date_col)

    all_notes = all_notes.reset_index(drop=True)

    notes_path = output_dir / "notes.parquet"
    all_notes.to_parquet(notes_path, index=False)
    logger.info("Saved %d notes for %d patients to %s", len(all_notes), all_notes[pid_col].nunique(), notes_path)


def _run_extraction(config: ProjectConfig, resume: bool = False):
    """Run the extraction stage."""
    from ..extraction.extractor import Extractor
    from ..extraction.chunked import ChunkedExtractor, CheckpointManager, concatenate_patient_notes, chunk_text_by_tokens
    from ..llm.base import LLMClient
    import pandas as pd

    output_dir = Path(config.output_dir)
    ext_config = config.extraction
    vs_config = ext_config.vllm_servers

    # Create output directories
    extractions_dir = output_dir / "extractions"
    extractions_dir.mkdir(parents=True, exist_ok=True)

    # --- Early exit: check if all LLM work is already done before starting servers ---
    if resume:
        ckpt = CheckpointManager(extractions_dir)
        existing_rounds = ckpt.find_round_files()
        if existing_rounds:
            # Load notes to determine expected chunk counts
            output_notes = output_dir / "notes.parquet"
            if not output_notes.exists():
                output_notes = output_dir / "notes.csv"
            if not output_notes.exists():
                output_notes = config.find_file("notes.parquet") or config.find_file("notes.csv")

            if output_notes and Path(output_notes).exists():
                if str(output_notes).endswith(".parquet"):
                    notes_df_check = pd.read_parquet(output_notes)
                else:
                    notes_df_check = pd.read_csv(output_notes, low_memory=False)

                # Filter to cohort
                cohort_ids = _load_cohort_ids(output_dir)
                if cohort_ids is not None and ext_config.patient_id_column in notes_df_check.columns:
                    cohort_set = set(str(x) for x in cohort_ids)
                    notes_df_check = notes_df_check[notes_df_check[ext_config.patient_id_column].astype(str).isin(cohort_set)]

                # Compute chunk counts per patient
                tokenizer = None
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(ext_config.llm.model)
                except Exception:
                    pass

                grouped = notes_df_check.sort_values(by=[ext_config.patient_id_column]).reset_index(drop=True)
                patient_groups = dict(list(grouped.groupby(ext_config.patient_id_column)))
                patient_num_chunks = {}
                for pid, pdf in patient_groups.items():
                    pid_str = str(pid)
                    text = concatenate_patient_notes(pdf, ext_config.notes_text_column, ext_config.notes_date_column, ext_config.notes_type_column)
                    if tokenizer:
                        num = len(chunk_text_by_tokens(text, tokenizer, ext_config.chunk_tokens, ext_config.overlap_tokens))
                    else:
                        num = 1
                    patient_num_chunks[pid_str] = num

                all_ids = set(patient_num_chunks.keys())
                max_rounds = max(patient_num_chunks.values()) if patient_num_chunks else 0
                resume_round, _ = ckpt.determine_resume_state(all_ids, patient_num_chunks)

                if resume_round >= max_rounds:
                    logger.info("All extraction rounds already complete. Skipping server startup.")
                    results_df = ckpt.build_final_output()
                    logger.info("Extraction complete: %d rows", len(results_df))
                    return

    # --- Normal extraction flow ---
    # Determine if we use managed vLLM servers
    use_managed_servers = vs_config.gpus and ext_config.llm.provider == "openai"
    server_mgr = None
    llm_client = None

    try:
        if use_managed_servers:
            from ..llm.vllm_server import VLLMServerManager
            from ..llm.multi_client import MultiVLLMClient

            server_mgr = VLLMServerManager(
                model=ext_config.llm.model,
                gpus=vs_config.gpus,
                gpus_per_server=vs_config.gpus_per_server,
                base_port=vs_config.base_port,
                extra_args=vs_config.extra_args,
                log_dir=output_dir / "logs",
            )
            server_mgr.start()
            llm_client = MultiVLLMClient.from_base_urls(
                base_urls=server_mgr.base_urls,
                model=ext_config.llm.model,
                api_key=ext_config.llm.resolve_api_key(),
            )
        else:
            llm_client = _create_llm_client(ext_config.llm)

        # Create extractor
        extractor = Extractor(
            llm_client=llm_client,
            ontology_ids=ext_config.ontology_ids,
            cancer_type=ext_config.cancer_type,
        )

        # Try to load a tokenizer for chunking
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(ext_config.llm.model)
            logger.info("Loaded tokenizer for %s", ext_config.llm.model)
        except Exception:
            logger.info("No tokenizer available; using single-chunk extraction")

        chunked = ChunkedExtractor(
            extractor=extractor,
            tokenizer=tokenizer,
            chunk_size=ext_config.chunk_tokens,
            overlap=ext_config.overlap_tokens,
            max_retries=ext_config.max_retries,
            patient_workers=ext_config.patient_workers,
        )

        # Find notes file
        output_notes = output_dir / "notes.parquet"
        if not output_notes.exists():
            output_notes = output_dir / "notes.csv"
        if not output_notes.exists():
            notes_path = config.find_file("notes.parquet") or config.find_file("notes.csv")
            if notes_path is None:
                logger.warning("No notes file found. Run 'prepare_notes' stage first, or place notes.parquet/notes.csv in input paths.")
                return
            output_notes = notes_path

        if output_notes.suffix == ".parquet":
            notes_df = pd.read_parquet(output_notes)
        else:
            notes_df = pd.read_csv(output_notes, low_memory=False)
        logger.info("Loaded %d notes from %s", len(notes_df), output_notes)

        # Filter to cohort
        cohort_ids = _load_cohort_ids(output_dir)
        if cohort_ids is not None and ext_config.patient_id_column in notes_df.columns:
            cohort_set = set(str(x) for x in cohort_ids)
            before = len(notes_df)
            before_patients = notes_df[ext_config.patient_id_column].astype(str).nunique()
            notes_df = notes_df[notes_df[ext_config.patient_id_column].astype(str).isin(cohort_set)]
            after_patients = notes_df[ext_config.patient_id_column].astype(str).nunique()
            logger.info("Filtered notes to cohort: %d -> %d rows (%d -> %d patients)",
                         before, len(notes_df), before_patients, after_patients)
            if notes_df.empty:
                logger.warning("No notes remain after cohort filtering")
                return
        else:
            logger.info("No cohort_ids.json found; processing all patients")

        # Run extraction
        results_df = chunked.extract_cohort(
            notes_df=notes_df,
            output_dir=extractions_dir,
            patient_id_column=ext_config.patient_id_column,
            text_column=ext_config.notes_text_column,
            date_column=ext_config.notes_date_column,
            type_column=ext_config.notes_type_column,
            resume=resume,
        )

        logger.info("Extraction complete: %d rows", len(results_df))

    finally:
        if server_mgr is not None:
            server_mgr.shutdown()


def _run_harmonization(config: ProjectConfig):
    """Run the harmonization stage."""
    from ..harmonization.harmonizer import Harmonizer

    if not config.field_mappings:
        logger.info("No field mappings configured. Use the discovery agent to create them.")
        return

    harmonizer = Harmonizer.from_config(config.field_mappings)
    logger.info("Loaded %d field mappings", len(harmonizer.mappings))

    output_dir = Path(config.output_dir)
    (output_dir / "harmonized").mkdir(parents=True, exist_ok=True)

    # Filter to cohort
    cohort_ids = _load_cohort_ids(output_dir)
    cohort_set = set(cohort_ids) if cohort_ids else None
    if cohort_set is None:
        logger.info("No cohort_ids.json found; harmonizing all patients")

    pid_col = config.cohort.patient_id_column

    # Collect all source columns referenced by field mappings so we can
    # skip files that have no relevant columns without loading them fully.
    mapped_source_columns = {m.source_column for m in harmonizer.mappings}

    for source_file in config.resolve_input_files():
        import pandas as pd
        import pyarrow.parquet as pq

        # Read only the column names to check for relevant mappings
        if source_file.suffix == ".parquet":
            file_columns = set(pq.read_schema(source_file).names)
        else:
            file_columns = set(pd.read_csv(source_file, nrows=0).columns)

        # Account for patient ID column renaming
        file_pid = config.get_patient_id_column(source_file.name)
        if file_pid != pid_col and file_pid in file_columns and pid_col not in file_columns:
            file_columns.discard(file_pid)
            file_columns.add(pid_col)

        if not mapped_source_columns & file_columns:
            logger.info("Skipping %s (no mapped columns found)", source_file.name)
            continue

        logger.info("Harmonizing %s ...", source_file.name)

        if source_file.suffix == ".parquet":
            src_df = pd.read_parquet(source_file)
        else:
            src_df = pd.read_csv(source_file, low_memory=False)

        # Rename per-file patient ID column to the standard name
        file_pid = config.get_patient_id_column(source_file.name)
        if file_pid != pid_col and file_pid in src_df.columns and pid_col not in src_df.columns:
            src_df = src_df.rename(columns={file_pid: pid_col})
            logger.info("  Renamed patient ID column '%s' -> '%s' in %s", file_pid, pid_col, source_file.name)

        # Filter to cohort
        if cohort_set and pid_col in src_df.columns:
            before = len(src_df)
            src_df = src_df[src_df[pid_col].astype(str).isin(set(str(x) for x in cohort_set))]
            logger.info("  Filtered %s to cohort: %d -> %d rows", source_file.name, before, len(src_df))
            if src_df.empty:
                logger.info("  No rows after filtering, skipping")
                continue

        # Save to temp file for harmonize_file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        tmp_path = Path(tmp.name)
        src_df.to_parquet(tmp_path, index=False)

        results = harmonizer.harmonize_file(tmp_path, patient_id_column=pid_col)
        tmp_path.unlink(missing_ok=True)

        for category, df in results.items():
            harmonize_path = output_dir / "harmonized" / f"{source_file.stem}_{category}.parquet"
            df.to_parquet(harmonize_path, index=False)
            logger.info("  %s: %d rows -> %s", category, len(df), harmonize_path)


def _run_propose_tables(config: ProjectConfig):
    """Display proposed DuckDB table structure based on ontologies and field mappings."""
    from ..ontologies.registry import OntologyRegistry
    import pandas as pd

    output_dir = Path(config.output_dir)
    ext_config = config.extraction

    proposed = {}

    # Cohort table
    cohort_cols = ("record_id", "sex", "race", "ethnicity",
                   "birth_to_last_followup_or_death_years", "died_yes_or_no")
    cohort_path = output_dir / "cohort.parquet"
    if cohort_path.exists():
        df = pd.read_parquet(cohort_path)
        cohort_cols = list(df.columns)
    proposed["cohort"] = {"sources": ["cohort builder"], "columns": cohort_cols}

    # Extraction tables from ontologies
    for ont_id in ext_config.ontology_ids:
        try:
            ont = OntologyRegistry.get(ont_id)
        except ValueError:
            logger.warning("Unknown ontology '%s'; skipping", ont_id)
            continue

        cancer_type = ext_config.cancer_type if ext_config.cancer_type != "generic" else None
        categories = ont.get_site_specific_items(cancer_type or "generic")
        for cat in categories:
            table_name = _table_name_from_category(cat.id)
            if table_name not in proposed:
                proposed[table_name] = {"sources": [], "columns": []}
            proposed[table_name]["sources"].append("AI extraction")
            for item in cat.items:
                col_name = item.json_field or item.id
                if col_name not in proposed[table_name]["columns"] and col_name not in ("record_id", "category", "source"):
                    proposed[table_name]["columns"].append(col_name)
            proposed[table_name]["columns"].insert(0, "record_id")

    # Harmonized tables from field mappings
    if config.field_mappings:
        for category_key, mapping in config.field_mappings.items():
            if isinstance(mapping, list):
                table_name = _table_name_from_category(category_key)
                if table_name not in proposed:
                    proposed[table_name] = {"sources": [], "columns": ["record_id"]}
                proposed[table_name]["sources"].append("harmonized structured data")
                for item in mapping:
                    target = item.get("target", "")
                    if target and target not in proposed[table_name]["columns"]:
                        proposed[table_name]["columns"].append(target)

    # Check for existing harmonized files
    harmonized_dir = output_dir / "harmonized"
    if harmonized_dir.exists():
        for pf in sorted(harmonized_dir.glob("*.parquet")):
            parts = pf.stem.rsplit("_", 1)
            if len(parts) > 1:
                table_name = _table_name_from_category(parts[-1])
            else:
                table_name = _table_name_from_category(parts[0])
            if table_name not in proposed:
                proposed[table_name] = {"sources": ["harmonized structured data"], "columns": ["record_id"]}

    # Print summary
    print("=== Proposed Database Tables ===")
    for table_name in sorted(proposed):
        info = proposed[table_name]
        sources_str = ", ".join(info["sources"])
        cols = info["columns"]
        if len(cols) > 15:
            cols_str = ", ".join(cols[:15]) + f", ... (+{len(cols) - 15} more)"
        else:
            cols_str = ", ".join(cols)
        print(f"  {table_name:<25s} [{sources_str}]  columns: {cols_str}")

    print(f"Total: {len(proposed)} proposed tables")

    # Save proposal
    proposal_path = output_dir / "proposed_tables.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(proposal_path, "w") as f:
        json.dump(proposed, f, indent=2)
    logger.info("Saved table proposal to %s", proposal_path)


def _table_name_from_category(category: str) -> str:
    """Convert a category name to a database table name.

    Delegates to the canonical implementation in database.builder.
    """
    from ..database.builder import _table_name_from_category as _impl
    return _impl(category)


def _run_database(config: ProjectConfig):
    """Run the database creation stage."""
    from ..database.builder import DatabaseBuilder

    builder = DatabaseBuilder(config)
    builder.build()
    logger.info("Database created at %s", config.db_path)


def _run_metadata(config: ProjectConfig):
    """Run the metadata generation stage."""
    from ..database.metadata import generate_schema, generate_summary, generate_summary_stats
    import duckdb

    db_path = config.db_path
    if not db_path.exists():
        logger.warning("Database not found at %s. Run 'database' stage first.", db_path)
        return

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        forbidden = set(config.database.forbidden_output_columns)
        min_cell = config.query.min_cell_size

        schema = generate_schema(con, config.name, forbidden_columns=forbidden)
        config.schema_path.parent.mkdir(parents=True, exist_ok=True)
        config.schema_path.write_text(schema)
        logger.info("Schema written to %s", config.schema_path)

        summary = generate_summary(con, config.name, forbidden_columns=forbidden, min_cell_size=min_cell)
        config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        config.summary_path.write_text(summary)
        logger.info("Summary written to %s", config.summary_path)

        import json
        stats = generate_summary_stats(con, config.name, forbidden_columns=forbidden, min_cell_size=min_cell)
        config.summary_stats_path.parent.mkdir(parents=True, exist_ok=True)
        config.summary_stats_path.write_text(json.dumps(stats, indent=2, default=str))
        logger.info("Summary stats JSON written to %s", config.summary_stats_path)
    finally:
        con.close()


def _create_llm_client(llm_config):
    """Create an LLM client from config."""
    if llm_config.provider in ("anthropic", "vertex"):
        from ..llm.claude_client import create_claude_client_from_config
        return create_claude_client_from_config(llm_config)
    else:
        from ..llm.vllm_client import VLLMClient
        return VLLMClient(
            base_url=llm_config.base_url or "http://localhost:8000/v1",
            model=llm_config.model,
            api_key=llm_config.resolve_api_key(),
        )
