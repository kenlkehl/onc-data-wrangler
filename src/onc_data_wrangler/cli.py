"""Command-line interface for Onc-Data-Wrangler."""

import argparse
import asyncio
import logging
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="onc-data-wrangler",
        description="Onc-Data-Wrangler: Build agentic clinical dataset query systems.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # pipeline
    p_pipeline = subparsers.add_parser("pipeline", help="Run the data processing pipeline")
    p_pipeline.add_argument("config", help="Path to project YAML config")
    p_pipeline.add_argument("--stages", nargs="+", choices=("cohort", "prepare_notes", "extract", "harmonize", "propose_tables", "database", "metadata"), help="Stages to run (default: all)")
    p_pipeline.add_argument("--resume", action="store_true", help="Resume extraction from checkpoint")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the MCP query server")
    p_serve.add_argument("config", nargs="?", help="Path to project YAML config")
    p_serve.add_argument("--host", default=None, help="Server host")
    p_serve.add_argument("--port", type=int, default=None, help="Server port")

    # chat
    p_chat = subparsers.add_parser("chat", help="Start the web chatbot")
    p_chat.add_argument("config", nargs="?", help="Path to project YAML config")
    p_chat.add_argument("--host", default=None, help="Server host")
    p_chat.add_argument("--port", type=int, default=None, help="Server port")

    # setup
    p_setup = subparsers.add_parser("setup", help="Interactive agentic walkthrough to configure a new project")
    p_setup.add_argument("data_paths", nargs="*", default=[], help="Files and/or directories with source data (asked interactively if omitted)")
    p_setup.add_argument("--output-dir", default=None, help="Directory for pipeline outputs (asked interactively if omitted)")
    p_setup.add_argument("--config", default=None, help="Path for the generated config YAML (asked interactively if omitted)")
    p_setup.add_argument("--max-budget", type=float, default=10.0, help="Maximum agent budget in USD (default: 10.0)")
    p_setup.add_argument("--provider", choices=["claude", "ollama"], default="claude", help="LLM provider for the setup agent (default: claude)")
    p_setup.add_argument("--model", default=None, help="Model name (required for ollama, e.g. llama3.1:70b)")
    p_setup.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL (default: http://localhost:11434)")

    # discover
    p_discover = subparsers.add_parser("discover", help="Run the field discovery agent")
    p_discover.add_argument("data_paths", nargs="+", help="Files and/or directories with source data")
    p_discover.add_argument("--ontologies", nargs="+", default=["naaccr"], help="Ontology IDs to match against")
    p_discover.add_argument("--output", default=None, help="Path to save discovered field mappings")
    p_discover.add_argument("--max-budget", type=float, default=10.0, help="Maximum agent budget in USD (default: 10.0)")

    # metadata
    p_meta = subparsers.add_parser("metadata", help="Generate schema and summary metadata from database")
    p_meta.add_argument("config", help="Path to project YAML config")

    # ui
    p_ui = subparsers.add_parser("ui", help="Start the web UI (setup wizard + pipeline dashboard + chatbot)")
    p_ui.add_argument("config", nargs="?", default=None, help="Path to project YAML config (optional for setup mode)")
    p_ui.add_argument("--host", default="0.0.0.0", help="Server host")
    p_ui.add_argument("--port", type=int, default=8080, help="Server port")

    # extract (standalone)
    p_extract = subparsers.add_parser("extract", help="Extract structured data from a notes file (no config needed)")
    p_extract.add_argument("input", help="Input file: .txt (single note), .csv, or .parquet (tabular with text column)")
    p_extract.add_argument("-o", "--output", default=None, help="Output file path (default: <input>_extractions.json)")
    p_extract.add_argument("--ontology", nargs="+", default=["naaccr"], help="Ontology IDs to extract (default: naaccr)")
    p_extract.add_argument("--cancer-type", default="generic", help="Cancer type for site-specific items (default: generic)")
    p_extract.add_argument("--vllm-url", default=None, help="vLLM server URL (e.g. http://localhost:8000/v1)")
    p_extract.add_argument("--model", default=None, help="Model name on the vLLM server")
    p_extract.add_argument("--provider", choices=["openai", "anthropic", "vertex", "azure"], default="openai", help="LLM provider (default: openai)")
    p_extract.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)")
    p_extract.add_argument("--text-column", default="text", help="Column containing note text (for CSV/parquet, default: text)")
    p_extract.add_argument("--patient-id-column", default="patient_id", help="Column containing patient IDs (for CSV/parquet, default: patient_id)")
    p_extract.add_argument("--date-column", default="date", help="Column containing note dates (for CSV/parquet, default: date)")
    p_extract.add_argument("--note-type-column", default="note_type", help="Column containing note type labels (for CSV/parquet, default: note_type)")
    p_extract.add_argument("--items-per-call", type=int, default=50, help="Fields per LLM call (default: 50)")
    p_extract.add_argument("--format", choices=["json", "csv", "naaccr-xml", "naaccr-csv"], default="json", help="Output format (default: json)")
    p_extract.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens per LLM call (default: 16384)")
    p_extract.add_argument("--chunk-tokens", type=int, default=40000, help="Tokens per text chunk (default: 40000)")
    p_extract.add_argument("--overlap-tokens", type=int, default=200, help="Overlap between text chunks (default: 200)")
    p_extract.add_argument("--patient-workers", type=int, default=8, help="Patients to process concurrently in standalone cohort mode (default: 8)")
    p_extract.add_argument("--resume", action="store_true", help="Resume extraction from checkpoint in work directory")
    p_extract.add_argument("--work-dir", default=None, help="Directory for intermediate round files (default: <output_stem>_work/)")

    # qa (clinical question answering)
    p_qa = subparsers.add_parser("qa", help="Answer clinical questions from patient notes")
    p_qa.add_argument("--input", required=True, help="Notes file (CSV/parquet)")
    p_qa.add_argument("--questions", required=True, help="Questions file (text, one per line)")
    p_qa.add_argument("-o", "--output", default=None, help="Output JSONL path")
    p_qa.add_argument("--patient-id-column", default="patient_id", help="Column containing patient IDs (default: patient_id)")
    p_qa.add_argument("--text-column", default="text", help="Column containing note text (default: text)")
    p_qa.add_argument("--date-column", default="date", help="Column containing note dates (default: date)")
    p_qa.add_argument("--note-type-column", default="note_type", help="Column containing note type labels (default: note_type)")
    p_qa.add_argument("--provider", choices=["openai", "anthropic", "vertex", "azure"], default="openai", help="LLM provider (default: openai)")
    p_qa.add_argument("--vllm-url", default=None, help="vLLM server URL (e.g. http://localhost:8000/v1)")
    p_qa.add_argument("--model", default=None, help="Model name on the LLM server")
    p_qa.add_argument("--api-key", default=None, help="API key (or set env var)")
    p_qa.add_argument("--chunk-tokens", type=int, default=50000, help="Tokens per text chunk (default: 50000)")
    p_qa.add_argument("--overlap-tokens", type=int, default=500, help="Overlap between text chunks (default: 500)")
    p_qa.add_argument("--patient-workers", type=int, default=8, help="Patients to process concurrently (default: 8)")
    p_qa.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens per LLM call (default: 16384)")
    p_qa.add_argument("--resume", action="store_true", help="Resume from checkpoint in work directory")
    p_qa.add_argument("--work-dir", default=None, help="Directory for intermediate round files (default: <output_stem>_work/)")

    # finetune
    p_finetune = subparsers.add_parser("finetune", help="Fine-tune a summary model using GRPO")
    p_finetune.add_argument("config", help="Path to project YAML config")
    p_finetune.add_argument("--gpus", default=None, help="Comma-separated GPU IDs (overrides config)")
    p_finetune.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    p_finetune.add_argument("--batch-size", type=int, default=None, help="Training batch size (overrides config)")
    p_finetune.add_argument("--max-patients", type=int, default=None, help="Limit number of training patients")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "pipeline":
        from .agents.pipeline import run_pipeline
        run_pipeline(config_path=args.config, stages=args.stages, resume=args.resume)

    elif args.command == "serve":
        from .query.mcp_server import create_server_from_config
        from .config import load_config
        config = load_config(args.config)
        if args.host:
            config.query.mcp_host = args.host
        if args.port:
            config.query.mcp_port = args.port
        mcp = create_server_from_config(config)
        mcp.run(transport="streamable-http")

    elif args.command == "chat":
        from .web.app import create_app_from_config
        from .config import load_config
        config = load_config(args.config)
        app = create_app_from_config(config)
        HOST = args.host or config.chatbot.host
        PORT = args.port or config.chatbot.port
        import uvicorn
        uvicorn.run(app, host=HOST, port=PORT)

    elif args.command == "setup":
        from .agents.setup import run_setup_agent

        provider = args.provider
        model = args.model
        ollama_url = args.ollama_url

        # Interactive model selection for Ollama when --model not specified
        if provider == "ollama" and not model:
            model = _select_ollama_model(ollama_url)
            if model is None:
                sys.exit(1)

        run_setup_agent(
            data_paths=args.data_paths or None,
            output_dir=args.output_dir,
            config_path=args.config,
            max_budget_usd=args.max_budget,
            provider=provider,
            model=model,
            ollama_base_url=ollama_url,
        )

    elif args.command == "discover":
        from .agents.discovery import run_discovery_agent
        asyncio.run(run_discovery_agent(
            data_paths=args.data_paths,
            ontology_ids=args.ontologies,
            output_config_path=args.output,
            max_budget_usd=args.max_budget,
        ))

    elif args.command == "ui":
        from .web.app import create_ui_app
        from .config import load_config as _load_config
        config = _load_config(args.config) if args.config else None
        app = create_ui_app(config)
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "extract":
        _run_standalone_extract(args)

    elif args.command == "qa":
        _run_qa(args)

    elif args.command == "metadata":
        from .agents.pipeline import _run_metadata
        from .config import load_config
        config = load_config(args.config)
        _run_metadata(config)

    elif args.command == "finetune":
        from .config import load_config
        config = load_config(args.config)

        # Apply CLI overrides
        if args.gpus:
            config.training.gpus = [int(g) for g in args.gpus.split(",")]
        if args.epochs is not None:
            config.training.num_epochs = args.epochs
        if args.batch_size is not None:
            config.training.batch_size = args.batch_size
        if args.max_patients is not None:
            config.training.max_patients = args.max_patients

        _run_finetune(config)


def _run_standalone_extract(args):
    """Run extraction directly from a notes file without a project config."""
    import json as _json
    from pathlib import Path

    import pandas as pd

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # --- Determine input type and normalize into a notes DataFrame ---
    suffix = input_path.suffix.lower()

    if suffix == ".txt":
        text = input_path.read_text(encoding="utf-8")
        notes_df = pd.DataFrame([{
            args.patient_id_column: input_path.stem,
            args.text_column: text,
            args.date_column: "",
            args.note_type_column: "text",
        }])
        patient_order = [input_path.stem]
        print(f"Loaded single text file: {len(text):,} characters")

    elif suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        raw_df = pd.read_csv(input_path, sep=sep, low_memory=False)
        notes_df, patient_order = _prepare_standalone_notes_df(
            raw_df,
            patient_id_col=args.patient_id_column,
            text_col=args.text_column,
            date_col=args.date_column,
        )
        print(f"Loaded {len(raw_df)} rows, {len(patient_order)} patients from {input_path.name}")

    elif suffix == ".parquet":
        raw_df = pd.read_parquet(input_path)
        notes_df, patient_order = _prepare_standalone_notes_df(
            raw_df,
            patient_id_col=args.patient_id_column,
            text_col=args.text_column,
            date_col=args.date_column,
        )
        print(f"Loaded {len(raw_df)} rows, {len(patient_order)} patients from {input_path.name}")

    else:
        text = input_path.read_text(encoding="utf-8")
        notes_df = pd.DataFrame([{
            args.patient_id_column: input_path.stem,
            args.text_column: text,
            args.date_column: "",
            args.note_type_column: "text",
        }])
        patient_order = [input_path.stem]
        print(f"Loaded as text: {len(text):,} characters")

    if args.text_column not in notes_df.columns:
        print(f"ERROR: Text column not found: {args.text_column}")
        sys.exit(1)

    # --- Create LLM client ---
    llm_client = _create_standalone_llm(args)

    # --- Create extractor ---
    from .extraction.extractor import create_extractor

    extractor = create_extractor(
        llm_client=llm_client,
        ontology_ids=args.ontology,
        cancer_type=args.cancer_type,
        items_per_call=args.items_per_call,
    )

    print(f"Extracting with ontologies: {', '.join(args.ontology)}")
    print(f"Cancer type: {args.cancer_type}")

    # --- Optionally load a tokenizer for token-based chunking ---
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        model_name = args.model or "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        pass

    # --- Write output ---
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_extractions.json")
    fmt = args.format

    # --- Run extraction using the cohort chunking path ---
    from .extraction.chunked import ChunkedExtractor, CheckpointManager

    # Use a persistent work directory so per-patient round files survive crashes
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = output_path.parent / f"{output_path.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running cohort-style extraction: {len(notes_df):,} notes, "
        f"{len(patient_order):,} patients, {args.patient_workers} workers"
    )
    print(f"Work directory: {work_dir}")

    chunked = ChunkedExtractor(
        extractor=extractor,
        tokenizer=tokenizer,
        chunk_size=args.chunk_tokens,
        overlap=args.overlap_tokens,
        max_retries=3,
        patient_workers=args.patient_workers,
        max_tokens=args.max_tokens,
    )
    chunked.extract_cohort(
        notes_df=notes_df,
        output_dir=work_dir,
        patient_id_column=args.patient_id_column,
        text_column=args.text_column,
        date_column=args.date_column,
        type_column=args.note_type_column,
        resume=args.resume,
    )
    final_extractions = CheckpointManager(work_dir).load_final_extractions()

    all_results = {pid: final_extractions.get(pid, []) for pid in patient_order}

    if fmt == "json":
        with open(output_path, "w") as f:
            _json.dump(all_results, f, indent=2, default=str)
        print(f"Written: {output_path}")

    elif fmt == "csv":
        rows = []
        for pid, result_list in all_results.items():
            for entry in result_list:
                if not isinstance(entry, dict):
                    continue
                # Multi-diagnosis per-diagnosis fields
                if "_diagnoses" in entry:
                    for diag_entry in entry["_diagnoses"]:
                        tumor_idx = diag_entry.get("tumor_index", 0)
                        for category, fields in diag_entry.items():
                            if category == "tumor_index" or not isinstance(fields, dict):
                                continue
                            for field_name, value in fields.items():
                                rows.append({
                                    "patient_id": pid,
                                    "tumor_index": tumor_idx,
                                    "category": category,
                                    "field": field_name,
                                    "value": value,
                                })
                    continue
                # Patient-level or legacy single-diagnosis fields
                for category, fields in entry.items():
                    if category.startswith("_") or not isinstance(fields, dict):
                        continue
                    for field_name, value in fields.items():
                        if field_name.startswith("_"):
                            continue
                        rows.append({
                            "patient_id": pid,
                            "tumor_index": -1,
                            "category": category,
                            "field": field_name,
                            "value": value,
                        })
        out_df = pd.DataFrame(rows)
        csv_path = output_path.with_suffix(".csv") if output_path.suffix != ".csv" else output_path
        out_df.to_csv(csv_path, index=False)
        print(f"Written: {csv_path} ({len(rows)} rows)")

    elif fmt == "naaccr-xml":
        if "naaccr" not in args.ontology:
            print("ERROR: --format naaccr-xml requires --ontology naaccr")
            sys.exit(1)
        from .output.naaccr_writer import NAACCRWriter
        from .ontologies.builtins.naaccr.dictionary import NAACCRDictionary
        dictionary = NAACCRDictionary()
        dictionary.load()
        writer = NAACCRWriter(dictionary)
        naaccr_results = _flatten_for_naaccr_multi(all_results)
        xml_path = output_path.with_suffix(".xml") if output_path.suffix != ".xml" else output_path
        writer.write_xml(naaccr_results, xml_path)
        print(f"Written: {xml_path}")

    elif fmt == "naaccr-csv":
        if "naaccr" not in args.ontology:
            print("ERROR: --format naaccr-csv requires --ontology naaccr")
            sys.exit(1)
        from .output.naaccr_writer import NAACCRWriter
        from .ontologies.builtins.naaccr.dictionary import NAACCRDictionary
        dictionary = NAACCRDictionary()
        dictionary.load()
        writer = NAACCRWriter(dictionary)
        naaccr_results = _flatten_for_naaccr_multi(all_results)
        csv_path = output_path.with_suffix(".csv") if output_path.suffix != ".csv" else output_path
        writer.write_csv(naaccr_results, csv_path)
        print(f"Written: {csv_path}")

    # --- Also write metadata if available ---
    meta_rows = []
    for pid, result_list in all_results.items():
        for entry in result_list:
            if not isinstance(entry, dict) or "_extraction_results" not in entry:
                continue
            er = entry["_extraction_results"]
            if "patient" in er:
                # Multi-diagnosis metadata format
                for fid, r in er.get("patient", {}).items():
                    meta_rows.append({
                        "patient_id": pid,
                        "tumor_index": -1,
                        "field_id": fid,
                        "field_name": r.get("field_name", ""),
                        "extracted_value": r.get("extracted_value", ""),
                        "resolved_code": r.get("resolved_code", ""),
                        "confidence": r.get("confidence", 0),
                        "evidence": r.get("evidence_text", ""),
                        "ontology_id": r.get("ontology_id", ""),
                    })
                for key, results_dict in er.items():
                    if not key.startswith("diagnosis_"):
                        continue
                    tidx = int(key.split("_", 1)[1])
                    for fid, r in results_dict.items():
                        meta_rows.append({
                            "patient_id": pid,
                            "tumor_index": tidx,
                            "field_id": fid,
                            "field_name": r.get("field_name", ""),
                            "extracted_value": r.get("extracted_value", ""),
                            "resolved_code": r.get("resolved_code", ""),
                            "confidence": r.get("confidence", 0),
                            "evidence": r.get("evidence_text", ""),
                            "ontology_id": r.get("ontology_id", ""),
                        })
            else:
                # Legacy single-diagnosis metadata
                for fid, r in er.items():
                    meta_rows.append({
                        "patient_id": pid,
                        "tumor_index": 0,
                        "field_id": fid,
                        "field_name": r.get("field_name", ""),
                        "extracted_value": r.get("extracted_value", ""),
                        "resolved_code": r.get("resolved_code", ""),
                        "confidence": r.get("confidence", 0),
                        "evidence": r.get("evidence_text", ""),
                        "ontology_id": r.get("ontology_id", ""),
                    })
    if meta_rows:
        meta_path = output_path.with_name(f"{output_path.stem}_metadata.csv")
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
        print(f"Metadata: {meta_path} ({len(meta_rows)} field extractions)")

    print("Done.")


def _run_qa(args):
    """Run clinical question-answering from a notes file."""
    from pathlib import Path

    import pandas as pd

    from .extraction.qa_extractor import parse_questions, build_qa_output

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"ERROR: Questions file not found: {questions_path}")
        sys.exit(1)

    # --- Load questions ---
    questions = parse_questions(str(questions_path))
    print(f"Loaded {len(questions)} questions from {questions_path.name}")

    # --- Load notes ---
    suffix = input_path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        raw_df = pd.read_csv(input_path, sep=sep, low_memory=False)
    elif suffix == ".parquet":
        raw_df = pd.read_parquet(input_path)
    else:
        print(f"ERROR: Unsupported file type: {suffix} (use CSV or parquet)")
        sys.exit(1)

    notes_df, patient_order = _prepare_standalone_notes_df(
        raw_df,
        patient_id_col=args.patient_id_column,
        text_col=args.text_column,
        date_col=args.date_column,
    )
    print(f"Loaded {len(raw_df)} rows, {len(patient_order)} patients from {input_path.name}")

    if args.text_column not in notes_df.columns:
        print(f"ERROR: Text column not found: {args.text_column}")
        sys.exit(1)

    # --- Create LLM client ---
    llm_client = _create_standalone_llm(args)

    # --- Create QA extractor ---
    from .extraction.extractor import create_extractor

    extractor = create_extractor(
        llm_client=llm_client,
        ontology_ids=[],
        questions=questions,
    )

    # --- Optionally load a tokenizer ---
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        model_name = args.model or "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        pass

    # --- Output paths ---
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_qa.jsonl")

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = output_path.parent / f"{output_path.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running QA extraction: {len(patient_order)} patients, "
        f"{len(questions)} questions, {args.patient_workers} workers"
    )
    print(f"Work directory: {work_dir}")

    # --- Run extraction ---
    from .extraction.chunked import ChunkedExtractor, CheckpointManager

    chunked = ChunkedExtractor(
        extractor=extractor,
        tokenizer=tokenizer,
        chunk_size=args.chunk_tokens,
        overlap=args.overlap_tokens,
        max_retries=3,
        patient_workers=args.patient_workers,
        max_tokens=args.max_tokens,
    )
    chunked.extract_cohort(
        notes_df=notes_df,
        output_dir=work_dir,
        patient_id_column=args.patient_id_column,
        text_column=args.text_column,
        date_column=args.date_column,
        type_column=args.note_type_column,
        resume=args.resume,
    )

    # --- Write JSONL + CSV output ---
    final_extractions = CheckpointManager(work_dir).load_final_extractions()
    build_qa_output(final_extractions, output_path)

    print(f"JSONL: {output_path}")
    print(f"CSV:   {output_path.with_suffix('.csv')}")
    print("Done.")


def _create_standalone_llm(args):
    """Create an LLM client from CLI arguments."""
    import os

    provider = args.provider

    if provider == "openai":
        from .llm.vllm_client import VLLMClient
        base_url = args.vllm_url or "http://localhost:8000/v1"
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "none")
        model = args.model or "default"
        return VLLMClient(base_url=base_url, api_key=api_key, model=model)

    elif provider in ("anthropic", "vertex"):
        from .llm.claude_client import ClaudeClient
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        model = args.model or "claude-sonnet-4-20250514"
        return ClaudeClient(provider=provider, model=model, api_key=api_key)

    elif provider == "azure":
        from .llm.azure_client import AzureClient
        api_key = args.api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        model = args.model or "gpt-4o"
        azure_endpoint = args.vllm_url or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        return AzureClient(azure_endpoint=azure_endpoint, api_key=api_key, model=model)

    else:
        print(f"ERROR: Unknown provider: {provider}")
        sys.exit(1)


def _group_notes(df, patient_id_col: str, text_col: str, date_col: str) -> dict[str, list[str]]:
    """Group a DataFrame of notes by patient ID, concatenating chronologically."""
    patients: dict[str, list[str]] = {}

    if patient_id_col not in df.columns:
        # No patient ID column -- treat entire file as one patient
        texts = df[text_col].dropna().astype(str).tolist() if text_col in df.columns else []
        patients["patient_0"] = texts
        return patients

    if date_col in df.columns:
        df = df.sort_values(by=[patient_id_col, date_col])

    for pid, group in df.groupby(patient_id_col):
        if text_col not in group.columns:
            continue
        texts = group[text_col].dropna().astype(str).tolist()
        texts = [t for t in texts if len(t.strip()) > 10]
        if texts:
            patients[str(pid)] = texts

    return patients


def _prepare_standalone_notes_df(df, patient_id_col: str, text_col: str, date_col: str):
    """Normalize standalone tabular input into cohort-style notes rows."""
    import pandas as pd

    if text_col not in df.columns:
        return df.copy(), []

    notes_df = df.copy()
    notes_df = notes_df[notes_df[text_col].notna()].copy()
    notes_df[text_col] = notes_df[text_col].astype(str)
    notes_df = notes_df[notes_df[text_col].str.strip().str.len() > 10].copy()

    if patient_id_col not in notes_df.columns:
        notes_df[patient_id_col] = "patient_0"

    notes_df[patient_id_col] = notes_df[patient_id_col].astype(str)

    sort_cols = [patient_id_col]
    if date_col in notes_df.columns:
        sort_cols.append(date_col)
    notes_df = notes_df.sort_values(by=sort_cols).reset_index(drop=True)

    patient_order = notes_df[patient_id_col].dropna().astype(str).drop_duplicates().tolist()
    if not patient_order and not notes_df.empty:
        patient_order = ["patient_0"]

    return notes_df, patient_order


def _flatten_for_naaccr(all_results: dict) -> dict[str, dict[str, str]]:
    """Convert extraction results to {patient_id: {item_number_str: value}} for NAACCRWriter."""
    naaccr_data: dict[str, dict[str, str]] = {}
    for pid, result_list in all_results.items():
        items: dict[str, str] = {}
        for entry in result_list:
            if not isinstance(entry, dict):
                continue
            if "_extraction_results" in entry:
                for fid, r in entry["_extraction_results"].items():
                    if r.get("ontology_id") == "naaccr":
                        items[fid] = r.get("resolved_code", "") or r.get("extracted_value", "")
            else:
                for category, fields in entry.items():
                    if category.startswith("_") or not isinstance(fields, dict):
                        continue
                    for field_name, value in fields.items():
                        items[field_name] = str(value)
        if items:
            naaccr_data[pid] = items
    return naaccr_data


def _flatten_for_naaccr_multi(all_results: dict) -> dict[str, list[dict[str, str]]]:
    """Convert multi-diagnosis results to {patient_id: [tumor_items_dict, ...]} for NAACCRWriter.

    Each patient maps to a list of dicts (one per tumor).  Patient-level
    items are merged into every tumor dict so the writer can split them
    into the Patient vs Tumor XML elements.
    """
    naaccr_data: dict[str, list[dict[str, str]]] = {}

    for pid, result_list in all_results.items():
        patient_items: dict[str, str] = {}
        tumor_items: dict[int, dict[str, str]] = {}

        for entry in result_list:
            if not isinstance(entry, dict):
                continue

            if "_extraction_results" in entry:
                er = entry["_extraction_results"]
                if "patient" in er:
                    # Multi-diagnosis metadata
                    for fid, r in er.get("patient", {}).items():
                        if r.get("ontology_id") == "naaccr":
                            patient_items[fid] = r.get("resolved_code", "") or r.get("extracted_value", "")
                    for key, results_dict in er.items():
                        if not key.startswith("diagnosis_"):
                            continue
                        tidx = int(key.split("_", 1)[1])
                        if tidx not in tumor_items:
                            tumor_items[tidx] = {}
                        for fid, r in results_dict.items():
                            if r.get("ontology_id") == "naaccr":
                                tumor_items[tidx][fid] = r.get("resolved_code", "") or r.get("extracted_value", "")
                else:
                    # Legacy single-diagnosis
                    for fid, r in er.items():
                        if r.get("ontology_id") == "naaccr":
                            patient_items[fid] = r.get("resolved_code", "") or r.get("extracted_value", "")

        if not tumor_items and patient_items:
            # Legacy single-diagnosis: everything as one tumor
            tumor_items[0] = patient_items
            naaccr_data[pid] = [tumor_items[0]]
        elif tumor_items:
            # Merge patient-level items into each tumor dict
            tumors = []
            for tidx in sorted(tumor_items.keys()):
                merged = dict(patient_items)
                merged.update(tumor_items[tidx])
                tumors.append(merged)
            naaccr_data[pid] = tumors

    return naaccr_data


def _select_ollama_model(ollama_url: str) -> str | None:
    """Interactively select an Ollama model. Returns model name or None on failure."""
    import urllib.request
    import json as _json

    tags_url = f"{ollama_url.rstrip('/')}/api/tags"
    print(f"\nChecking Ollama server at {ollama_url}...")
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            data = _json.loads(resp.read())
    except Exception as exc:
        print(
            f"\nError: Cannot reach Ollama at {ollama_url}.\n"
            f"Ensure Ollama is installed and running (`ollama serve`).\n"
            f"Details: {exc}"
        )
        return None

    models = data.get("models", [])
    if not models:
        print(
            "\nNo models found on the Ollama server.\n"
            "Pull a model first: ollama pull llama3.1:70b"
        )
        return None

    print(f"\nAvailable Ollama models:")
    for i, m in enumerate(models, 1):
        name = m.get("name", "unknown")
        size_bytes = m.get("size", 0)
        size_gb = size_bytes / (1024 ** 3)
        print(f"  [{i}] {name}  ({size_gb:.1f} GB)")

    while True:
        try:
            choice = input("\nSelect a model number, or type a model name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not choice:
            continue
        # Try as a number
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                selected = models[idx - 1]["name"]
                print(f"Using model: {selected}")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(models)}.")
                continue
        except ValueError:
            pass
        # Use as a model name directly
        print(f"Using model: {choice}")
        return choice


def _run_finetune(config):
    """Run the GRPO fine-tuning workflow."""
    import pandas as pd
    from pathlib import Path

    output_dir = Path(config.output_dir)
    ext_config = config.extraction

    if not config.training.model:
        print("ERROR: training.model must be set in config YAML")
        sys.exit(1)

    # Load notes
    notes_path = output_dir / "notes.parquet"
    if not notes_path.exists():
        notes_path = output_dir / "notes.csv"
    if not notes_path.exists():
        notes_path = config.find_file("notes.parquet") or config.find_file("notes.csv")

    if notes_path is None or not Path(notes_path).exists():
        print("ERROR: No notes file found. Run 'pipeline --stages prepare_notes' first.")
        sys.exit(1)

    if str(notes_path).endswith(".parquet"):
        notes_df = pd.read_parquet(notes_path)
    else:
        notes_df = pd.read_csv(notes_path, low_memory=False)

    print(f"Loaded {len(notes_df)} notes from {notes_path}")

    # Filter to cohort if available
    from .agents.pipeline import _load_cohort_ids
    cohort_ids = _load_cohort_ids(output_dir)
    if cohort_ids is not None and ext_config.patient_id_column in notes_df.columns:
        cohort_set = set(str(x) for x in cohort_ids)
        notes_df = notes_df[notes_df[ext_config.patient_id_column].astype(str).isin(cohort_set)]
        print(f"Filtered to cohort: {len(notes_df)} notes")

    from .training.grpo_trainer import run_grpo_training
    run_grpo_training(config, notes_df)


if __name__ == "__main__":
    main()
