"""Command-line interface for Talk-to-Data."""

import argparse
import asyncio
import logging
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="talk-to-data",
        description="Talk-to-Data: Build agentic clinical dataset query systems.",
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

    # discover
    p_discover = subparsers.add_parser("discover", help="Run the field discovery agent")
    p_discover.add_argument("data_paths", nargs="+", help="Files and/or directories with source data")
    p_discover.add_argument("--ontologies", nargs="+", default=["naaccr"], help="Ontology IDs to match against")
    p_discover.add_argument("--output", default=None, help="Path to save discovered field mappings")
    p_discover.add_argument("--max-budget", type=float, default=10.0, help="Maximum agent budget in USD (default: 10.0)")

    # metadata
    p_meta = subparsers.add_parser("metadata", help="Generate schema and summary metadata from database")
    p_meta.add_argument("config", help="Path to project YAML config")

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
        run_setup_agent(
            data_paths=args.data_paths or None,
            output_dir=args.output_dir,
            config_path=args.config,
            max_budget_usd=args.max_budget,
        )

    elif args.command == "discover":
        from .agents.discovery import run_discovery_agent
        asyncio.run(run_discovery_agent(
            data_paths=args.data_paths,
            ontology_ids=args.ontologies,
            output_config_path=args.output,
            max_budget_usd=args.max_budget,
        ))

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
