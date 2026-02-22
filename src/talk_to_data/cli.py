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
        mcp = create_server_from_config(config)
        host = args.host or config.query.mcp_host
        port = args.port or config.query.mcp_port
        mcp.run(transport="streamable-http", host=host, port=port)

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


if __name__ == "__main__":
    main()
