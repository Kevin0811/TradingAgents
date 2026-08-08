#!/usr/bin/env python
"""Entry point script to run the TradingAgents API server.

Usage:
    python run_server.py                    # Default: localhost:8000
    python run_server.py --host 0.0.0.0    # Bind to all interfaces
    python run_server.py --port 8080       # Custom port
    python run_server.py --reload          # Enable auto-reload for development

Or use uvicorn directly:
    uvicorn tradingagents.api:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import sys
import os

# Add project root to path if run as script
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    """Parse arguments and start the server."""
    parser = argparse.ArgumentParser(
        description="Run the TradingAgents API server",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed. "
            "Install it with: pip install uvicorn[standard] "
            "or: pip install -e '.[api]'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from tradingagents.api import create_app
    except ImportError as e:
        print(
            f"Error: Failed to import tradingagents.api: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.workers > 1 and not args.reload:
        print(
            f"Warning: --workers {args.workers} is not supported by the task API.\n"
            "  Task state lives in process memory, so a task created in one worker\n"
            "  returns 404 when the status poll lands on another, and the effective\n"
            "  concurrency becomes task_max_concurrent x workers.\n"
            "  Raise TRADINGAGENTS_TASK_MAX_CONCURRENT instead of adding workers.",
            file=sys.stderr,
        )

    print(f"Starting TradingAgents API server on {args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/api/v1/docs")
    print(f"Health check: http://{args.host}:{args.port}/health")

    uvicorn.run(
        "tradingagents.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,  # reload only supports 1 worker
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()