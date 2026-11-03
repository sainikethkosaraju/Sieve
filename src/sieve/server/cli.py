"""``sieve serve`` entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sieve", description="Sieve confidence-gated LLM cascade")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--config", required=True, help="path to the YAML config")
    serve.add_argument("--host", help="override bind host")
    serve.add_argument("--port", type=int, help="override bind port")
    serve.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from sieve.server.app import create_app
        from sieve.server.config import ServerSettings

        overrides: dict[str, object] = {}
        if args.host:
            overrides["host"] = args.host
        if args.port:
            overrides["port"] = args.port
        try:
            settings = ServerSettings.from_yaml(args.config, **overrides)
            app = create_app(settings)
        except Exception as e:  # config / artifact / pin errors: refuse to start
            print(f"sieve: refusing to start: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=args.log_level,
            timeout_keep_alive=5,
            limit_concurrency=settings.limits.max_concurrency + settings.limits.max_queue + 64,
            h11_max_incomplete_event_size=16 * 1024,
            server_header=False,
        )
        return 0
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
