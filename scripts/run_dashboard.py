#!/usr/bin/env python
"""CLI: build, or serve, the live status page of a ``synthdata-run`` run.

``synthdata-run`` already rewrites ``<UTC time>_dashboard.html`` next to its
log while it runs. This command builds the same page on demand: for a run
started with ``--dashboard-interval 0``, for an older or finished run, or
after copying a run's ``logs/`` folder to another machine.

With ``--serve`` it rebuilds the page on every request and serves it at
http://127.0.0.1:8765/ (only the page: no other file of the run is served).
On a remote machine, forward the port over SSH and open that address on
your own computer:

    ssh -L 8765:localhost:8765 augur
    synthdata-dashboard --config configs/config_sim.yaml --serve

Usage:
    synthdata-dashboard --config CONFIG [--log RUN_LOG] [--serve] [--port 8765]
                        [--bind 127.0.0.1] [--refresh SECONDS]
"""

import argparse
import contextlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from synthdata.config import load_config
from synthdata.dashboard import build, latest_log, write_dashboard


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(description="Build or serve a run's live status page.")
    parser.add_argument("--config", required=True, help="The YAML config the run used.")
    parser.add_argument(
        "--log",
        default=None,
        help="A *_run.log file (default: the newest one in <output root>/logs/; "
        "with --serve, the newest one at each request).",
    )
    parser.add_argument(
        "--serve", action="store_true", help="Serve the page instead of writing it."
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve (default 8765).")
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Address for --serve (default 127.0.0.1: this machine only; reach it over SSH).",
    )
    parser.add_argument(
        "--refresh", type=int, default=30, help="Seconds between page reloads (default 30)."
    )
    return parser.parse_args(argv)


def _find_log(args, logs_dir: Path) -> Path | None:
    return Path(args.log) if args.log else latest_log(logs_dir)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config)
    from scripts.run_pipeline import log_dir

    logs_dir = log_dir(cfg)
    if not args.serve:
        log = _find_log(args, logs_dir)
        if log is None or not log.exists():
            sys.exit(f"No run log found ({args.log or logs_dir / '*_run.log'}).")
        print(write_dashboard(log, cfg, args.refresh))
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            log = _find_log(args, logs_dir)
            if log is None or not log.exists():
                self.send_error(404, "No run log found yet")
                return
            page = build(log, cfg, args.refresh).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *_):  # keep the terminal quiet
            pass

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"Serving the run dashboard at http://{args.bind}:{args.port}/ (Ctrl+C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
