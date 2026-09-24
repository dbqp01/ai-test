from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .core import UniversalOperator, WebsiteExplorer
from .policy import ModelDecisionPolicy


class Handler(BaseHTTPRequestHandler):
    server_version = "site2tools/0.1"

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json(200, {"ok": True, "service": "site2tools", "version": "0.1.0"})
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self.read_json()
            data_dir = Path(getattr(self.server, "data_dir"))
            if self.path == "/explore":
                graph = WebsiteExplorer(
                    payload["start_url"], data_dir,
                    max_states=int(payload.get("max_states", 25)),
                    max_depth=int(payload.get("max_depth", 3)),
                    allowed_hosts=payload.get("allowed_hosts"),
                ).explore()
                self.send_json(200, graph)
                return
            if self.path == "/operate":
                result = UniversalOperator(
                    payload["start_url"], data_dir,
                    max_steps=int(payload.get("max_steps", 12)),
                    allowed_hosts=payload.get("allowed_hosts"),
                    policy=getattr(self.server, "policy", None),
                ).run(
                    goal=payload["goal"],
                    context=payload.get("context"),
                    confirm=bool(payload.get("confirm", False)),
                    allow_writes=bool(payload.get("allow_writes", False)),
                )
                self.send_json(200, result)
                return
            self.send_json(404, {"error": "not_found"})
        except KeyError as exc:
            self.send_json(400, {"error": "missing_field:" + str(exc.args[0])})
        except Exception as exc:
            self.send_json(500, {"error": type(exc).__name__, "detail": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal website operator MVP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--policy-adapter", type=Path, default=None)
    parser.add_argument("--policy-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--policy-device", default="cuda")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.data_dir = args.data.resolve()
    server.policy = None
    if args.policy_adapter:
        server.policy = ModelDecisionPolicy(
            str(args.policy_adapter),
            base_model=args.policy_model,
            device=args.policy_device,
        )
    print("site2tools escuchando en http://" + args.host + ":" + str(args.port))
    print("datos: " + str(server.data_dir))
    server.serve_forever()


if __name__ == "__main__":
    main()
