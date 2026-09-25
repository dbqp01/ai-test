from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import UniversalOperator, WebsiteExplorer
from .policy import ModelDecisionPolicy


def main() -> None:
    parser = argparse.ArgumentParser(prog="site2tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    explore = subparsers.add_parser("explore")
    explore.add_argument("start_url")
    explore.add_argument("--data", type=Path, default=Path("data"))
    explore.add_argument("--max-states", type=int, default=25)
    explore.add_argument("--max-depth", type=int, default=3)
    operate = subparsers.add_parser("operate")
    operate.add_argument("start_url")
    operate.add_argument("goal")
    operate.add_argument("--data", type=Path, default=Path("data"))
    operate.add_argument("--max-steps", type=int, default=12)
    operate.add_argument("--confirm", action="store_true")
    operate.add_argument("--allow-writes", action="store_true")
    operate.add_argument("--policy-adapter", type=Path, default=None)
    operate.add_argument("--policy-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    operate.add_argument("--policy-device", default="cuda")
    args = parser.parse_args()
    if args.command == "explore":
        result = WebsiteExplorer(args.start_url, args.data, args.max_states, args.max_depth).explore()
    else:
        policy = None
        if args.policy_adapter:
            policy = ModelDecisionPolicy(
                str(args.policy_adapter),
                base_model=args.policy_model,
                device=args.policy_device,
            )
        result = UniversalOperator(args.start_url, args.data, args.max_steps, policy=policy).run(
            args.goal, confirm=args.confirm, allow_writes=args.allow_writes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
