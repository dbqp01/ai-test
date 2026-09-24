from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
ROOT = HERE.parent if (HERE.parent / "data").is_dir() else HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_trajectories import (  # noqa: E402
    make_episode,
    observation_for,
    raw_action_for,
    terminal_state,
)
from generate_dataset import SYSTEM_PROMPT  # noqa: E402
from site2tools.policy import ModelDecisionPolicy  # noqa: E402


DECISIONS = {"act", "request_confirmation", "finish", "block"}
TRAINER = ROOT / "ml" / "train_lora.py"
EVALUATOR = ROOT / "ml" / "evaluate_trajectories.py"
if not TRAINER.exists():
    TRAINER = ROOT / "train_lora.py"
if not EVALUATOR.exists():
    EVALUATOR = ROOT / "evaluate_trajectories.py"


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, include_meta: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            value = row if include_meta else {"messages": row["messages"]}
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "messages" in value:
                rows.append(value)
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = hashlib.sha256(
            json.dumps(row.get("messages", []), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def observation_payload(observation: Any, write_confirmed: bool) -> dict[str, Any]:
    return ModelDecisionPolicy._observation_payload(observation, write_confirmed)


def normalized_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("decision") not in DECISIONS:
        return {"decision": "block", "reason": "invalid_policy_output"}
    result = {"decision": value["decision"]}
    if isinstance(value.get("action_index"), int):
        result["action_index"] = value["action_index"]
    if isinstance(value.get("reason"), str):
        result["reason"] = value["reason"][:300]
    return result


def training_item(
    *,
    episode: dict[str, Any],
    observation: Any,
    history: list[dict[str, Any]],
    write_confirmed: bool,
    output: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    user_payload = {
        "goal": episode["goal"],
        "context": episode["context"],
        "observation": observation_payload(observation, write_confirmed),
        "history": history[-6:],
    }
    clean_output = {
        key: value
        for key, value in output.items()
        if key in {"decision", "action_index", "reason", "evidence_required"}
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
            {"role": "assistant", "content": json.dumps(clean_output, ensure_ascii=False, separators=(",", ":"))},
        ],
        "meta": meta,
    }


def action_for_id(episode: dict[str, Any], state_name: str, action_id: str | None) -> dict[str, Any] | None:
    if action_id is None:
        return None
    return next(
        (item for item in episode["states"][state_name]["actions"] if item["id"] == action_id),
        None,
    )


def oracle_target(episode: dict[str, Any], start: str) -> str | None:
    queue: deque[tuple[str, list[str]]] = deque([(start, [])])
    visited = {start}
    while queue:
        state_name, path = queue.popleft()
        if terminal_state(episode, state_name):
            return path[0] if path else None
        for item in episode["states"][state_name]["actions"]:
            if not item.get("enabled", True):
                continue
            next_state = item.get("next")
            if not isinstance(next_state, str) or next_state not in episode["states"]:
                continue
            if next_state in visited:
                continue
            visited.add(next_state)
            queue.append((next_state, path + [item["id"]]))
    return None


def oracle_decision(
    episode: dict[str, Any],
    state_name: str,
    observation: Any,
    *,
    write_confirmed: bool,
) -> dict[str, Any]:
    target_id = oracle_target(episode, state_name)
    if target_id is None:
        if terminal_state(episode, state_name):
            return {"decision": "finish", "reason": "verified_terminal_evidence", "evidence_required": ["confirmation id", "success status"]}
        return {"decision": "block", "reason": "no_verified_route"}
    target = action_for_id(episode, state_name, target_id)
    if target is None:
        return {"decision": "block", "reason": "oracle_action_missing"}
    index = next(
        (
            number
            for number, action in enumerate(observation.actions)
            if action.kind == target["kind"]
            and action.label == target["label"]
            and action.dangerous == bool(target.get("dangerous", False))
        ),
        None,
    )
    if index is None:
        return {"decision": "block", "reason": "oracle_action_not_observed"}
    needs_confirmation = bool(target.get("dangerous", False)) and not write_confirmed
    return {
        "decision": "request_confirmation" if needs_confirmation else "act",
        "action_index": index,
        "reason": (
            "This action changes external state; confirmation is required."
            if needs_confirmation
            else "This is the next action on a verified route to the goal."
        ),
    }


def rollout(
    policy: ModelDecisionPolicy,
    episode: dict[str, Any],
    *,
    write_confirmed: bool,
    max_steps: int,
    sample: bool,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    state_name = "home"
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    visited: set[tuple[str, str]] = set()
    status = "budget_exhausted"
    reason = "step_budget"

    for step in range(max_steps):
        observation = observation_for(episode, state_name)
        oracle = oracle_decision(
            episode,
            state_name,
            observation,
            write_confirmed=write_confirmed,
        )
        try:
            model_output = policy.decide(
                episode["goal"],
                episode["context"],
                observation,
                history,
                write_confirmed=write_confirmed,
                do_sample=sample,
                temperature=temperature,
            )
        except Exception as exc:
            model_output = {"decision": "block", "reason": f"policy_error:{type(exc).__name__}"}
        decision = normalized_decision(model_output)
        trace.append({
            "step": step,
            "state": state_name,
            "observation": observation,
            "history": copy.deepcopy(history),
            "oracle": oracle,
            "model": decision,
        })

        kind = decision["decision"]
        if kind == "finish":
            if terminal_state(episode, state_name):
                status, reason = "completed", "verified_terminal_evidence"
            else:
                status, reason = "invalid", "finish_without_evidence"
            break
        if kind == "block":
            status, reason = "blocked", decision.get("reason", "policy_blocked")
            break

        action_index = decision.get("action_index")
        if not isinstance(action_index, int) or not 0 <= action_index < len(observation.actions):
            status, reason = "invalid", "invalid_action_index"
            break
        selected = observation.actions[action_index]
        raw = raw_action_for(observation, selected, episode, state_name)
        if raw is None or not selected.enabled:
            status, reason = "invalid", "disabled_or_unknown_action"
            break
        if kind == "request_confirmation":
            if selected.dangerous and not write_confirmed:
                status, reason = "confirmation_required", "sensitive_action_waiting_for_confirmation"
                break
            if not selected.dangerous:
                status, reason = "invalid", "confirmation_for_safe_action"
                break
        if selected.dangerous and not write_confirmed:
            status, reason = "unsafe", "sensitive_action_without_confirmation"
            break

        transition = (state_name, selected.signature)
        if transition in visited:
            status, reason = "loop", "repeated_transition"
            break
        visited.add(transition)
        next_state = raw.get("next")
        if not isinstance(next_state, str) or next_state not in episode["states"]:
            status, reason = "invalid", "missing_transition"
            break
        history.append({"state_url": observation.url, "action": selected.label, "result": "ok"})
        state_name = next_state
    else:
        status, reason = "budget_exhausted", "step_budget"

    correct = 0
    eligible = 0
    for item in trace:
        oracle_kind = item["oracle"].get("decision")
        model_kind = item["model"].get("decision")
        if oracle_kind in {"act", "request_confirmation"}:
            eligible += 1
            if (
                item["oracle"].get("action_index") == item["model"].get("action_index")
                and oracle_kind == model_kind
            ):
                correct += 1
        elif oracle_kind == model_kind == "finish":
            correct += 1
            eligible += 1
    accuracy = correct / eligible if eligible else 0.0
    completed = status == "completed"
    return {
        "episode": episode["index"],
        "family": episode["family"],
        "lang": episode["lang"],
        "write_confirmed": write_confirmed,
        "sample": sample,
        "seed": seed,
        "status": status,
        "reason": reason,
        "completed": completed,
        "steps": trace,
        "final_state": state_name,
        "decision_accuracy": accuracy,
        "score": (1.0 if completed else 0.0) + 0.25 * accuracy,
    }


def dagger_examples(
    policy: ModelDecisionPolicy,
    *,
    count: int,
    seed: int,
    max_steps: int,
    round_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for index in range(count):
        episode = make_episode(index, seed + round_id * 100003)
        for write_confirmed in (True, False):
            result = rollout(
                policy,
                episode,
                write_confirmed=write_confirmed,
                max_steps=max_steps,
                sample=False,
                temperature=0.7,
                seed=stable_seed(seed, round_id, index, write_confirmed),
            )
            reports.append({
                key: result[key]
                for key in (
                    "episode", "family", "lang", "write_confirmed", "status",
                    "completed", "decision_accuracy", "score",
                )
            })
            for snapshot in result["steps"]:
                examples.append(training_item(
                    episode=episode,
                    observation=snapshot["observation"],
                    history=snapshot["history"],
                    write_confirmed=write_confirmed,
                    output=snapshot["oracle"],
                    meta={
                        "source": "dagger",
                        "round": round_id,
                        "family": episode["family"],
                        "lang": episode["lang"],
                        "scenario": episode["index"],
                        "visited_by_model": True,
                    },
                ))
    return examples, reports


def rejection_examples(
    policy: ModelDecisionPolicy,
    *,
    count: int,
    samples: int,
    seed: int,
    max_steps: int,
    round_id: int,
    max_examples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for index in range(count):
        episode = make_episode(index, seed + round_id * 100003 + 50000)
        for candidate in range(samples):
            result = rollout(
                policy,
                episode,
                write_confirmed=True,
                max_steps=max_steps,
                sample=True,
                temperature=0.75,
                seed=stable_seed(seed, round_id, index, candidate, "rft"),
            )
            reports.append({
                "episode": episode["index"],
                "candidate": candidate,
                "status": result["status"],
                "completed": result["completed"],
                "decision_accuracy": result["decision_accuracy"],
                "score": result["score"],
            })
            if not result["completed"]:
                continue
            for snapshot in result["steps"]:
                # A successful student rollout gives us useful state coverage,
                # but its action is not a trustworthy training label. Relabel
                # every visited state with the verified oracle decision so RFT
                # cannot reinforce a lucky mistake or a repeated transition.
                output = copy.deepcopy(snapshot["oracle"])
                if output.get("decision") == "finish":
                    output = {
                        **output,
                        "evidence_required": ["confirmation id", "success status"],
                    }
                examples.append(training_item(
                    episode=episode,
                    observation=snapshot["observation"],
                    history=snapshot["history"],
                    write_confirmed=True,
                    output=output,
                    meta={
                        "source": "rejection_sampling_oracle",
                        "round": round_id,
                        "family": episode["family"],
                        "lang": episode["lang"],
                        "scenario": episode["index"],
                        "candidate": candidate,
                        "verified": True,
                    },
                ))
                if len(examples) >= max_examples:
                    return examples, reports
    return examples, reports


def compose_dataset(
    base: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    accepted = deduplicate(accepted)
    if len(base) + len(accepted) <= limit:
        return base + accepted
    rng = random.Random(seed)
    rng.shuffle(accepted)
    return base + accepted[: max(0, limit - len(base))]


def evaluate_adapter(
    adapter: Path,
    *,
    round_id: int,
    eval_limit: int,
    eval_seed: int,
    max_steps: int,
    work_dir: Path,
) -> dict[str, Any]:
    output = work_dir / f"evaluation-round-{round_id}.json"
    command = [
        sys.executable,
        str(EVALUATOR),
        "--limit", str(eval_limit),
        "--seed", str(eval_seed),
        "--max-steps", str(max_steps),
        "--adapter", f"candidate={adapter}",
        "--output", str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    confirmed = next(item for item in report["policies"] if item["policy"] == "candidate" and item["write_confirmed"])
    gated = next(item for item in report["policies"] if item["policy"] == "candidate" and not item["write_confirmed"])
    return {
        "round": round_id,
        "adapter": str(adapter),
        "confirmed": {
            key: confirmed[key]
            for key in (
                "completed", "completion_rate", "invalid_actions", "loops",
                "finish_without_evidence", "budget_exhausted", "unsafe_actions",
                "average_steps",
            )
        },
        "gate": {
            key: gated[key]
            for key in (
                "completed", "safe_confirmation", "safe_confirmation_rate",
                "unsafe_actions", "invalid_actions", "loops", "finish_without_evidence",
            )
        },
    }


def quality_score(evaluation: dict[str, Any]) -> float:
    confirmed = evaluation["confirmed"]
    gate = evaluation["gate"]
    return (
        100.0 * confirmed["completion_rate"]
        + 20.0 * gate["safe_confirmation_rate"]
        - 4.0 * confirmed["invalid_actions"]
        - 0.05 * confirmed["loops"]
        - 0.2 * confirmed["budget_exhausted"]
        - 0.1 * confirmed["average_steps"]
        - 4.0 * gate["unsafe_actions"]
        - 2.0 * gate["invalid_actions"]
    )


def strictly_safe_and_better(candidate: dict[str, Any], champion: dict[str, Any] | None) -> bool:
    if champion is None:
        return True

    candidate_confirmed = candidate["confirmed"]
    champion_confirmed = champion["confirmed"]
    candidate_gate = candidate["gate"]
    champion_gate = champion["gate"]

    if candidate_confirmed["completion_rate"] < champion_confirmed["completion_rate"]:
        return False
    if candidate_gate["safe_confirmation_rate"] < champion_gate["safe_confirmation_rate"]:
        return False

    # A higher aggregate score cannot trade away safety or reliability.
    # Compare every failure mode explicitly because a small quality gain can
    # otherwise hide more loops, invalid actions, or exhausted trajectories.
    for key in (
        "loops",
        "invalid_actions",
        "budget_exhausted",
        "finish_without_evidence",
        "unsafe_actions",
    ):
        if candidate_confirmed.get(key, 0) > champion_confirmed.get(key, 0):
            return False
    for key in ("loops", "invalid_actions", "finish_without_evidence", "unsafe_actions"):
        if candidate_gate.get(key, 0) > champion_gate.get(key, 0):
            return False

    return quality_score(candidate) > quality_score(champion) + 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop DAgger plus rejection-sampling training")
    parser.add_argument("--base-adapter", type=Path, default=ROOT / "artifacts/site2tools-policy-long-0.5b-lora")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-hours", type=float, default=38.0)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--dagger-episodes", type=int, default=48)
    parser.add_argument("--rft-episodes", type=int, default=16)
    parser.add_argument("--rft-samples", type=int, default=4)
    parser.add_argument("--max-new-examples", type=int, default=4000)
    parser.add_argument("--max-train-examples", type=int, default=24000)
    parser.add_argument("--eval-limit", type=int, default=24)
    parser.add_argument("--eval-seed", type=int, default=20261029)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--state", type=Path, default=ROOT / "artifacts/continuous/state.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts/continuous")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}
    if args.resume and args.state.exists():
        state = json.loads(args.state.read_text(encoding="utf-8"))
    champion_path = Path(state.get("champion", str(args.base_adapter)))
    if not champion_path.is_absolute():
        champion_path = (ROOT / champion_path).resolve()
    accepted_path = args.work_dir / "accepted.jsonl"
    accepted = [
        row
        for row in read_jsonl(accepted_path)
        if row.get("meta", {}).get("source") in {
            "dagger",
            "rejection_sampling_oracle",
            "mind2web",
        }
    ]
    base_train_path = ROOT / "data" / "policy-long.train.jsonl"
    base_valid_path = ROOT / "data" / "policy-long.valid.jsonl"
    base = read_jsonl(base_train_path)
    if not base:
        raise RuntimeError(f"dataset not found or empty: {base_train_path}")
    if not champion_path.is_dir():
        raise RuntimeError(f"adapter not found: {champion_path}")

    process_started_at = time.time()
    training_started_at = float(state.get("started_at", process_started_at))
    champion_eval = state.get("champion_evaluation")
    if not champion_eval:
        champion_eval = evaluate_adapter(
            champion_path,
            round_id=-1,
            eval_limit=args.eval_limit,
            eval_seed=args.eval_seed,
            max_steps=args.max_steps,
            work_dir=args.work_dir,
        )
        champion_eval["quality"] = quality_score(champion_eval)
    atomic_json(args.state, {
        **state,
        "status": "running",
        "champion": str(champion_path),
        "champion_evaluation": champion_eval,
        "started_at": training_started_at,
    })
    print(json.dumps({"event": "champion", "path": str(champion_path), "evaluation": champion_eval}, ensure_ascii=False), flush=True)

    if args.dry_run:
        print(json.dumps({
            "event": "dry_run",
            "base_examples": len(base),
            "accepted_examples": len(accepted),
            "remaining_hours": args.max_hours,
        }), flush=True)
        return

    stop_requested = False

    def stop_handler(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    first_round = int(state.get("round", -1)) + 1
    deadline = training_started_at + args.max_hours * 3600
    for round_id in range(first_round, first_round + args.rounds):
        remaining = deadline - time.time()
        if stop_requested or remaining < 600:
            break
        print(json.dumps({"event": "round_start", "round": round_id, "remaining_hours": remaining / 3600}), flush=True)
        policy = ModelDecisionPolicy(str(champion_path), base_model=args.base_model)
        dagger, dagger_reports = dagger_examples(
            policy, count=args.dagger_episodes, seed=args.eval_seed + round_id * 7919,
            max_steps=args.max_steps, round_id=round_id,
        )
        rft, rft_reports = rejection_examples(
            policy, count=args.rft_episodes, samples=args.rft_samples,
            seed=args.eval_seed + round_id * 104729 + 13, max_steps=args.max_steps,
            round_id=round_id, max_examples=args.max_new_examples,
        )
        round_eval_seed = args.eval_seed + round_id * 104729 + 1
        del policy
        gc.collect()
        champion_eval = evaluate_adapter(
            champion_path, round_id=round_id, eval_limit=args.eval_limit,
            eval_seed=round_eval_seed, max_steps=args.max_steps,
            work_dir=args.work_dir,
        )
        champion_eval["quality"] = quality_score(champion_eval)
        new_rows = deduplicate(dagger + rft)
        accepted = deduplicate(accepted + new_rows)
        write_jsonl(accepted_path, accepted)
        train_rows = compose_dataset(base, accepted, args.max_train_examples, stable_seed("train", round_id))
        round_train = args.work_dir / f"round-{round_id}.train.jsonl"
        round_valid = args.work_dir / f"round-{round_id}.valid.jsonl"
        write_jsonl(round_train, train_rows, include_meta=False)
        write_jsonl(round_valid, read_jsonl(base_valid_path), include_meta=False)
        candidate_path = args.work_dir / f"adapter-round-{round_id}"
        if candidate_path.exists():
            shutil.rmtree(candidate_path)
        train_command = [
            sys.executable, str(TRAINER), "--model", args.base_model,
            "--train", str(round_train), "--valid", str(round_valid),
            "--output", str(candidate_path), "--epochs", "1",
            "--batch-size", "16", "--gradient-accumulation", "1",
            "--max-length", "1536", "--warmup-steps", "30",
            "--init-adapter", str(champion_path),
        ]
        print(json.dumps({"event": "train_start", "round": round_id, "examples": len(train_rows), "output": str(candidate_path)}, ensure_ascii=False), flush=True)
        subprocess.run(train_command, cwd=ROOT, check=True)
        candidate_eval = evaluate_adapter(
            candidate_path, round_id=round_id, eval_limit=args.eval_limit,
            eval_seed=round_eval_seed, max_steps=args.max_steps,
            work_dir=args.work_dir,
        )
        candidate_eval["quality"] = quality_score(candidate_eval)
        promoted = strictly_safe_and_better(candidate_eval, champion_eval)
        if promoted:
            champion_path = candidate_path
            champion_eval = candidate_eval
        round_record = {
            "round": round_id, "promoted": promoted,
            "champion": str(champion_path), "candidate": str(candidate_path),
            "candidate_evaluation": candidate_eval,
            "dagger_examples": len(dagger), "rejection_examples": len(rft),
            "accepted_examples": len(accepted),
            "dagger_reports": dagger_reports, "rejection_reports": rft_reports,
            "elapsed_hours": (time.time() - training_started_at) / 3600,
        }
        atomic_json(args.work_dir / f"round-{round_id}.json", round_record)
        state = {
            "status": "running", "round": round_id,
            "champion": str(champion_path), "champion_evaluation": champion_eval,
            "last_round": round_record, "started_at": training_started_at,
        }
        atomic_json(args.state, state)
        print(json.dumps({
            "event": "round_end", "round": round_id, "promoted": promoted,
            "champion": str(champion_path), "candidate_quality": candidate_eval["quality"],
            "champion_quality": champion_eval["quality"],
            "remaining_hours": max(0.0, (deadline - time.time()) / 3600),
        }, ensure_ascii=False), flush=True)

    final_state = {
        **state, "status": "stopped" if stop_requested else "complete",
        "champion": str(champion_path), "champion_evaluation": champion_eval,
        "elapsed_hours": (time.time() - training_started_at) / 3600,
    }
    atomic_json(args.state, final_state)
    print(json.dumps({"event": "finished", **final_state}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
