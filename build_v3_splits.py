from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_hash(row: dict) -> str:
    for m in row.get("messages", []):
        if m["role"] == "user":
            return hashlib.md5(m["content"].encode("utf-8")).hexdigest()
    return hashlib.md5(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()


def decision(row: dict) -> str:
    try:
        return json.loads(row["messages"][-1]["content"]).get("decision")
    except Exception:
        return "?"


v1_train = load(Path("data/student-17b.train.jsonl"))
v1_valid = load(Path("data/student-17b.valid.jsonl"))
policy_valid = load(Path("data/policy.valid.jsonl"))

train_hashes = {prompt_hash(r) for r in v1_train}
print(f"v1 train={len(v1_train)} unicos={len(train_hashes)}  v1 valid={len(v1_valid)}")

universe: dict[str, dict] = {}
for row in v1_train + v1_valid:
    universe.setdefault(prompt_hash(row), row)
print(f"universo deduplicado={len(universe)}")

clean = {h: r for h, r in universe.items() if h not in train_hashes}
print(f"filas nunca vistas por v1={len(clean)}  decisiones={collections.Counter(decision(r) for r in clean.values()).most_common()}")

transfer_hashes = set(clean)
transfer: dict[str, dict] = {}
for row in policy_valid:
    h = prompt_hash(row)
    if h in train_hashes or h in transfer_hashes:
        continue
    if decision(row) in {"finish", "request_confirmation"}:
        transfer[h] = row
transfer_items = list(transfer.items())[:300]
print(f"transfer (finish/confirm, no vistas por v1)={len(transfer)} -> uso {len(transfer_items)}")

exclude = set(clean) | {h for h, _ in transfer_items}
train_out = [r for h, r in universe.items() if h not in exclude]
print(f"train v3={len(train_out)}  valid v3={len(clean)}  transfer={len(transfer_items)}")
print("dist decision train v3:", collections.Counter(decision(r) for r in train_out).most_common())


def dump(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"escrito {path} ({len(rows)})")


dump(Path("data/v3.train.jsonl"), train_out)
dump(Path("data/v3.valid.jsonl"), list(clean.values()))
dump(Path("data/v3.transfer.jsonl"), [r for _, r in transfer_items])
