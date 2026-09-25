"""Convert Mind2Web train steps into harness-format policy rows.

Differences vs convert_mind2web.py: same JSON observation the server actually sends,
labels resolved from the element's real text (not its HTML tag), candidates restricted to
visible boxes, and a split by domain so no site appears in both train and valid.
"""
from __future__ import annotations

import glob
import html
import json
import random
import re
import sys
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

MAX_CANDIDATES = 20
TEXT_CHARS = 300
SEED = 1234

DANGEROUS = {
    "submit", "send", "enviar", "entregar", "delete", "eliminar", "remove",
    "buy", "purchase", "pagar", "pay", "publish", "publicar", "post", "transfer",
    "confirm", "confirmar",
}


class TextIndexer(HTMLParser):
    """Map backend_node_id -> visible text and <title>, in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: dict[str, str] = defaultdict(str)
        self.title = ""
        self._stack: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        node = dict(attrs).get("backend_node_id", "")
        if node:
            self._stack.append(node)
        else:
            self._stack.append(self._stack[-1] if self._stack else "")
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if self._stack:
            self._stack.pop()
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        chunk = clean(data)
        if not chunk:
            return
        if self._in_title:
            self.title = (self.title + " " + chunk).strip()[:120]
        node = self._stack[-1] if self._stack else ""
        if node:
            self.text[node] = (self.text[node] + " " + chunk).strip()


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def attrs_of(candidate: dict) -> dict:
    try:
        return json.loads(candidate.get("attributes") or "{}")
    except json.JSONDecodeError:
        return {}


def visible(candidate: dict) -> bool:
    box = attrs_of(candidate).get("bounding_box_rect", "")
    parts = [float(x) for x in box.split(",") if x.replace(".", "").isdigit()]
    if len(parts) == 4:
        return parts[2] > 1 and parts[3] > 1
    return True


def describe(candidate: dict, texts: dict[str, str]) -> str:
    attrs = attrs_of(candidate)
    node = str(candidate.get("backend_node_id", ""))
    for key in ("aria_label", "aria-label", "placeholder", "name", "title", "value", "id"):
        value = clean(attrs.get(key, ""))
        if value and not value.startswith("data-"):
            return value[:60]
    text = clean(texts.get(node, ""))
    if text:
        return text[:60]
    return ""


def kind_of(candidate: dict) -> tuple[str, str, str]:
    tag = candidate.get("tag", "a").lower()
    if tag in {"input", "textarea", "select"}:
        input_type = attrs_of(candidate).get("type", "text") if tag == "input" else (
            "select-one" if tag == "select" else "textarea"
        )
        return "input", input_type, tag
    return "click", "", tag


def role_of(tag: str) -> str:
    return {"a": "link", "button": "button", "select": "combobox", "textarea": "textbox"}.get(
        tag, "textbox" if tag == "input" else tag
    )


def build_rows(path: Path, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for task in json.load(path.open(encoding="utf-8")):
        goal = clean(task.get("confirmed_task", ""))
        # "domain" en este subconjunto trae la CATEGORIA (Travel/Shopping/...), no el sitio:
        # agrupar por él deja 3 grupos y un valid mayor que el train. El sitio real es "website".
        site = clean(task.get("website", "")) or clean(task.get("subdomain", ""))
        steps = task.get("actions") or []
        reprs = task.get("action_reprs") or []
        history: list[dict] = []
        for index, step in enumerate(steps[:-1]):
            pos = step.get("pos_candidates") or []
            neg = step.get("neg_candidates") or []
            if not pos or not neg:
                continue
            if len(pos) > 1:
                # Hay pasos con decenas de positivos (y alguno con 3109). El objetivo real es
                # el marcado como original, y si no hay uno solo, el paso no es evaluable.
                original = [c for c in pos if c.get("is_original_target")]
                if len(original) != 1:
                    continue
                pos = original
            indexer = TextIndexer()
            try:
                indexer.feed(step.get("cleaned_html") or "")
            except Exception:
                continue
            pool = [c for c in neg if visible(c) and describe(c, indexer.text)]
            keep = pos + rng.sample(pool, min(MAX_CANDIDATES - 1, len(pool)))
            rng.shuffle(keep)
            actions = []
            gold_index = None
            inputs: dict[str, str] = {}
            for position, candidate in enumerate(keep):
                label = describe(candidate, indexer.text)
                kind, input_type, tag = kind_of(candidate)
                attrs = attrs_of(candidate)
                name = clean(attrs.get("name", "") or attrs.get("id", ""))[:40]
                if candidate is pos[0]:
                    gold_index = position
                    operation = step.get("operation") or {}
                    value = clean(str(operation.get("value", "")))
                    if kind == "input" and value:
                        inputs[name or label] = value
                actions.append({
                    "index": position,
                    "action_id": f"s2t-{position}",
                    "kind": kind,
                    "label": label,
                    "role": role_of(tag),
                    "input_type": input_type,
                    "name": name,
                    "placeholder": clean(attrs.get("placeholder", ""))[:40],
                    "dangerous": bool(set(label.lower().split()) & DANGEROUS),
                    "enabled": not attrs.get("disabled"),
                })
            if gold_index is None or len(actions) < 4:
                continue
            full_text = " ".join(indexer.text.values())
            gold = {"decision": "act", "reason": "This is the next action that advances the goal.",
                    "action_index": gold_index}
            payload = {
                "goal": goal,
                "context": ({"inputs": inputs} if inputs else {}),
                "observation": {
                    "url": "/" + site,
                    "title": indexer.title,
                    "text": full_text[:TEXT_CHARS],
                    "actions": actions,
                    "permissions": {"write_confirmed": False},
                },
                "history": history[-6:],
            }
            system = (
                "You are the decision policy inside a universal website operator.\n"
                "Choose the next step from the observed actions for the user's goal.\n"
                "Return exactly one JSON object and no markdown.\n"
                "Allowed decisions: act, request_confirmation, finish, block.\n"
                "Use action_index only when decision is act or request_confirmation.\n"
                "Never claim finish without observable evidence. Never perform a sensitive write "
                "without write_confirmed=true.\n"
                "If a required file, credential, or prerequisite is missing, choose block and "
                "explain the missing prerequisite.\n"
            )
            rows.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                    {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))},
                ],
                "meta": {"decision": "act", "target_action_index": gold_index, "site": site,
                         "kind": "mind2web-v2", "n_candidates": len(actions), "step": index},
            })
            label = describe(pos[0], indexer.text)
            history.append({"state_url": "/" + site, "action": label or (reprs[index] if index < len(reprs) else "")})
    return rows


def main() -> None:
    out_train = Path(sys.argv[1] if len(sys.argv) > 1 else "data/m2w2.train.jsonl")
    out_valid = Path(sys.argv[2] if len(sys.argv) > 2 else "data/m2w2.valid.jsonl")
    files = sorted(glob.glob("/root/huggingface/mind2web-full/data/train/*.json"))
    rng = random.Random(SEED)
    by_site: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        try:
            for row in build_rows(Path(path), rng):
                by_site[row["meta"]["site"]].append(row)
        except Exception as exc:
            print(f"skip {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    sites = sorted(by_site)
    rng.shuffle(sites)
    # Split balanceado por filas pero agrupado por sitio: un sitio nunca está en los dos lados.
    total = sum(len(v) for v in by_site.values())
    valid_sites: set[str] = set()
    taken = 0
    for site in sites:
        if taken >= total // 10:
            break
        valid_sites.add(site)
        taken += len(by_site[site])
    train_sites = set(sites) - valid_sites

    def dump(path: Path, site_subset) -> int:
        n = 0
        with path.open("w", encoding="utf-8") as handle:
            for site in sorted(site_subset):
                for row in by_site[site]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
        return n

    n_valid = dump(out_valid, valid_sites)
    n_train = dump(out_train, train_sites)
    counts = [row["meta"]["n_candidates"] for d in by_site for row in by_site[d]]
    counts.sort()
    print(json.dumps({
        "train_rows": n_train, "valid_rows": n_valid,
        "sitios": len(sites), "sitios_en_valid": len(valid_sites),
        "candidatos_mediana": counts[len(counts) // 2],
        "candidatos_p90": counts[int(len(counts) * 0.9)],
        "candidatos_max": counts[-1],
    }))


if __name__ == "__main__":
    main()
