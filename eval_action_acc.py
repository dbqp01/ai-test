from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
parser.add_argument("--adapter", default=None)
parser.add_argument("--valid", type=Path, default=Path("data/student-17b.valid.jsonl"))
parser.add_argument("--limit", type=int, default=250)
parser.add_argument("--max-new-tokens", type=int, default=96)
args = parser.parse_args()
if args.adapter and args.adapter.lower() == "none":
    # "none" es literal en nuestras líneas de lanzamiento. Sin normalizar, PeftModel intenta
    # cargar un adapter en la ruta "none" y el tokenizer también (ambos usan args.adapter),
    # así que las dos filas de control sin adapter se caían y no había basal con la que
    # comparar — el único modo de demostrar que el LoRA aporta algo.
    args.adapter = None

try:
    from modelload import load_base

    model = load_base(args.model, torch_dtype=torch.bfloat16)
except Exception:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter)
model = model.to("cuda").eval()

tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, use_fast=True)
model.config.use_cache = True

rows = [json.loads(line) for line in args.valid.read_text(encoding="utf-8").splitlines() if line.strip()][: args.limit]
counters: Counter[str] = Counter()
errors: list[dict[str, object]] = []

for row in rows:
    messages = row["messages"][:-1]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    match = re.search(r"\{.*\}", generated, flags=re.DOTALL)
    predicted = None
    if match:
        try:
            value = json.loads(match.group(0))
            predicted = value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            predicted = None
    expected = json.loads(row["messages"][-1]["content"])
    counters["total"] += 1
    if predicted is None:
        counters["invalid_json"] += 1
    else:
        counters["valid_json"] += 1
        if predicted.get("decision") == expected.get("decision"):
            counters["decision_correct"] += 1
            if predicted.get("action_index") == expected.get("action_index"):
                counters["exact_action"] += 1
    if len(errors) < 12 and (
        predicted is None
        or predicted.get("decision") != expected.get("decision")
        or predicted.get("action_index") != expected.get("action_index")
    ):
        errors.append({"expected": expected, "predicted": predicted, "generated": generated[:300]})

total = max(1, counters["total"])
result = {
    **counters,
    "json_rate": counters["valid_json"] / total,
    "decision_accuracy": counters["decision_correct"] / total,
    "exact_action_accuracy": counters["exact_action"] / total,
    "_model": args.model,
    "_adapter": args.adapter,
    "_rows": total,
    "errors": errors,
}
print("ACC " + json.dumps(result, ensure_ascii=False), flush=True)
