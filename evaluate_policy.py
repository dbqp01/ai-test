from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def parse_json(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the site2tools policy adapter")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/site2tools-policy-lora"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--valid", type=Path, default=Path("data/policy.valid.jsonl"))
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/PyTorch no detecta GPU")
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), use_fast=True)
    try:
        from transformers import AutoModelForImageTextToText
        try:
            base = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        except ValueError:
            base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    except ImportError:
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, str(args.adapter)).eval()
    rows = [json.loads(line) for line in args.valid.read_text(encoding="utf-8").splitlines() if line.strip()][:args.limit]
    counters = Counter()
    errors: list[dict[str, object]] = []
    for row in rows:
        messages = row["messages"][:-1]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = parse_json(generated)
        expected = json.loads(row["messages"][-1]["content"])
        counters["total"] += 1
        if predicted is None:
            counters["invalid_json"] += 1
        else:
            counters["valid_json"] += 1
            if predicted.get("decision") == expected.get("decision"):
                counters["decision_correct"] += 1
            if predicted.get("action_index") == expected.get("action_index") and predicted.get("decision") == expected.get("decision"):
                counters["exact_action"] += 1
        if predicted is None or predicted.get("decision") != expected.get("decision") or (
            "action_index" in expected and predicted.get("action_index") != expected.get("action_index")
        ):
            if len(errors) < 20:
                errors.append({"expected": expected, "predicted": predicted, "generated": generated[:500]})
    total = max(1, counters["total"])
    result = {
        **counters,
        "json_rate": counters["valid_json"] / total,
        "decision_accuracy": counters["decision_correct"] / total,
        "exact_action_accuracy": counters["exact_action"] / total,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
