from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
parser.add_argument("--adapter", default=None)
parser.add_argument("--valid", type=Path, default=Path("data/student-17b.valid.jsonl"))
parser.add_argument("--max-length", type=int, default=1536)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

try:
    from modelload import load_base

    model = load_base(args.model, torch_dtype=torch.bfloat16)
except Exception:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.use_cache = False

if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

dataset = load_dataset("json", data_files={"validation": str(args.valid)})["validation"]
if args.limit:
    dataset = dataset.select(range(min(args.limit, len(dataset))))


def format_item(item: dict[str, object]) -> dict[str, str]:
    try:
        text = tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
    return {"text": text}


def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
    return tokenizer(batch["text"], max_length=args.max_length, truncation=True)


dataset = dataset.map(format_item, remove_columns=dataset.column_names)
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="/tmp/eval-only",
    per_device_eval_batch_size=args.batch_size,
    bf16=True,
    eval_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    logging_strategy="no",
)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    processing_class=tokenizer,
)
metrics = dict(trainer.evaluate(eval_dataset=dataset))
metrics["_model"] = args.model
metrics["_adapter"] = args.adapter
metrics["_valid"] = str(args.valid)
metrics["_rows"] = len(dataset)
print("RESULT " + json.dumps(metrics, ensure_ascii=False), flush=True)
