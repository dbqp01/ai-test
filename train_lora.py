from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModel, AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine tune the site2tools decision policy")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("data/policy.train.jsonl"))
    parser.add_argument("--valid", type=Path, default=Path("data/policy.valid.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/site2tools-policy-lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="Existing LoRA adapter to continue training instead of starting a new adapter",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/PyTorch no detecta una GPU; no iniciaré un entrenamiento costoso en CPU")
    device_name = torch.cuda.get_device_name(0)
    print(json.dumps({"device": device_name, "hip": torch.version.hip, "torch": torch.__version__}), flush=True)

    dataset = load_dataset("json", data_files={"train": str(args.train), "validation": str(args.valid)})
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def format_item(item: dict[str, object]) -> dict[str, str]:
        try:
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    formatted = dataset.map(format_item, remove_columns=dataset["train"].column_names)

    try:
        from modelload import load_base
        try:
            model = load_base(args.model, torch_dtype=torch.bfloat16)
        except ValueError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    if args.init_adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
        print(json.dumps({"initialized_from_adapter": str(args.init_adapter)}), flush=True)
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(batch["text"], max_length=args.max_length, truncation=True)

    tokenized = formatted.map(tokenize, batched=True, remove_columns=["text"])
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    metrics = dict(result.metrics)
    metrics.update({"model": args.model, "device": device_name, "hip": torch.version.hip, "torch": torch.__version__})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
