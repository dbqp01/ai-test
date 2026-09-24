from __future__ import annotations

import json
import re
from typing import Any


SYSTEM_PROMPT = """You are the decision policy inside a universal website operator.
Choose the next step from the observed actions for the user's goal.
Return exactly one JSON object and no markdown.
Allowed decisions: act, request_confirmation, finish, block.
Use action_index only when decision is act or request_confirmation.
Never claim finish without observable evidence. Never perform a sensitive write without write_confirmed=true.
If a required file, credential, or prerequisite is missing, choose block and explain the missing prerequisite.
"""


def parse_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


class ModelDecisionPolicy:
    """Loads a LoRA policy and turns observations into validated JSON proposals."""

    def __init__(
        self,
        adapter_path: str,
        *,
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        max_input_tokens: int = 2048,
        max_new_tokens: int = 96,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self.adapter_path = adapter_path
        self.base_model_name = base_model
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            from modelload import load_base
            try:
                base = load_base(base_model, dtype=dtype).to(device)
            except ValueError:
                base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
        except ImportError:
            base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
        self.model = PeftModel.from_pretrained(base, adapter_path).eval()

    @staticmethod
    def _observation_payload(observation: Any, write_confirmed: bool) -> dict[str, Any]:
        return {
            "url": observation.url,
            "title": observation.title,
            "text": observation.text[:12000],
            "actions": [
                {
                    "index": index,
                    "action_id": action.action_id,
                    "kind": action.kind,
                    "label": action.label,
                    "role": action.role,
                    "input_type": action.input_type,
                    "name": action.name,
                    "placeholder": action.placeholder,
                    "dangerous": action.dangerous,
                    "enabled": action.enabled,
                }
                for index, action in enumerate(observation.actions)
            ],
            "permissions": {"write_confirmed": write_confirmed},
        }

    def decide(
        self,
        goal: str,
        context: dict[str, Any],
        observation: Any,
        history: list[dict[str, Any]],
        *,
        write_confirmed: bool,
    ) -> dict[str, Any] | None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": goal,
                        "context": context,
                        "observation": self._observation_payload(observation, write_confirmed),
                        "history": history[-6:],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        with __import__("torch").inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        decision = parse_json_object(generated)
        if decision is not None:
            decision["_raw"] = generated[:1000]
        return decision
