"""Prueba de despliegue real en la RTX 3050 6 GB del portátil.

Lo que tiene que responder esta prueba, y nada más:
  1) cabe Qwen3-1.7B en bf16 + el adapter v4 en 6 GB con contexto 2048,
  2) a cuántos tokens/s genera,
  3) si emite una decisión válida en el FORMATO QUE SIRVE EL SERVIDOR (no el de juguete).
Usa `dtype=` porque transformers 5.x deprecó `torch_dtype`, igual que se aprendió en la VM.
"""
import json
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-1.7B"
ADAPTER = "adapter-v4"

row = json.loads(open("row.jsonl", encoding="utf-8").read())
messages = row["messages"][:-1]

# Segundo caso: el peor realista para la memoria — menú de 10 acciones y 1500 caracteres de
# texto de página, que es exactamente el techo que fija rank_actions/text_chars en core.py.
obs = json.loads(messages[1]["content"])
obs["observation"]["actions"] = (obs["observation"]["actions"] * 4)[:10]
for i, a in enumerate(obs["observation"]["actions"]):
    a["index"] = i
    a["action_id"] = "s2t-%d" % i
    a["label"] = "Room type %d with a long descriptive label including amenities" % i
obs["observation"]["text"] = (obs["observation"]["text"] + " ") * 200
obs["observation"]["text"] = obs["observation"]["text"][:1500]
hard = [messages[0], {"role": "user", "content": json.dumps(obs, ensure_ascii=False, separators=(",", ":"))}]

print("cargando tokenizer y modelo base en bf16...")
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda")
torch.cuda.synchronize()
peak_load = torch.cuda.max_memory_allocated() / 1e9
print("base cargada | VRAM reservada tras base: %.2f GB" % peak_load)

model = PeftModel.from_pretrained(model, ADAPTER).eval()
torch.cuda.synchronize()
peak_adapter = torch.cuda.max_memory_allocated() / 1e9
print("adapter v4 montado | VRAM: %.2f GB (libre teórica %.2f GB)" % (
    peak_adapter, torch.cuda.get_device_properties(0).total_memory / 1e9 - peak_adapter))

for name, msgs in (("holdout real", messages), ("peor caso 10 acciones", hard)):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    inp = tok(prompt, return_tensors="pt").to("cuda")
    n_ctx = inp["input_ids"].shape[1]
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    # transformers 5 puede devolver [batch, seq] o directamente [seq]; se normaliza antes de medir.
    first = out[0] if out.dim() > 1 else out
    gen = tok.decode(first[n_ctx:], skip_special_tokens=True)
    torch.cuda.synchronize()
    new_tokens = int(first.numel()) - n_ctx
    print("--- %s | contexto %d tokens | %.1f tok/s | pico %.2f GB" % (
        name, n_ctx, new_tokens / max(dt, 1e-6), torch.cuda.max_memory_allocated() / 1e9))
    print("    salida:", gen[:260].replace("\n", " "))
    try:
        parsed = json.loads(gen[gen.find("{"):gen.rfind("}") + 1])
        print("    decision valida:", parsed.get("decision"), "| action_index:", parsed.get("action_index"))
    except Exception as exc:
        print("    JSON invalido:", type(exc).__name__)

print("LISTO_3050")
