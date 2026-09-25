"""Operador COMPLETO en la RTX 3050 local, contra el sitio del usuario (solo lectura).

Prueba lo que el test de tensors no prueba: que el paquete `site2tools` (routing por grafo,
menú recortado a 10, puente es->en, respuestas estructuradas y patrones) corre en el hardware
objetivo y reproduce el resultado medido en la droplet. No escribe nada: writes OFF, max 8 pasos.
"""
import json
import time
from pathlib import Path

from site2tools.core import UniversalOperator
from site2tools.policy import ModelDecisionPolicy

SITE = "https://usgarhoteles.com/"
GOALS = [
    ("precio-doble-superior", "Cual es el precio por noche de la habitacion Doble Superior"),
    ("desayuno-hora", "A que hora se sirve el desayuno"),
    ("direccion", "Cual es la direccion del hotel"),
    ("piscina", "Tiene piscina el hotel"),
]

print("cargando policy v4 en cuda...")
t0 = time.time()
policy = ModelDecisionPolicy("adapter-v4", base_model="Qwen/Qwen3-1.7B", device="cuda")
print("policy lista en %.0fs" % (time.time() - t0))

Path("data-local").mkdir(exist_ok=True)
ok = answered = 0
rows = []
for label, goal in GOALS:
    op = UniversalOperator(SITE, Path("data-local"), max_steps=8,
                           allowed_hosts=["usgarhoteles.com"], policy=policy)
    t0 = time.time()
    try:
        res = op.run(goal, context={}, confirm=False, allow_writes=False)
    except Exception as exc:
        print("##### %s: FALLO %s: %s" % (label, type(exc).__name__, str(exc)[:160]), flush=True)
        rows.append({"label": label, "status": "exception"})
        continue
    st = res.get("status")
    ans = res.get("answer") or {}
    if st in {"completed", "answered", "finished_unverified", "not_found_in_survey"}:
        ok += 1
    if ans.get("value"):
        answered += 1
    print("##### %s: %s pasos=%d %.0fs" % (label, st, len(res.get("steps") or []), time.time() - t0))
    if ans:
        print("   RESPUESTA:", ans.get("kind"), "=", str(ans.get("value"))[:170], flush=True)
    rows.append({"label": label, "goal": goal, "status": st, "answer": ans,
                 "steps": len(res.get("steps") or []), "seconds": round(time.time() - t0, 1)})

print("=== TASA LOCAL: %d/%d concluyen, %d con respuesta ===" % (ok, len(GOALS), answered))
Path("logs-local.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
