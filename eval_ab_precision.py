"""Corre la misma medida sobre el `site2tools` del directorio actual.

Se usa desde dos arboles distintos (worktree viejo y codigo actual) para que el antes/despues se
compare en la MISMA cubeta: objetivos de ACCION, donde afirmar es falso por definicion.
"""
import json, random, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from site2tools.core import pick_answer

DATOS = Path(sys.argv[1])
DE_DATO = re.compile(r"\b(price|cost|how much|cuanto|hour|time|hours|open|close|closes|phone|number|"
                     r"email|e-mail|address|where|donde|rating|score|fee)\b", re.I)
filas = [json.loads(l) for l in DATOS.open(encoding="utf-8")]
random.Random(20260925).shuffle(filas)
accion = af_acc = dato = af_dato = 0
for f in filas[:2000]:
    try:
        o = json.loads(f["messages"][1]["content"])
    except Exception:
        continue
    g = str(o.get("goal") or ""); t = str((o.get("observation") or {}).get("text") or "")
    if len(g) < 12 or len(t) < 200:
        continue
    es_dato = bool(DE_DATO.search(g))
    if es_dato:
        dato += 1
    else:
        accion += 1
    try:
        a = pick_answer(g, t)
    except Exception:
        continue
    if a and a.get("value"):
        if es_dato:
            af_dato += 1
        else:
            af_acc += 1
import site2tools.core as c
print("reglas presentes:", "is_navigation_text" in dir(c), "| ACCION %d/%d = %.1f%% | DATO %d/%d = %.1f%%"
      % (af_acc, accion, 100.0 * af_acc / max(1, accion), af_dato, dato, 100.0 * af_dato / max(1, dato)))
