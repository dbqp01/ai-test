"""Diagnóstico de por qué "que restaurante tiene el hotel" no se responde.

Imprime, por cada página observada en las últimas corridas locales: su ruta, si es contenido
propio del hotel y qué términos de restauración contiene. Al final lanza pick_answer sobre el
corpus propio acumulado, que es exactamente lo que hace el operador al agotar el presupuesto.
"""
import json
from urllib.parse import urlparse

import site2tools.core as c

rs = [json.loads(line) for line in open("data-local/traces.jsonl", encoding="utf-8")]
obs = [r for r in rs if r.get("type") == "observation"]
print("observaciones totales:", len(obs))

WORDS = ("cafeteria", "coffee", "cafe", "restaurant", "dining", "station", "bar ")
for o in obs[-16:]:
    url = o.get("url") or ""
    text = (o.get("text") or "").lower()
    path = urlparse(url).path or "/"
    found = [w for w in WORDS if w in text]
    print("  page", path[:30], "propia", c.is_hotel_content(url), "terms", found[:4])

goal = "Que restaurante tiene el hotel"
own = " ".join((o.get("text") or "") for o in obs if c.is_hotel_content(o.get("url") or ""))
print("longitud corpus propio:", len(own))
res = c.pick_answer(goal, own)
print("pick_answer:", res if res else "NINGUNA")
gt = c.tokens(c.expand_goal(goal)) - c.GENERIC_GOAL_WORDS
print("terminos del objetivo:", sorted(gt))
print("presentes en el corpus:", sorted(t for t in gt if t in own.lower()))
