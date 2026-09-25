"""Benchmark de 16 objetivos END-TO-END en la RTX 3050 local (sin droplet, sin crédito).

Se ejecuta en proceso (UniversalOperator directamente), no por HTTP: así se mide el stack
completo — routing por grafo, menú recortado a 10, puente es->en, FAQ pregunta-a-pregunta y
patrones — sobre el hardware objetivo. Solo lectura, writes OFF, 8 pasos como máximo.
"""
import json
import sys
import time
from pathlib import Path

from site2tools.core import UniversalOperator
from site2tools.policy import ModelDecisionPolicy

SITE = "https://usgarhoteles.com/"
GOALS = [
    ("precio-doble-superior", "Cual es el precio por noche de la habitacion Doble Superior"),
    ("desayuno-hora", "A que hora se sirve el desayuno"),
    ("checkin-hora", "A que hora es el check-in"),
    ("checkout-hora", "A que hora es el check-out"),
    ("direccion", "Cual es la direccion del hotel"),
    ("telefono", "Cual es el telefono de contacto"),
    ("correo", "Cual es el correo electronico"),
    ("room-types", "Que tipos de habitacion hay"),
    ("wifi-clave", "Cual es la clave del wifi"),
    ("piscina", "Tiene piscina el hotel"),
    ("gastronomia", "Que restaurante tiene el hotel"),
    ("ubicacion-barrio", "En que barrio esta el hotel"),
    ("idiomas", "El personal habla otros idiomas"),
    ("aeropuerto", "Como llego desde el aeropuerto"),
    ("equipaje", "Puedo dejar el equipaje despues del check-out"),
    ("oxigeno", "Hay oxigeno para la altura"),
]

print("cargando policy v4 en cuda...")
# La consola de Windows es cp1252 y el texto del sitio trae glifos decorativos (U+2726): un
# print con el valor crudo tumbó el benchmark en el objetivo 9 y costó la medición completa.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
t0 = time.time()
policy = ModelDecisionPolicy("adapter-v4", base_model="Qwen/Qwen3-1.7B", device="cuda")
print("policy lista en %.0fs" % (time.time() - t0), flush=True)

Path("data-local").mkdir(exist_ok=True)
ok = answered = con_procedencia = 0
rows = []
OUT = Path("bench16-local.json")
for label, goal in GOALS:
    op = UniversalOperator(SITE, Path("data-local"), max_steps=8,
                           allowed_hosts=["usgarhoteles.com"], policy=policy)
    t0 = time.time()
    try:
        res = op.run(goal, context={}, confirm=False, allow_writes=False)
    except Exception as exc:
        print("##### %s: FALLO %s: %s" % (label, type(exc).__name__, str(exc)[:150]), flush=True)
        rows.append({"label": label, "goal": goal, "status": "exception"})
        OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        continue
    st = res.get("status")
    ans = res.get("answer") or {}
    if st in {"completed", "answered", "finished_unverified", "not_found_in_survey"}:
        ok += 1
    if ans.get("value"):
        answered += 1
        if ans.get("source_url"):
            con_procedencia += 1
    print("##### %s: %s pasos=%d %.0fs" % (label, st, len(res.get("steps") or []), time.time() - t0),
          flush=True)
    if ans:
        print("   R:", ans.get("kind"), "=", str(ans.get("value"))[:160], flush=True)
    rows.append({"label": label, "goal": goal, "status": st, "answer": ans,
                 "survey_pages": res.get("survey_pages"), "survey_chars": res.get("survey_chars"),
                 "steps": len(res.get("steps") or []), "seconds": round(time.time() - t0, 1)})
    # Se guarda tras cada objetivo, no al final: un fallo en el print no debe costar la corrida.
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

print("=== BENCH16 LOCAL: %d/%d concluyen, %d con respuesta, %d con procedencia verificada ==="
      % (ok, len(GOALS), answered, con_procedencia), flush=True)
# Los dos grados no son intercambiables: "literal" es evidencia leida del innerText de una pagina,
# "estructurada" es un campo publicado (JSON-LD / FAQ / select). Sumarlos en una sola cifra infla la
# afirmacion de "cero invenciones", que es justo lo que este numero tiene que impedir.
lit = sum(1 for r in rows if (r.get("answer") or {}).get("verbatim"))
est = sum(1 for r in rows if (r.get("answer") or {}).get("value")
          and not r["answer"].get("verbatim") and r["answer"].get("source_field"))
print("   procedencia: %d literal en pagina + %d campo estructurado (jsonld/faq/select)" % (lit, est),
      flush=True)
sin_proc = [r["label"] for r in rows
            if (r.get("answer") or {}).get("value") and not r["answer"].get("source_url")]
if sin_proc:
    print("RESPUESTAS SIN PROCEDENCIA (revisar antes de afirmar que no hay invencion):", sin_proc,
          flush=True)
for r in [x for x in rows if x["status"] == "not_found_in_survey"]:
    print("   ausencia %s afirmada sobre %d paginas / %d caracteres leidos"
          % (r["label"], r.get("survey_pages") or 0, r.get("survey_chars") or 0), flush=True)
