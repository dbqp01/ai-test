"""Precision de la capa de respuesta fuera del dominio para el que se ajusto.

`m2w2.valid.jsonl` trae paginas reales de otros dominios (booking, vuelos, compras...) con su
objetivo. Los objetivos son de accion, no de dato, asi que aqui no se mide "acerto" sino lo unico
que importa para la politica del proyecto: **cuantas veces la heuristica afirma algo cuando no
deberia**. Se importa solo el extractor (sin navegador ni modelo), y el muestreo es determinista.
"""
import json
import random
import re
import sys
from pathlib import Path

from site2tools.core import clean, pick_answer

FUERA = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9.-]+)", re.I)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
# Segundo argumento opcional = tope de sobreafirmacion admitido. Con eso este script pasa de ser una
# medida puntual a ser un guard: el gate lo llama y cualquier cambio que vuelva a dejar entrar cromo
# de navegacion como evidencia se pilla en segundos, sin esperar a otra auditoria manual.
MAXIMO = float(sys.argv[2]) if len(sys.argv) > 2 else None

rng = random.Random(20260925)

POSIBLES = [Path("data/m2w2.valid.jsonl"),
            Path("../site2tools-backup/snap0042/data/m2w2.valid.jsonl")]
ruta = next((p for p in POSIBLES if p.exists()), None)
if ruta is None:
    raise SystemExit("no esta el corpus: " + " | ".join(str(p) for p in POSIBLES))
filas = [json.loads(l) for l in ruta.open(encoding="utf-8")]
rng.shuffle(filas)

sitios = {}
respondidas = 0
ejemplos = []
for fila in filas[:N]:
    try:
        obs = json.loads(fila["messages"][1]["content"])
    except (KeyError, IndexError, ValueError):
        continue
    goal = str(obs.get("goal") or "")
    texto = str((obs.get("observation") or {}).get("text") or "")
    url = str((obs.get("observation") or {}).get("url") or "")
    if len(goal) < 12 or len(texto) < 200:
        continue
    sitios[FUERA.match(url).group(1) if FUERA.match(url) else url[:24]] = \
        sitios.get(FUERA.match(url).group(1) if FUERA.match(url) else url[:24], 0) + 1
    try:
        ans = pick_answer(goal, texto)
    except Exception as exc:  # que un texto raro reviente el extractor tambien es un resultado
        ejemplos.append(("EXCEPCION", type(exc).__name__, goal[:60], "", url[:30]))
        continue
    if ans and ans.get("value"):
        respondidas += 1
        if len(ejemplos) < 24:
            valor = clean(ans["value"])[:110]
            verbatim = re.sub(r"\s+", " ", valor).lower() in re.sub(r"\s+", " ", texto).lower()
            ejemplos.append((ans.get("kind"), "literal" if verbatim else "NO-LITERAL",
                             goal[:58], valor, url[:26]))

print(f"filas leidas: {min(N, len(filas))} | sitios distintos: {len(sitios)}")
print("top sitios:", ", ".join(f"{s}={c}" for s, c in sorted(sitios.items(), key=lambda kv: -kv[1])[:6]))
print(f"la heuristica_afirmo_algo en {respondidas} de {N} objetivos de ACCION "
      f"({100.0 * respondidas / max(1, N):.1f}%)")
print("--- muestras para clasificar a mano ---")
for kind, lit, goal, valor, url in ejemplos[:18]:
    print(f"  [{kind}|{lit}] {url}\n     goal: {goal}\n     resp: {valor}")

porcentaje = 100.0 * respondidas / max(1, N)
if MAXIMO is not None:
    if porcentaje > MAXIMO:
        print(f"FALLO: sobreafirmacion {porcentaje:.1f}% por encima del tope {MAXIMO:.1f}%")
        raise SystemExit(1)
    print(f"OK: sobreafirmacion {porcentaje:.1f}% <= tope {MAXIMO:.1f}%")

