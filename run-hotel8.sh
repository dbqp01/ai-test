cd /root/site2tools-mvp
# No relanza el servidor: evalua contra el adapter que este vivo en :8082
# (la cola run-finish.sh lo levanta con student v4).
.venv/bin/python - <<'PY'
import json, time, urllib.request

GOALS = [
    ("precio-doble-superior", "Cual es el precio por noche de la habitacion Doble Superior"),
    ("desayuno-hora", "A que hora se sirve el desayuno"),
    ("checkin-hora", "A que hora es el check-in"),
    ("direccion", "Cual es la direccion del hotel"),
    ("telefono", "Cual es el telefono de contacto"),
    ("wifi-clave", "Cual es la clave del wifi"),
    ("piscina", "Tiene piscina el hotel"),
    ("room-types", "Que tipos de habitacion hay"),
]

ok = answered = 0
rows = []
for label, goal in GOALS:
    body = json.dumps({
        "start_url": "https://usgarhoteles.com/",
        "goal": goal, "max_steps": 8, "allow_writes": False, "confirm": False, "context": {},
    }).encode()
    req = urllib.request.Request("http://localhost:8082/operate", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
    except Exception as exc:
        print(f"##### {label}: FALLO HTTP {type(exc).__name__} {exc}", flush=True)
        rows.append({"label": label, "goal": goal, "status": "http_error"})
        continue
    st = d.get("status")
    ans = d.get("answer") or {}
    if st in {"completed", "answered", "finished_unverified"}:
        ok += 1
    if ans.get("value"):
        answered += 1
    print(f"##### {label}: {st} steps={len(d.get('steps', []))} {time.time()-t0:.0f}s")
    if ans:
        print("  RESPUESTA:", ans.get("kind"), "=", str(ans.get("value"))[:190])
    ev = (d.get("evidence") or "")
    if ev:
        print("  evidencia:", ev[:220].replace("\n", " "), flush=True)
    rows.append({"label": label, "goal": goal, "status": st, "answer": ans,
                 "steps": len(d.get("steps", [])), "seconds": round(time.time()-t0, 1)})

print(f"=== TASA: {ok}/{len(GOALS)} concluyen, {answered}/{len(GOALS)} con respuesta literal ===")
with open("logs/hotel8-results.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
PY
