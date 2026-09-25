cd /root/site2tools-mvp
.venv/bin/python - <<'PY'
import json, urllib.request

GOALS = [
    ("precio", "Cual es el precio por noche de la habitacion Doble Superior"),
    ("desayuno", "A que hora se sirve el desayuno"),
    ("wifi", "Cual es la clave del wifi"),
]

for label, goal in GOALS:
    body = json.dumps({
        "start_url": "https://usgarhoteles.com/",
        "goal": goal, "max_steps": 8, "allow_writes": False, "confirm": False, "context": {},
    }).encode()
    req = urllib.request.Request("http://localhost:8082/operate", data=body,
                                 headers={"content-type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
    except Exception as exc:
        print(f"##### {label}: FALLO HTTP {type(exc).__name__} {exc}")
        continue
    print(f"##### {label}: {d.get('status')} reason={d.get('reason') or ''} steps={len(d.get('steps', []))}")
    for s in d.get("steps", []):
        a = s.get("action", {})
        p = s.get("policy") or {}
        r = s.get("result", {})
        print(f"  {s['index']} [{p.get('decision')}/{p.get('action_index')}] "
              f"{a.get('kind')} {repr((a.get('label') or '')[:38])} -> {r.get('ok')} {r.get('reason','')}")
    ev = (d.get("evidence") or "")
    if ev:
        print("  evidencia:", ev[:400].replace("\n", " "))
PY
