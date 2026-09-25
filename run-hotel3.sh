cd /root/site2tools-mvp
pid=$(ss -ltnp 2>/dev/null | grep ":8082" | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$pid" ] && { kill $pid; sleep 3; }
setsid nohup .venv/bin/python -m site2tools.server --host 127.0.0.1 --port 8082 --data data \
  --policy-adapter artifacts/site2tools-student-17b --policy-model Qwen/Qwen3-1.7B --policy-device cuda \
  > logs/demo-17b-patched.log 2>&1 < /dev/null &
for i in $(seq 1 30); do sleep 6; curl -s -m 5 localhost:8082/health | grep -q ok && { echo SERVIDOR_OK; break; }; done

.venv/bin/python - <<'PY'
import json, time, urllib.request

# n=2 no informaba: se pasa a 8 objetivos informativos comprobables contra el sitio.
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
        d = json.load(urllib.request.urlopen(req, timeout=900))
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
        print("  RESPUESTA:", ans.get("kind"), "=", ans.get("value"), "|", (ans.get("context") or "")[:180])
    for s in d.get("steps", [])[-4:]:
        a = s.get("action", {})
        p = s.get("policy") or {}
        r = s.get("result", {})
        print(f"  {s['index']} [{p.get('decision')}/{p.get('action_index')}{'/'+p.get('_attempt','') if p.get('_attempt') else ''}] "
              f"{a.get('kind')} {repr((a.get('label') or '')[:34])} -> {r.get('ok')} {r.get('reason','')[:40]}")
    ev = (d.get("evidence") or "")
    if ev:
        print("  evidencia:", ev[:300].replace("\n", " "), flush=True)
    rows.append({"label": label, "goal": goal, "status": st, "answer": ans,
                 "steps": len(d.get("steps", [])), "seconds": round(time.time()-t0, 1)})

print(f"=== TASA: {ok}/{len(GOALS)} concluyen, {answered}/{len(GOALS)} con respuesta literal ===")
with open("logs/hotel3-results-v4.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
PY
