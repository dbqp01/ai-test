cd /root/site2tools-mvp || exit 1
# n=8 seguía siendo poca muestra para la tasa titular. Aquí van 16 objetivos informativos,
# elegidos leyendo antes qué publica el sitio (trazas + JSON-LD) para no contar como fallo
# algo que la página no dice.
PID=$(ss -ltnp 2>/dev/null | grep ":8082" | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$PID" ] && { kill "$PID"; sleep 4; }
setsid nohup .venv/bin/python -m site2tools.server --host 127.0.0.1 --port 8082 --data data \
  --policy-adapter artifacts/site2tools-student-17b-v4 --policy-model Qwen/Qwen3-1.7B --policy-device cuda \
  > logs/demo-bench16.log 2>&1 < /dev/null &
for i in $(seq 1 40); do sleep 6; curl -s -m 5 localhost:8082/health | grep -q ok && { echo SERVIDOR_OK; break; }; done

.venv/bin/python - <<'PY'
import json, time, urllib.request
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
ok = answered = 0
rows = []
for label, goal in GOALS:
    body = json.dumps({"start_url": "https://usgarhoteles.com/", "goal": goal, "max_steps": 8,
                       "allow_writes": False, "confirm": False, "context": {}}).encode()
    req = urllib.request.Request("http://localhost:8082/operate", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
    except Exception as exc:
        print("#####", label, "FALLO_HTTP", type(exc).__name__, flush=True)
        rows.append({"label": label, "status": "http_error"})
        continue
    st = d.get("status"); a = d.get("answer") or {}
    if st in {"completed", "answered", "finished_unverified", "not_found_in_survey"}:
        ok += 1
    if a.get("value"):
        answered += 1
    print("#####", label, st, "pasos=", len(d.get("steps", [])), "%.0fs" % (time.time() - t0))
    if a:
        print("   R:", a.get("kind"), "=", str(a.get("value"))[:150], flush=True)
    rows.append({"label": label, "goal": goal, "status": st, "answer": a,
                 "steps": len(d.get("steps", []))})
print("=== BENCH16: %d/%d concluyen, %d con respuesta ===" % (ok, len(GOALS), answered))
json.dump(rows, open("logs/bench16-results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY
