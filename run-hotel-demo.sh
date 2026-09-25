#!/bin/bash
cd /root/site2tools-mvp
run() {
  local label=$1 goal=$2
  .venv/bin/python - "$goal" <<'PY' > /tmp/req.json
import json, sys
print(json.dumps({
    "start_url": "https://usgarhoteles.com/",
    "goal": sys.argv[1],
    "max_steps": 6,
    "allow_writes": False,
    "confirm": False,
    "context": {},
}, ensure_ascii=False))
PY
  echo "##### $label"
  curl -s -m 600 -X POST localhost:8082/operate -H 'content-type: application/json' --data @/tmp/req.json \
    | .venv/bin/python -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception as e: print('PARSE FAIL', e); raise SystemExit
print('status:', d.get('status'), '| reason:', d.get('reason') or (d.get('error') or {}).get('reason'), '| steps:', len(d.get('steps',[])))
for s in d.get('steps',[]):
    a=s.get('action',{}); r=s.get('result',{})
    print('  -', a.get('kind'), repr((a.get('label') or '')[:40]), '->', r.get('ok'), r.get('reason',''), (r.get('url') or '')[:80])
ev=(d.get('evidence') or '')
print('evidence:', ev[:700])
"
}

run precio "Cual es el precio por noche de la habitacion Doble Superior"
run desayuno "A que hora se sirve el desayuno"
