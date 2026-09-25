cd /root/site2tools-mvp
echo "esperando a que termine v4..."
while [ "$(ps aux | grep -c '[t]rain_lora')" -gt 0 ]; do sleep 20; done
echo "=== v4 terminado $(date -u) ==="
ls -la artifacts/site2tools-student-17b-v4/adapter_model.safetensors 2>/dev/null || { echo "SIN ADAPTER v4"; exit 1; }

ev() {
  echo "### $1"
  if [ "$3" = "none" ]; then
    .venv/bin/python eval_action_acc.py --valid "$2" --limit 400
  else
    .venv/bin/python eval_action_acc.py --valid "$2" --limit 400 --adapter "$3"
  fi
}
ev "v4-realista" data/m2w2.valid.jsonl artifacts/site2tools-student-17b-v4
ev "v3e3-realista" data/m2w2.valid.jsonl artifacts/site2tools-student-17b-v3-e3
ev "v4-juguetes" data/v3.valid.jsonl artifacts/site2tools-student-17b-v4
echo "### FIN_EVALS"

pid=$(ss -ltnp 2>/dev/null | grep ":8082" | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$pid" ] && { kill $pid; sleep 3; }
setsid nohup .venv/bin/python -m site2tools.server --host 127.0.0.1 --port 8082 --data data \
  --policy-adapter artifacts/site2tools-student-17b-v4 --policy-model Qwen/Qwen3-1.7B --policy-device cuda \
  > logs/demo-17b-v4.log 2>&1 < /dev/null &
for i in $(seq 1 30); do sleep 6; curl -s -m 5 localhost:8082/health | grep -q ok && { echo SERVIDOR_V4_OK; break; }; done
bash run-tasks.sh > logs/hotel-v4.log 2>&1
echo "### TODO_LISTO $(date -u)"
