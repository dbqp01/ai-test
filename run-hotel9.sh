cd /root/site2tools-mvp || exit 1
PID=$(ss -ltnp 2>/dev/null | grep ":8082" | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$PID" ] && { kill "$PID"; sleep 4; }
ADAPTER="${1:-artifacts/site2tools-student-17b-v4}"
setsid nohup .venv/bin/python -m site2tools.server --host 127.0.0.1 --port 8082 --data data \
  --policy-adapter "$ADAPTER" --policy-model Qwen/Qwen3-1.7B --policy-device cuda \
  > logs/demo-server-last.log 2>&1 < /dev/null &
echo "esperando servidor con $ADAPTER"
for i in $(seq 1 40); do
  sleep 6
  curl -s -m 5 localhost:8082/health | grep -q ok && { echo SERVIDOR_OK; break; }
done
curl -s -m 5 localhost:8082/health | head -3
bash run-hotel8.sh
