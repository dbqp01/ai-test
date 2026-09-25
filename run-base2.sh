cd /root/site2tools-mvp || exit 1
LOG=logs/eval-base2.log
: > "$LOG"
# Smoke primero: el control sin adapter se caía al resolver args.adapter="none" contra
# PeftModel y contra el tokenizer. 20 filas prueban que el camino funciona antes de gastar
# los 640+382.
.venv/bin/python eval_action_acc.py --valid data/v3.valid.jsonl --limit 20 --adapter none >> "$LOG" 2>&1
if grep -q "ACC {" "$LOG"; then
  echo "SMOKE_OK" >> "$LOG"
else
  echo "SMOKE_FAIL" >> "$LOG"
  exit 1
fi
ev() {
  echo "### $1 valid=$2 adapter=$3" >> "$LOG"
  .venv/bin/python eval_action_acc.py --valid "$2" --limit "$4" --adapter "$3" >> "$LOG" 2>&1
}
ev base-clean    data/v3.valid.jsonl              none 700
ev base-plano    data/v3.valid-flatformat.jsonl   none 400
ev v4-transfer   data/v3.transfer.jsonl           artifacts/site2tools-student-17b-v4 200
ev base-juguetes data/holdout-jsonformat.jsonl    none 300
echo EVAL_DONE_BASE2 >> "$LOG"
