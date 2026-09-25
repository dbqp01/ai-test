cd /root/site2tools-mvp || exit 1
LOG=logs/eval-r2.log
: > "$LOG"
ev() {
  echo "### $1 valid=$2 adapter=$3" >> "$LOG"
  .venv/bin/python eval_action_acc.py --valid "$2" --limit 700 --adapter "$3" >> "$LOG" 2>&1
}
ev base-clean    data/v3.valid.jsonl           none
ev e3-clean       data/v3.valid.jsonl           artifacts/site2tools-student-17b-v3-e3
ev e3-transfer    data/v3.transfer.jsonl        artifacts/site2tools-student-17b-v3-e3
ev v1-plano       data/v3.valid-flatformat.jsonl artifacts/site2tools-student-17b
ev v3-plano       data/v3.valid-flatformat.jsonl artifacts/site2tools-student-17b-v3
ev base-plano     data/v3.valid-flatformat.jsonl none
echo EVAL_DONE_R2 >> "$LOG"
