cd /root/site2tools-mvp
run() {
  echo "### $1 valid=$2 adapter=$3"
  if [ "$3" = "none" ]; then
    .venv/bin/python eval_action_acc.py --valid "$2" --limit 700
  else
    .venv/bin/python eval_action_acc.py --valid "$2" --limit 700 --adapter "$3"
  fi
}
run "clean-v1" data/v3.valid.jsonl artifacts/site2tools-student-17b
run "clean-v3" data/v3.valid.jsonl artifacts/site2tools-student-17b-v3
run "transfer-v1" data/v3.transfer.jsonl artifacts/site2tools-student-17b
run "transfer-v3" data/v3.transfer.jsonl artifacts/site2tools-student-17b-v3
run "transfer-base" data/v3.transfer.jsonl none
echo "### EVAL_DONE"
