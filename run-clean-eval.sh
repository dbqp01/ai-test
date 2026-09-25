#!/bin/bash
cd /root/site2tools-mvp
V=data/student-17b.valid-clean.jsonl
echo "BASE###"
.venv/bin/python eval_action_acc.py --valid "$V" --limit 649
echo "V1###"
.venv/bin/python eval_action_acc.py --valid "$V" --limit 649 --adapter artifacts/site2tools-student-17b
echo "ALLDONE###"
