cd /root/site2tools-mvp
.venv/bin/python -c "import ast; ast.parse(open('convert_mind2web_v2.py').read())" || exit 1
echo "=== conversor ==="
.venv/bin/python convert_mind2web_v2.py
wc -l data/m2w2.train.jsonl data/m2w2.valid.jsonl
echo "=== v1 sobre dataset realista ==="
.venv/bin/python eval_action_acc.py --valid data/m2w2.valid.jsonl --limit 400 --adapter artifacts/site2tools-student-17b
echo "=== base sobre dataset realista ==="
.venv/bin/python eval_action_acc.py --valid data/m2w2.valid.jsonl --limit 400
echo "=== M2W_DONE ==="
