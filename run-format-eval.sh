cd /root/site2tools-mvp
.venv/bin/python - <<PY
import json
def is_json_format(row):
    u = [m for m in row["messages"] if m["role"] == "user"][0]["content"]
    try:
        return isinstance((json.loads(u).get("observation") or {}).get("actions"), list)
    except Exception:
        return False

rows = [json.loads(l) for l in open("data/v3.valid.jsonl")]
js = [r for r in rows if is_json_format(r)]
pl = [r for r in rows if not is_json_format(r)]
for name, subset in (("data/v3.valid-jsonformat.jsonl", js), ("data/v3.valid-flatformat.jsonl", pl)):
    with open(name, "w", encoding="utf-8") as f:
        for r in subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(subset))
PY

while ! grep -q EVAL_DONE logs/eval-v3.log; do sleep 20; done
echo "### formato-json-v1" >> logs/eval-v3.log
.venv/bin/python eval_action_acc.py --valid data/v3.valid-jsonformat.jsonl --limit 400 --adapter artifacts/site2tools-student-17b >> logs/eval-v3.log 2>&1
echo "### formato-plano-v1" >> logs/eval-v3.log
.venv/bin/python eval_action_acc.py --valid data/v3.valid-flatformat.jsonl --limit 400 --adapter artifacts/site2tools-student-17b >> logs/eval-v3.log 2>&1
echo "### FORMATO_DONE" >> logs/eval-v3.log
