import json, glob, hashlib, os, re, collections

def norm(row):
    msgs = row.get("messages") or []
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    # quitar contador de paso / numeros de step para cazar casi-duplicados
    u2 = re.sub(r"\s*step\s+\d+\s*/\s*\d+", " ", u)
    u2 = re.sub(r"\s+", " ", u2).strip()
    return hashlib.sha1((u2 + "\x00" + a.strip()).encode()).hexdigest(), hashlib.sha1(u2.encode()).hexdigest()

def load(p):
    out = []
    try:
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if l:
                out.append(json.loads(l))
    except Exception as e:
        print("  (ilegible %s: %s)" % (p, e))
    return out

train_pool = []
for pat in ["data/train.jsonl", "data/*.train.jsonl", "data/accepted.jsonl", "data/pool*.jsonl",
            "data/operations.jsonl", "data/synth*.jsonl", "data/m2w2.train.jsonl", "data/mind2web.train.jsonl"]:
    for p in sorted(glob.glob(pat)):
        rs = load(p)
        if not rs:
            continue
        train_pool.append((p, rs))

exact_sets, user_sets = set(), set()
for p, rs in train_pool:
    for r in rs:
        e, u = norm(r)
        exact_sets.add(e)
        user_sets.add(u)
    print("%-34s filas=%-6d" % (p, len(rs)))
print("POOL total uniques exact=%d user=%d\n" % (len(exact_sets), len(user_sets)))

for vp in ["data/v3.valid.jsonl", "data/v3.valid-jsonformat.jsonl", "data/v3.valid-flatformat.jsonl", "data/v3.transfer.jsonl"]:
    rs = load(vp)
    if not rs:
        continue
    ne = nu = 0
    for r in rs:
        e, u = norm(r)
        if e in exact_sets:
            ne += 1
        if u in user_sets:
            nu += 1
    print("%-34s n=%-5d FUGA exacta=%5.1f%%  mismo estado(user)=%5.1f%%" % (vp, len(rs), 100.0 * ne / len(rs), 100.0 * nu / len(rs)))

# que formato domina las filas filtradas por fuga
js = load("data/v3.valid-jsonformat.jsonl")
pl = load("data/v3.valid-flatformat.jsonl")
for name, rs in (("jsonformat", js), ("flatformat", pl)):
    kept = []
    for r in rs:
        e, u = norm(r)
        if e not in exact_sets and u not in user_sets:
            kept.append(r)
    with open("data/holdout-%s.jsonl" % name, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("HOLDOUT limpio %s: %d/%d conservadas -> data/holdout-%s.jsonl" % (name, len(kept), len(rs), name))
