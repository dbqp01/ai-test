import json, glob, random
from generate_dataset import SYSTEM_PROMPT

def short(c):
    try:
        attrs = json.loads(c.get('attributes', '{}'))
    except Exception:
        attrs = {}
    label = attrs.get('aria_label') or attrs.get('id') or attrs.get('placeholder') or c.get('tag', '?')
    s = '[' + str(c.get('tag', '?')) + '] ' + str(label)
    if attrs.get('role'):
        s = s + ' (' + str(attrs['role']) + ')'
    if attrs.get('value'):
        s = s + ' = ' + str(attrs['value'])[:40]
    return s[:120]

def main():
    rng = random.Random(20260923)
    files = sorted(glob.glob('/root/huggingface/mind2web-full/data/train/*.json'))
    ot = open('data/mind2web.train.jsonl', 'w', encoding='utf-8')
    ov = open('data/mind2web.valid.jsonl', 'w', encoding='utf-8')
    ntask = nstep = 0
    sites = {}
    for fp in files:
        print('FILE', fp)
        tasks = json.load(open(fp, encoding='utf-8'))
        for t in tasks:
            web = t.get('website', '?')
            sites[web] = sites.get(web, 0) + 1
            dest = ov if (ntask % 20 == 0) else ot
            hist = []
            acts = t.get('actions', [])
            for i, a in enumerate(acts):
                op = (a.get('operation') or {}).get('op', 'CLICK')
                val = (a.get('operation') or {}).get('value', '')
                cands = [short(c) for c in a.get('pos_candidates', [])[:3]]
                negs = [short(c) for c in a.get('neg_candidates', [])]
                rng.shuffle(negs)
                cands = cands + negs[:7]
                order = list(range(len(cands)))
                rng.shuffle(order)
                shown = [cands[j] for j in order]
                correct = order.index(0)
                obs = 'Goal: ' + t.get('confirmed_task', '')
                obs = obs + chr(10) + 'Site: ' + web + ' step ' + str(i+1) + '/' + str(len(acts)) + chr(10)
                if hist:
                    obs = obs + 'Done: ' + ' | '.join(hist[-3:]) + chr(10)
                obs = obs + 'Actions:' + chr(10) + chr(10).join(str(k) + '. ' + s for k, s in enumerate(shown))
                if op == 'TYPE' and val:
                    obs = obs + chr(10) + 'Type text: ' + val[:80]
                msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': obs}]
                resp = {'decision': 'act', 'action_index': correct}
                if val:
                    resp['arguments'] = {'value': val[:80]}
                msgs.append({'role': 'assistant', 'content': json.dumps(resp)})
                dest.write(json.dumps({'messages': msgs}, ensure_ascii=False) + chr(10))
                nstep = nstep + 1
                hist.append(op)
            ntask = ntask + 1
    ot.close()
    ov.close()
    print(json.dumps({'tasks': ntask, 'steps': nstep, 'sites': len(sites)}))

if __name__ == '__main__':
    main()
