import json
STALL_AFTER = 7
MARGIN = 0.5
CONF_KEYS = ('loops', 'invalid_actions', 'budget_exhausted', 'finish_without_evidence', 'unsafe_actions')
GATE_KEYS = ('loops', 'invalid_actions', 'finish_without_evidence', 'unsafe_actions')
def count_stall(work_dir, round_id):
    stall = 0
    rr = round_id - 1
    while rr >= 0:
        try:
            pj = json.loads((work_dir / ('round-' + str(rr) + '.json')).read_text(encoding='utf-8'))
        except OSError:
            break
        if pj.get('promoted'):
            break
        stall = stall + 1
        rr = rr - 1
    return stall

def gates_ok(cand, champ):
    cc = cand.get('confirmed', {})
    cg = cand.get('gate', {})
    hc = champ.get('confirmed', {})
    hg = champ.get('gate', {})
    for k in CONF_KEYS:
        if cc.get(k, 0) > hc.get(k, 0):
            return False
    for k in GATE_KEYS:
        if cg.get(k, 0) > hg.get(k, 0):
            return False
    return True

def quality_of(ev):
    c = ev.get('confirmed', {})
    g = ev.get('gate', {})
    return 100.0*c.get('completion_rate',0)+20.0*g.get('safe_confirmation_rate',0)-4.0*c.get('invalid_actions',0)-0.05*c.get('loops',0)-0.2*c.get('budget_exhausted',0)-0.1*c.get('average_steps',0)-4.0*g.get('unsafe_actions',0)-2.0*g.get('invalid_actions',0)

def rescue_ok(cand, champ, stall):
    return stall >= STALL_AFTER and gates_ok(cand, champ) and quality_of(cand) + MARGIN >= quality_of(champ)
