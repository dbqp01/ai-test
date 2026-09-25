"""Filas de entrenamiento por hindsight a partir de las trazas del propio operador.

Mentira que Mind2Web no puede dar: su cleaned_html es el estado ANTES de la acción de cada paso,
así que no existe el estado "tarea ya terminada" (ver RESUMEN). Aquí en cambio se usan
transiciones realmente ejecutadas y verificadas:

  - el primer estado de la trayectoria donde la respuesta YA está en pantalla  -> oro `finish`
  - los estados previos de esa misma trayectoria, con la acción que el operador ejecutó y que
    llevó al siguiente estado (`result.ok=true`)                                -> oro `act`

Cero invención: ninguna fila se etiqueta sobre un estado que no se visitó. El payload se
construye con `ModelDecisionPolicy._observation_payload`, el serializador real del servidor, para
que el formato del dataset sea idéntico al que se sirve (fue una de las causas del 0/2).
"""
import json

from site2tools.core import Action, Observation, pick_answer
from site2tools.policy import ModelDecisionPolicy, SYSTEM_PROMPT


def load_observations(path: str = "data/traces.jsonl") -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "observation" and r.get("state_id"):
            out[r["state_id"]] = r
    return out


def to_observation(record: dict) -> Observation:
    fields = {k: v for k, v in record.items() if k not in ("type", "actions", "state_id", "network")}
    actions = []
    for a in record.get("actions") or []:
        try:
            actions.append(Action(**a))
        except TypeError:
            return None
    try:
        return Observation(state_id=record["state_id"], network=[], actions=actions, **fields)
    except TypeError:
        return None


def payload_for(goal: str, obs: Observation) -> dict:
    from site2tools.policy import ModelDecisionPolicy

    return {
        "goal": goal,
        "context": {},
        "observation": ModelDecisionPolicy._observation_payload(obs, False),
        "history": [],
    }


def row(goal: str, obs: Observation, gold: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload_for(goal, obs), ensure_ascii=False, separators=(",", ":"))},
        {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))},
    ], "meta": {"goal": goal, "decision": gold["decision"], "site": "usgarhoteles.com"}}


def action_index(obs: Observation, action_id: str):
    for i, a in enumerate(obs.actions):
        if a.action_id == action_id:
            return i
    return None


def main() -> None:
    obs_by_id = load_observations()
    print("observaciones indexadas:", len(obs_by_id))
    rows = []
    stats = {"trayectorias": 0, "con_respuesta": 0, "act": 0, "finish": 0}
    for line in open("data/operations.jsonl", encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        steps = [s for s in (rec.get("steps") or []) if s.get("goal") and s.get("observation") in obs_by_id]
        if not steps:
            continue
        stats["trayectorias"] += 1
        goal = steps[0]["goal"]
        terminal = None
        for i, s in enumerate(steps):
            o = to_observation(obs_by_id[s["observation"]])
            if o is None:
                continue
            ans = pick_answer(goal, o.text)
            if ans and ans.get("value"):
                terminal = (i, o, ans)
                break
        if terminal is None:
            continue
        stats["con_respuesta"] += 1
        i, term_obs, ans = terminal
        rows.append(row(goal, term_obs, {
            "decision": "finish",
            "reason": "La respuesta ya es observable en la pagina: " + str(ans.get("value"))[:120],
        }))
        stats["finish"] += 1
        for prev in steps[:i]:
            po = to_observation(obs_by_id[prev["observation"]])
            if po is None or not (prev.get("result") or {}).get("ok"):
                continue
            idx = action_index(po, (prev.get("action") or {}).get("action_id"))
            if idx is None:
                continue
            rows.append(row(goal, po, {
                "decision": "act",
                "reason": "Esta accion acerca la informacion que falta para el objetivo.",
                "action_index": idx,
            }))
            stats["act"] += 1
    with open("data/v5.hindsight.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("stats:", stats, "-> filas escritas:", len(rows), "en data/v5.hindsight.jsonl")


if __name__ == "__main__":
    main()
