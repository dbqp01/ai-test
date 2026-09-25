from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any


_SIGNAL_WORDS = {
    "submitted", "enviada", "entregada", "confirmation", "comprobante",
    "success", "exito", "éxito", "ticket", "order", "download", "running",
}
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:12]


def _action_key(action: dict[str, Any]) -> str:
    return "|".join(
        str(action.get(name, ""))
        for name in ("kind", "label", "href", "role", "input_type", "name", "placeholder")
    ).lower()


def _action_name(action: dict[str, Any]) -> str:
    return _clean(
        action.get("label")
        or action.get("name")
        or action.get("placeholder")
        or action.get("href")
        or action.get("kind")
        or "site action"
    )


def _risk(action: dict[str, Any]) -> str:
    if bool(action.get("dangerous")) or str(action.get("input_type", "")).lower() in {"file", "password"}:
        return "high"
    if action.get("kind") == "input":
        return "medium"
    return "low"


def _merge_risk(current: str, candidate: str) -> str:
    return candidate if _RISK_ORDER[candidate] > _RISK_ORDER[current] else current


def _signals(text: str) -> list[str]:
    lowered = text.lower()
    return sorted(signal for signal in _SIGNAL_WORDS if signal in lowered)


def _dedupe(items: list[Any], limit: int | None = None) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return result


def _input_schema(action: dict[str, Any]) -> dict[str, Any] | None:
    if action.get("kind") != "input":
        return None
    field = _clean(action.get("name") or action.get("placeholder") or "value")
    input_type = str(action.get("input_type") or "text").lower()
    property_schema: dict[str, Any] = {"type": "string"}
    if input_type == "file":
        property_schema = {
            "type": "string",
            "format": "file-path",
            "description": "Ruta local del archivo que el usuario autoriza a subir.",
        }
    elif input_type == "password":
        property_schema["writeOnly"] = True
    return {
        "type": "object",
        "properties": {field: property_schema},
        "required": [field],
        "additionalProperties": False,
    }


class CapabilityManifestBuilder:
    """Turns a replayed site graph into one guarded, goal-routable capability manifest."""

    def __init__(self, graph: dict[str, Any]):
        self.graph = graph
        self.nodes: dict[str, dict[str, Any]] = graph.get("nodes", {}) or {}
        self.edges: list[dict[str, Any]] = graph.get("edges", []) or []

    def build(self) -> dict[str, Any]:
        capabilities: dict[str, dict[str, Any]] = {}
        for state_id, node in self.nodes.items():
            for action in node.get("actions", []) or []:
                self._observe(capabilities, state_id, node, action)

        for edge in self.edges:
            action = edge.get("action", {}) or {}
            if not action:
                continue
            key = _action_key(action)
            capability = capabilities.setdefault(
                key,
                self._new_capability(action),
            )
            source = self.nodes.get(str(edge.get("from", "")), {})
            target_id = edge.get("to")
            target = self.nodes.get(str(target_id), {}) if target_id else {}
            verification = edge.get("verification", {}) or {}
            status = str(verification.get("status") or ("verified" if target_id else "observed_guarded"))
            route = {
                "from_state": edge.get("from"),
                "to_state": target_id,
                "path": edge.get("path", []),
                "verification": status,
            }
            capability["routes"] = _dedupe(capability["routes"] + [route], limit=12)
            capability["verification"]["observed_transitions"] += 1
            if status == "verified" and target_id:
                capability["verification"]["verified_transitions"] += 1
                capability["verification"]["status"] = "verified"
                capability["effects"].append({
                    "type": "state_transition",
                    "to_state": target_id,
                    "url": target.get("url", ""),
                    "evidence_signals": _signals(target.get("text_preview", "")),
                })
                capability["evidence"].append({
                    "kind": "browser_replay",
                    "state_id": target_id,
                    "url": target.get("url", ""),
                    "snippet": _clean(target.get("text_preview", ""))[:500],
                })
            elif status in {"observed_guarded", "requires_confirmation"}:
                capability["verification"]["guarded_transitions"] += 1
                if capability["verification"]["status"] != "verified":
                    capability["verification"]["status"] = "guarded_requires_confirmation"
            else:
                if capability["verification"]["status"] == "unverified":
                    capability["verification"]["status"] = "observed_unverified"

            if source:
                capability["preconditions"].append({
                    "type": "page_state",
                    "state_id": edge.get("from"),
                    "url": source.get("url", ""),
                })

        output: list[dict[str, Any]] = []
        for capability in capabilities.values():
            capability["preconditions"] = _dedupe(capability["preconditions"], limit=16)
            capability["effects"] = _dedupe(capability["effects"], limit=16)
            capability["evidence"] = _dedupe(capability["evidence"], limit=12)
            capability["routes"] = _dedupe(capability["routes"], limit=12)
            verification = capability["verification"]
            verified = verification["verified_transitions"]
            observed = max(1, verification["observed_transitions"])
            has_evidence = any(item.get("evidence_signals") for item in capability["effects"])
            score = 0.20
            score += 0.25 if capability["name"] else 0.0
            score += 0.30 * min(1.0, verified / observed)
            score += 0.15 if capability["kind"] == "input" or capability["requires_confirmation"] else 0.0
            score += 0.10 if has_evidence else 0.0
            capability["usefulness"] = {
                "score": round(min(1.0, score), 3),
                "label": "high" if score >= 0.75 else "medium" if score >= 0.45 else "low",
                "basis": [
                    "visible_action",
                    "replay_verified" if verified else "observation_only",
                    "post_state_evidence" if has_evidence else "needs_runtime_result_check",
                ],
            }
            capability["risk"] = _risk(capability["action"])
            output.append(capability)

        output.sort(key=lambda item: (-item["usefulness"]["score"], item["name"].lower()))
        stats = self.graph.get("stats", {}) or {}
        return {
            "schema_version": "site2tools.manifest.v1",
            "start_url": self.graph.get("start_url", ""),
            "host": (urlparse(self.graph.get("start_url", "")).hostname or "").lower(),
            "generated_at": time.time(),
            "source_graph": {
                "states": stats.get("states", len(self.nodes)),
                "edges": stats.get("edges", len(self.edges)),
                "paths_considered": stats.get("paths_considered"),
            },
            "verification_policy": {
                "safe_transitions": "verified by browser replay when a target state is recorded",
                "external_writes": "observed and exposed only behind explicit confirmation",
                "completion": "requires visible evidence after execution",
            },
            "capabilities": output,
            "warnings": [
                "A guarded capability is a discovered action, not proof that it is safe to execute.",
                "The runtime must enforce host allowlists, confirmation gates and post-action evidence.",
            ],
        }

    @staticmethod
    def _new_capability(action: dict[str, Any]) -> dict[str, Any]:
        key = _action_key(action)
        requires_confirmation = bool(action.get("dangerous")) or str(action.get("input_type", "")).lower() in {"file", "password"}
        preconditions: list[dict[str, Any]] = []
        if requires_confirmation:
            preconditions.append({
                "type": "explicit_confirmation",
                "required": True,
                "reason": "This action can transmit data or change external state.",
            })
        schema = _input_schema(action)
        if schema and str(action.get("input_type", "")).lower() == "file":
            preconditions.append({
                "type": "authorized_file",
                "required": True,
                "field": next(iter(schema["properties"])),
            })
        return {
            "id": "cap-" + _hash(key),
            "name": _action_name(action),
            "description": "Use the observed website action: " + _action_name(action),
            "kind": action.get("kind", ""),
            "action": {
                "kind": action.get("kind", ""),
                "role": action.get("role", ""),
                "input_type": action.get("input_type", ""),
                "name": action.get("name", ""),
                "placeholder": action.get("placeholder", ""),
                "href": action.get("href", ""),
                "dangerous": bool(action.get("dangerous")),
            },
            "input_schema": schema,
            "requires_confirmation": requires_confirmation,
            "risk": _risk(action),
            "preconditions": preconditions,
            "effects": [],
            "evidence": [],
            "routes": [],
            "verification": {
                "status": "unverified",
                "observed_transitions": 0,
                "verified_transitions": 0,
                "guarded_transitions": 0,
            },
            "usefulness": {"score": 0.0, "label": "low", "basis": []},
            "action_key": key,
        }

    def _observe(
        self,
        capabilities: dict[str, dict[str, Any]],
        state_id: str,
        node: dict[str, Any],
        action: dict[str, Any],
    ) -> None:
        key = _action_key(action)
        capability = capabilities.setdefault(key, self._new_capability(action))
        capability["preconditions"].append({
            "type": "page_state",
            "state_id": state_id,
            "url": node.get("url", ""),
            "title": node.get("title", ""),
        })
        capability["evidence"].append({
            "kind": "observed_action",
            "state_id": state_id,
            "url": node.get("url", ""),
            "snippet": _clean(node.get("text_preview", ""))[:500],
        })

