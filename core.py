from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from playwright.sync_api import Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from .policy import ModelDecisionPolicy


DANGEROUS_WORDS = {
    "submit", "send", "enviar", "entregar", "delete", "eliminar", "remove",
    "buy", "purchase", "pagar", "pay", "publish", "publicar", "post", "transfer",
    "confirm", "confirmar", "grade", "calificar", "enroll", "matricular",
}


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return set(re.findall(r"[a-z0-9]{3,}", normalized.lower()))


def concepts(value: str) -> set[str]:
    """Small multilingual normalizer used only by the bootstrap policy."""
    result: set[str] = set()
    for word in tokens(value):
        result.add(word)
        if len(word) > 4 and word.endswith("s"):
            result.add(word[:-1])
        if len(word) > 5:
            result.add(word[:6])
    return result


def short_hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


def host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def allowed_url(url: str, allowed_hosts: set[str]) -> bool:
    current = host(url)
    return bool(current) and any(current == item or current.endswith("." + item) for item in allowed_hosts)


@dataclass
class Action:
    action_id: str
    kind: str
    label: str
    role: str = ""
    href: str = ""
    input_type: str = ""
    name: str = ""
    placeholder: str = ""
    dangerous: bool = False
    enabled: bool = True

    @property
    def signature(self) -> str:
        return "|".join((self.kind, clean(self.label).lower(), self.href, self.role))


@dataclass
class Observation:
    state_id: str
    url: str
    title: str
    text: str
    actions: list[Action]
    network: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class JsonStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "traces").mkdir(exist_ok=True)

    def append(self, filename: str, item: dict[str, Any]) -> None:
        with (self.root / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def save_graph(self, graph: dict[str, Any]) -> None:
        temporary = self.root / "site_graph.json.tmp"
        temporary.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.root / "site_graph.json")


class BrowserSession:
    """Owns an isolated browser, observation, execution and safety checks."""

    def __init__(
        self,
        start_url: str,
        data_dir: Path,
        allowed_hosts: Iterable[str] | None = None,
        trace_name: str | None = None,
    ):
        self.start_url = start_url
        self.allowed_hosts = {host(start_url)} | {item.lower() for item in (allowed_hosts or [])}
        self.store = JsonStore(data_dir)
        self.trace_name = trace_name
        self.network: list[dict[str, Any]] = []
        self.playwright = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    def __enter__(self) -> "BrowserSession":
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context()
        if self.trace_name:
            self.context.tracing.start(screenshots=True, snapshots=True, sources=True)
        self.page = self.context.new_page()
        self.page.on("request", lambda request: self.network.append({
            "type": "request", "method": request.method, "url": request.url,
        }))
        self.page.on("response", lambda response: self.network.append({
            "type": "response", "status": response.status, "url": response.url,
        }))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.context and self.trace_name:
            path = self.store.root / "traces" / (self.trace_name + ".zip")
            self.context.tracing.stop(path=str(path))
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def goto(self, url: str) -> None:
        if not allowed_url(url, self.allowed_hosts):
            raise ValueError("host no permitido: " + host(url))
        assert self.page is not None
        self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        self.page.wait_for_timeout(250)

    def observe(self) -> Observation:
        assert self.page is not None
        raw = self.page.evaluate(
            r"""
            () => {
              const selector = 'a,button,input,textarea,select,[role="button"],[role="link"],[role="tab"]';
              const nodes = [...document.querySelectorAll(selector)];
              const actions = [];
              for (let i = 0; i < nodes.length; i++) {
                const element = nodes[i];
                const rect = element.getBoundingClientRect();
                const style = getComputedStyle(element);
                if (rect.width === 0 || rect.height === 0 || style.visibility === 'hidden' || style.display === 'none') continue;
                const tag = element.tagName.toLowerCase();
                const label = (element.innerText || element.getAttribute('aria-label') || element.getAttribute('placeholder') || element.getAttribute('name') || '').trim().replace(/\s+/g, ' ');
                const role = element.getAttribute('role') || (tag === 'a' ? 'link' : tag === 'button' ? 'button' : tag);
                const kind = ['input', 'textarea', 'select'].includes(tag) ? 'input' : 'click';
                const actionId = 's2t-' + i;
                element.setAttribute('data-site2tools-id', actionId);
                actions.push({
                  action_id: actionId,
                  kind,
                  label,
                  role,
                  href: element.href || '',
                  input_type: element.type || '',
                  name: element.getAttribute('name') || '',
                  placeholder: element.getAttribute('placeholder') || '',
                  enabled: !element.disabled
                });
              }
              return {
                url: location.href,
                title: document.title || '',
                text: (document.body ? document.body.innerText : '').replace(/\s+/g, ' ').trim().slice(0, 20000),
                actions
              };
            }
            """
        )
        actions: list[Action] = []
        for item in raw["actions"]:
            signal = " ".join((item.get("label", ""), item.get("name", ""), item.get("placeholder", ""))).lower()
            item["dangerous"] = bool(tokens(signal) & DANGEROUS_WORDS)
            actions.append(Action(**item))
        observation = Observation(
            state_id=short_hash(raw["url"], clean(raw["text"])[:5000], "|".join(a.signature for a in actions)),
            url=raw["url"],
            title=clean(raw["title"]),
            text=clean(raw["text"]),
            actions=actions,
            network=self.network[-100:],
        )
        self.store.append("traces.jsonl", {"type": "observation", **asdict(observation)})
        return observation

    def check_host(self) -> None:
        assert self.page is not None
        if not allowed_url(self.page.url, self.allowed_hosts):
            raise ValueError("la navegación salió de hosts permitidos: " + host(self.page.url))

    def execute(
        self,
        action: Action,
        *,
        allow_writes: bool = False,
        input_value: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        assert self.page is not None
        if not action.enabled:
            return {"ok": False, "reason": "disabled"}
        if action.dangerous and not allow_writes:
            return {"ok": False, "reason": "confirmation_required", "action": asdict(action)}
        locator = self.page.locator('[data-site2tools-id="' + action.action_id + '"]').first
        try:
            if action.kind == "input":
                if action.input_type == "file":
                    if not file_path:
                        return {"ok": False, "reason": "file_required"}
                    locator.set_input_files(file_path)
                elif input_value is not None:
                    locator.fill(input_value)
                else:
                    return {"ok": False, "reason": "input_value_required"}
            else:
                locator.click(timeout=10000)
            self.page.wait_for_timeout(300)
            self.check_host()
            return {"ok": True, "url": self.page.url}
        except (PlaywrightTimeoutError, ValueError) as exc:
            return {"ok": False, "reason": str(exc)}


class WebsiteExplorer:
    def __init__(self, start_url: str, data_dir: Path, max_states: int = 25, max_depth: int = 3,
                 allowed_hosts: Iterable[str] | None = None, max_paths: int | None = None):
        self.start_url = start_url
        self.data_dir = data_dir
        self.max_states = max_states
        self.max_depth = max_depth
        self.allowed_hosts = allowed_hosts
        self.max_paths = max_paths or max(40, max_states * 4)
        self.graph: dict[str, Any] = {
            "start_url": start_url,
            "nodes": {},
            "edges": [],
            "generated_at": time.time(),
        }

    def explore(self) -> dict[str, Any]:
        frontier: list[list[str]] = [[]]
        visited_paths: set[tuple[str, ...]] = set()
        seen_states: set[str] = set()
        with BrowserSession(self.start_url, self.data_dir, self.allowed_hosts) as session:
          while frontier and len(seen_states) < self.max_states and len(visited_paths) < self.max_paths:
            path = frontier.pop(0)
            path_key = tuple(path)
            if path_key in visited_paths or len(path) > self.max_depth:
                continue
            visited_paths.add(path_key)
            session.goto(self.start_url)
            replay_ok = True
            for signature in path:
                observation = session.observe()
                candidate = next((item for item in observation.actions if item.signature == signature), None)
                if not candidate or candidate.dangerous:
                    replay_ok = False
                    break
                if not session.execute(candidate).get("ok"):
                    replay_ok = False
                    break
            if not replay_ok:
                continue
            observation = session.observe()
            self.graph["nodes"].setdefault(observation.state_id, {
                "state_id": observation.state_id,
                "url": observation.url,
                "title": observation.title,
                "text_preview": observation.text[:1000],
                "actions": [asdict(action) for action in observation.actions],
            })
            seen_states.add(observation.state_id)
            for action in observation.actions:
                if action.dangerous or not action.enabled:
                    continue
                next_path = path + [action.signature]
                frontier.append(next_path)
                self.graph["edges"].append({
                    "from": observation.state_id,
                    "action": asdict(action),
                    "path": next_path,
                })
        self.graph["stats"] = {
            "states": len(self.graph["nodes"]),
            "edges": len(self.graph["edges"]),
            "paths_considered": len(visited_paths),
        }
        JsonStore(self.data_dir).save_graph(self.graph)
        return self.graph


class UniversalOperator:
    def __init__(self, start_url: str, data_dir: Path, max_steps: int = 12,
                 allowed_hosts: Iterable[str] | None = None,
                 policy: ModelDecisionPolicy | None = None):
        self.start_url = start_url
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.allowed_hosts = allowed_hosts
        self.policy = policy

    @staticmethod
    def choose(goal: str, observation: Observation, context: dict[str, Any] | None = None,
               uploaded: bool = False, used_actions: set[tuple[str, str]] | None = None) -> Action | None:
        context = context or {}
        used_actions = used_actions or set()
        file_requested = bool(context.get("file") or context.get("files"))
        if file_requested and not uploaded:
            file_actions = [
                action for action in observation.actions
                if action.kind == "input" and action.input_type == "file"
                and (observation.state_id, action.signature) not in used_actions
            ]
            if file_actions:
                return file_actions[0]
        goal_tokens = concepts(goal)
        best: tuple[float, Action] | None = None
        for action in observation.actions:
            if (observation.state_id, action.signature) in used_actions:
                continue
            if action.kind == "click" and action.href and action.href == observation.url:
                continue
            label_tokens = concepts(" ".join((action.label, action.name, action.placeholder, action.href)))
            score = float(len(goal_tokens & label_tokens))
            if action.kind == "input":
                score -= 0.25
            if action.dangerous:
                score += 0.05
            if best is None or score > best[0]:
                best = (score, action)
        return best[1] if best and best[0] > 0 else None

    def run(self, goal: str, context: dict[str, Any] | None = None, confirm: bool = False,
            allow_writes: bool = False) -> dict[str, Any]:
        context = context or {}
        steps: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        uploaded = False
        used_actions: set[tuple[str, str]] = set()
        with BrowserSession(self.start_url, self.data_dir, self.allowed_hosts,
                            trace_name="operation-" + str(int(time.time()))) as session:
            session.goto(self.start_url)
            for index in range(self.max_steps):
                observation = session.observe()
                policy_decision: dict[str, Any] | None = None
                if self.policy is not None:
                    try:
                        policy_decision = self.policy.decide(
                            goal, context, observation, history,
                            write_confirmed=bool(confirm and allow_writes),
                        )
                    except Exception as exc:
                        policy_decision = {"_error": type(exc).__name__ + ": " + str(exc)}

                action = None
                if policy_decision and policy_decision.get("decision") == "block":
                    result = {
                        "status": "blocked",
                        "reason": "policy_blocked",
                        "policy": policy_decision,
                        "url": observation.url,
                        "steps": steps,
                        "evidence": observation.text[:2000],
                    }
                    JsonStore(self.data_dir).append("operations.jsonl", result)
                    return result
                if policy_decision and policy_decision.get("decision") == "finish":
                    lower = observation.text.lower()
                    if any(word in lower for word in (
                        "submitted", "enviada", "entregada", "confirmation id", "comprobante",
                        "download id", "ticket id", "order id", "status: running",
                    )):
                        result = {
                            "status": "completed",
                            "url": observation.url,
                            "steps": steps,
                            "evidence": observation.text[:3000],
                            "policy": policy_decision,
                        }
                        JsonStore(self.data_dir).append("operations.jsonl", result)
                        return result
                if policy_decision and policy_decision.get("decision") in {"act", "request_confirmation"}:
                    action_index = policy_decision.get("action_index")
                    if isinstance(action_index, int) and 0 <= action_index < len(observation.actions):
                        candidate = observation.actions[action_index]
                        if candidate.enabled:
                            action = candidate
                if action is None:
                    action = self.choose(goal, observation, context, uploaded, used_actions)
                if not action:
                    result = {
                        "status": "blocked",
                        "reason": "no_matching_action",
                        "url": observation.url,
                        "steps": steps,
                        "evidence": observation.text[:2000],
                        "policy": policy_decision,
                    }
                    JsonStore(self.data_dir).append("operations.jsonl", result)
                    return result
                if action.dangerous and not (confirm and allow_writes):
                    result = {
                        "status": "confirmation_required",
                        "url": observation.url,
                        "planned_action": asdict(action),
                        "steps": steps,
                        "policy": policy_decision,
                    }
                    JsonStore(self.data_dir).append("operations.jsonl", result)
                    return result
                value = None
                file_path = None
                if action.kind == "input":
                    inputs = context.get("inputs", {})
                    files = context.get("files", {})
                    value = inputs.get(action.name) or inputs.get(action.placeholder)
                    file_path = files.get(action.name) or files.get(action.placeholder) or context.get("file")
                act_result = session.execute(action, allow_writes=allow_writes,
                                             input_value=value, file_path=file_path)
                steps.append({
                    "index": index,
                    "observation": observation.state_id,
                    "action": asdict(action),
                    "result": act_result,
                    "policy": policy_decision,
                })
                history.append({
                    "state_url": observation.url,
                    "action": action.label,
                    "result": act_result.get("reason", "ok") if isinstance(act_result, dict) else str(act_result),
                })
                used_actions.add((observation.state_id, action.signature))
                if not act_result.get("ok"):
                    result = {"status": "error", "error": act_result, "steps": steps}
                    JsonStore(self.data_dir).append("operations.jsonl", result)
                    return result
                if action.kind == "input" and action.input_type == "file":
                    uploaded = True
                after = session.observe()
                lower = after.text.lower()
                if any(word in lower for word in (
                    "submitted", "enviada", "entregada", "confirmation id", "comprobante"
                )):
                    result = {
                        "status": "completed",
                        "url": after.url,
                        "steps": steps,
                        "evidence": after.text[:3000],
                        "policy": policy_decision,
                    }
                    JsonStore(self.data_dir).append("operations.jsonl", result)
                    return result
            result = {"status": "budget_exhausted", "steps": steps}
            JsonStore(self.data_dir).append("operations.jsonl", result)
            return result
