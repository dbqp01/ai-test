"""Structured-data answers: JSON-LD, FAQ blocks and <select> options.

Reads what a site *declares* instead of pattern-matching prose. Deliberately independent of
core.py so the browser harness can adopt it (or not) without a merge.
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any

CHECKIN_WORDS = ("check-in", "checkin", "check in", "llegada")
CHECKOUT_WORDS = ("check-out", "checkout", "check out", "salida")
ADDRESS_WORDS = ("address", "direccion", "dirección", "ubicacion", "ubicación", "addressstrip")
PHONE_WORDS = ("phone", "telephone", "telefono", "teléfono", "contactnumber")
PRICE_WORDS = ("price", "precio", "tarifa", "rate")
BREAKFAST_WORDS = ("breakfast", "desayuno")
POOL_WORDS = ("pool", "piscina", "natation")
WIFI_WORDS = ("wifi", "wi-fi", "wireless")
ROOM_WORDS = ("room", "habitacion", "habitación")

PATTS: dict[str, tuple[str, ...]] = {
    "check-in": CHECKIN_WORDS,
    "check-out": CHECKOUT_WORDS,
    "direccion": ADDRESS_WORDS,
    "telefono": PHONE_WORDS,
    "precio": PRICE_WORDS,
    "desayuno": BREAKFAST_WORDS,
    "piscina": POOL_WORDS,
    "wifi": WIFI_WORDS,
}


def match_kind(goal: str) -> str | None:
    """Best-effort mapping from a Spanish/English goal to a fact kind.

    El orden importa: "precio por noche de la habitacion Doble Superior" menciona habitación
    *y* precio, y responder con la lista de habitaciones es lo que hacía la versión ingenua.
    """
    g = goal.lower()
    if "check" in g and "out" in g:
        return "check-out"
    if any(w in g for w in CHECKIN_WORDS):
        return "check-in"
    if any(w in g for w in ("hora", "horario", "cuando", "cuándo", "time")):
        if any(w in g for w in BREAKFAST_WORDS):
            return "desayuno"
        return "hora"
    if any(w in g for w in ("clave", "password", "contraseña", "contrasena", "pin")):
        return "wifi-clave"
    if any(w in g for w in PRICE_WORDS) or "cuanto" in g or "cuánto" in g:
        return "precio"
    if any(w in g for w in ADDRESS_WORDS):
        return "direccion"
    if any(w in g for w in PHONE_WORDS):
        return "telefono"
    if any(w in g for w in POOL_WORDS):
        return "piscina"
    if any(w in g for w in WIFI_WORDS):
        return "wifi"
    if any(w in g for w in ("tipo", "tipos", "cuantas", "cuántas", "lista")) and any(w in g for w in ROOM_WORDS):
        return "room-types"
    return None


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.jsonld: list[Any] = []
        self._script: dict[str, Any] | None = None
        self.selects: dict[str, list[str]] = {}
        self._select: tuple[str, list[str]] | None = None
        self._in_option = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script":
            if a.get("type") == "application/ld+json":
                self._script = {"buf": ""}
            return
        if tag == "select":
            key = a.get("name") or a.get("aria-label") or a.get("id") or "select"
            self._select = (key, [])
            return
        if tag == "option" and self._select is not None:
            self._in_option = True

    def handle_endtag(self, tag):
        if tag == "script" and self._script is not None:
            try:
                self.jsonld.append(json.loads(self._script["buf"]))
            except json.JSONDecodeError:
                pass
            self._script = None
        elif tag == "select" and self._select is not None:
            self.selects[self._select[0]] = self._select[1]
            self._select = None
        elif tag == "option":
            self._in_option = False

    def handle_data(self, data):
        if self._script is not None:
            self._script["buf"] += data
        elif self._in_option and self._select is not None:
            text = data.strip()
            if text:
                self._select[1].append(text)


def _walk(node: Any):
    """Yield every dict inside a JSON-LD graph, flattening @graph and nesting."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def extract(html_text: str) -> dict[str, list[dict[str, str]]]:
    parser = _Extractor()
    try:
        parser.feed(html_text)
    except Exception:
        pass

    facts: dict[str, list[dict[str, str]]] = {}

    def add(kind: str, value: Any, source: str) -> None:
        text = _fmt(value)
        if not text or text.lower() in {"true", "false", "none", "0"}:
            return
        entry = {"kind": kind, "value": text[:400], "source": source}
        if entry not in facts.setdefault(kind, []):
            facts[kind].append(entry)

    for node in parser.jsonld:
        for item in _walk(node):
            for key, value in item.items():
                lk = key.lower()
                if "checkintime" in lk:
                    add("check-in", value, "jsonld")
                elif "checkouttime" in lk:
                    add("check-out", value, "jsonld")
                elif lk in {"telephon", "phonenumber"} or lk.endswith("telephone"):
                    add("telefono", value, "jsonld")
                elif lk == "address" and isinstance(value, dict):
                    parts = [str(v) for k, v in value.items()
                             if k.lower() in {"streetaddress", "addresslocality", "addressregion",
                                              "postalcode", "addresscountry"} and v]
                    if parts:
                        add("direccion", ", ".join(parts), "jsonld")
                elif lk == "address" and isinstance(value, str):
                    add("direccion", value, "jsonld")
                elif lk == "amenityfeature" and isinstance(value, list):
                    for amenity in value:
                        if not isinstance(amenity, dict):
                            continue
                        name = str(amenity.get("name", ""))
                        if any(w in name.lower() for w in POOL_WORDS):
                            add("piscina", "si" if amenity.get("value") is not False else "no",
                                f"jsonld:{name}")
                        if any(w in name.lower() for w in WIFI_WORDS):
                            add("wifi", name, f"jsonld:{amenity.get('value')}")
                elif lk == "price" or lk == "pricerange":
                    # "$$" es un rango cualitativo del schema, no un precio: responder con él
                    # a "cuánto cuesta la noche" sería otro falso positivo.
                    if any(ch.isdigit() for ch in str(value)):
                        add("precio", value, "jsonld")
                    else:
                        continue
            answer = item.get("acceptedAnswer") or item.get("acceptedanswer")
            question = _fmt(item.get("name", "") or item.get("question", ""))
            if answer is not None and question:
                answer_text = _fmt(answer.get("text")) if isinstance(answer, dict) else _fmt(answer)
                if answer_text:
                    for kind in match_from_text(question):
                        add(kind, answer_text, f"faq:{question[:60]}")
                    # Aditivo: la mayoria de las FAQ del hotel ("Is medical oxygen included?",
                    # "How do I arrange an airport pickup?") no casa con ninguno de los 9 tipos,
                    # y se tiraba a la basura respuesta+pregunta. Guardandolas bajo un tipo
                    # genericos, `answer_for` (que solo mira los tipos conocidos) no cambia de
                    # comportamiento, y el emparejador pregunta-a-pregunta si puede usarlas.
                    add("faq", answer_text, f"faq:{question[:60]}")

    for kind, options in parser.selects.items():
        values = [o for o in options if o and o.lower() not in {"any room", "cualquiera", "select"}]
        if kind.lower() in {"roomtype", "room", "room_type"} or any(w in kind.lower() for w in ROOM_WORDS):
            add("room-types", " | ".join(values), f"select:{kind}")
        else:
            add(f"opciones:{kind}", " | ".join(values), "select")

    return facts


def match_from_text(text: str) -> list[str]:
    lowered = text.lower()
    kinds = []
    for kind, words in PATTS.items():
        if any(w in lowered for w in words):
            kinds.append(kind)
    if "check" in lowered and ("out" in lowered or "salida" in lowered):
        kinds = [k for k in kinds if k != "check-in"] or ["check-out"]
    return kinds


TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm|a\.m\.|p\.m\.)?\s*(?:-|–|to|a)\s*\d{1,2}[:.]\d{2}\s*(?:am|pm|a\.m\.|p\.m\.)?\b", re.I)


def answer_for(facts: dict[str, list[dict[str, str]]], goal: str) -> dict[str, Any] | None:
    kind = match_kind(goal)
    g = goal.lower()

    # La clave del wifi casi nunca se publica: "Free Wi-Fi" como respuesta a "clave del wifi"
    # sería otra alucinación del tipo del check-in de v4.
    if kind == "wifi-clave":
        for entry in facts.get("wifi", []):
            value = entry["value"]
            if not value.lower().startswith(("free wi", "wi-fi", "wifi")) and len(value) >= 6:
                return {"kind": "wifi-clave", "value": value, "source": entry["source"]}
        return None

    candidates = [kind] if kind else []
    if kind == "hora":
        candidates = (["check-in"] if "check" in g else []) + (["desayuno"] if "desayuno" in g else [])
    for key in candidates:
        for entry in facts.get(key, []):
            value = entry["value"]
            if key == "desayuno":
                # El FAQ responde en prosa; mejor el tramo horario literal con la frase de contexto.
                when = TIME_RE.search(value)
                if when:
                    return {"kind": "desayuno-hora", "value": when.group(0).strip(),
                            "source": entry["source"], "context": value[:220]}
            return {"kind": key, "value": value, "source": entry["source"]}
    return None


def extract_from_url(url: str, timeout: int = 30) -> dict[str, list[dict[str, str]]]:
    import urllib.request

    request = urllib.request.Request(url, headers={"user-agent": "Mozilla/5.0"})
    body = urllib.request.urlopen(request, timeout=timeout).read().decode("utf-8", "replace")
    return extract(body)
