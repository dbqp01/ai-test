from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field, replace
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

# Rutas que inician una transacción real aunque el enlace lo ponga el sitio de reservas
# de un tercero; con writes OFF deben requerir confirmación igual que un botón "Pagar".
WRITE_PATH_WORDS = ("cancel", "checkout", "pagar", "pago", "comprar", "confirmar", "finalizar-pedido")

# El operator entrena `finish` en 54 de 12.779 filas: casi nunca cierra una tarea de lectura
# aunque tenga la respuesta delante, y agota el presupuesto clicando. Para objetivos
# informativos se saca el literal de la página en vez de esperar esa decisión.
ANSWER_KINDS = (
    ("precio", ("precio", "coste", "tarifa", "cuesta", "vale", "price", "rate", "cost"),
     r"(?:US\$\s?|\$\s?|S/\s?|€\s?)\d[\d.,]*"),
    ("hora", ("hora", "horario", "cuando", "time", "schedule", "opens", "closes"),
     r"\b\d{1,2}[:.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?|am|pm)?\b"),
    ("telefono", ("telefono", "numero", "phone", "movil", "whatsapp"),
     r"\+?\d[\d\s().-]{7,}\d"),
    ("correo", ("correo", "email", "mail", "electronico"),
     r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    # Dirección leída en la página real (25-sep): /contact/ dice `759 CALLE HOSPITAL, CUSCO` y
    # `759 Calle Hospital, Centro de Cusco`. El puente es->en daba "address"/"directions" y la
    # página escribe "Calle", así que sin este patrón la dirección publicada seguía sin responder.
    ("direccion", ("direccion", "address", "ubicacion", "location", "localizacion", "street",
                   "donde", "domicilio"),
     r"\d{1,5}\s+(?:Calle|Avenida|Av\.?|Jir[oó]n|Jr\.?|Pasaje|Callej[oó]n)\s+"
     r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9.'\- ]{3,40}"
     r"|(?:Calle|Avenida|Av\.?)\s+[A-Za-zÁÉÍÓÚáéíóúÑñ.'\- ]{3,30}\s+\d{1,5}"),
)


GENERIC_GOAL_WORDS = {"hotel", "hoteles", "hostal", "page", "pages", "website", "web", "site"}
# Interrogativos y verbos de apoyo: medido en el diagnostico de `gastronomia`, los terminos del
# objetivo salian como ['cafe','cafeteria','coffee','dining','que','restaurant','restaurante',
# 'tiene']. "que" y "tiene" no son informacion, y al contar como solape hacian que cualquier frase
# con un "que" fuera candidata a evidencia -- y hasta ganara el ranking. Se quitan de la lista de
# terminos (no del parseo): seguiran sin casar con nada porque no aportan nada.
GENERIC_GOAL_WORDS |= {
    "que", "cual", "cuales", "como", "cuando", "donde", "quien", "cuyo", "adonde",
    "tiene", "tienen", "tenia", "hay", "han", "ha", "es", "son", "ser", "esta", "estan",
    "puedo", "puedes", "puede", "pueden", "hace", "hacen", "necesito", "necesitas",
    "quiero", "quieres", "porfavor", "dime", "indique", "favor",
    "what", "which", "where", "when", "how", "why", "who", "does", "dont", "cant",
    "is", "are", "was", "were", "can", "could", "would", "should", "tell", "please",
    "about", "with", "from", "there", "here", "many", "much", "any", "some", "these",
}
# Una reseña de Booking/TripAdvisor es texto que el hotel muestra, pero no es un dato del hotel:
# "en que barrio esta el hotel" contestó con `"It's a nice older building with a walk-around
# courtyard"` de una reseña. No se descarta (a veces es lo único), se penaliza para que una frase
# propia del sitio gane cuando exista.
REVIEW_MARKERS = ("booking.com", "tripadvisor", "google", "guests say", "what our guests", "verified")
# Radio en el que se busca el sello de resena, mas ancho que la ventana que se devuelve: el
# carrusel del sitio pone "BOOKING.COM" despues de la cita, no dentro de ella.
REVIEW_SCAN_RADIUS = 400

# Glifos ornamentales medidos en usgarhoteles.com: el ticker de destinos ("San Pedro ✦ Cusco ✦
# Machu Picchu ✦ ...") y el cierre del carrusel de resenas los usan como costura entre bloques
# que innerText llega pegados. Dividir por ellos separa la cola de una resena ajena del dato
# propio que viene justo despues. Solo los dos que se ven en el sitio: el punto medio "·" y el
# rombo si se llegaran a medir, porque cada glifo anadido tambien relaja el descarte de resenas.
ORNAMENT_SEPARATORS = "\u2726\u2022"


def ornament_segments(text: str, base: int) -> list[tuple[int, int, str]]:
    """`text` partido por glifos ornamentales, con offsets absolutos dentro del cuerpo.

    Un solo segmento cuando no hay ningun glifo: la costura existe o no existe, y de eso depende
    que el candidato (y por tanto su contexto de descarte) sea el bloque entero o una pieza.
    """
    out: list[tuple[int, int, str]] = []
    pos = 0
    for m in re.finditer("[" + ORNAMENT_SEPARATORS + "]", text):
        out.append((base + pos, base + m.start(), text[pos:m.start()]))
        pos = m.end()
    out.append((base + pos, base + len(text), text[pos:]))
    return [g for g in out if len(clean(g[2])) >= 25] or [(base, base + len(text), text)]


def review_region(body: str, start: int, end: int) -> str:
    """Contexto que puede descalificar el chunk `body[start:end]` como palabra de un tercero.

    El radio a pelo sobre el interior del texto era demasiado corto y demasiado largo a la vez:
    "We had to leave at 4am for our tour - they packed breakfast for all of us." llega como chunk
    sin marca dentro (su "BOOKING.COM" esta dos chunks antes), mientras que el bloque propio
    "CULINARY & COFFEE 04 AUKA RESTOBAR..." quedaba descartado por un BOOKING.COM de la resena
    anterior. El glifo ornamental es la costura real entre bloques, asi que se recorta por el y
    se le pone techo de radio para que una pagina sin adornos no herede el carrusel entero.
    """
    lo = max((body.rfind(g, 0, start) for g in ORNAMENT_SEPARATORS), default=-1) + 1
    after = [h for h in (body.find(g, end) for g in ORNAMENT_SEPARATORS) if h >= 0]
    hi = min(after, default=len(body))
    lo = max(lo, start - REVIEW_SCAN_RADIUS)
    hi = min(hi, end + REVIEW_SCAN_RADIUS)
    return body[lo:max(lo, hi)]


TIME_ANCHORS = ("breakfast", "desayuno", "check-in", "checkin", "check in", "check-out", "checkout",
                "check out", "llegada", "salida", "arrival", "departure", "open", "close", "opens",
                "closes", "hours", "hora", "horario", "lunch", "dinner", "cafeteria", "front desk")


def anchored_time(text: str) -> dict[str, Any] | None:
    """Una hora solo si está pegada a la palabra que la nombra.

    Leído en /book/ del hotel: `Check-in: 12:00 hrs | Check-out: 10:30 hrs`. Con el patrón libre,
    la pregunta "a qué hora es el check-in" devolvía `6:00` de otro sitio de la página: una hora
    equivocada dicha con seguridad es peor que no contestar.
    """
    low = (text or "").lower()
    for anchor in TIME_ANCHORS:
        for m in re.finditer(re.escape(anchor), low):
            window = (text or "")[m.start(): m.end() + 60]
            tm = re.search(r"\d{1,2}[:.]\d{2}(?:\s*(?:hrs?|am|pm|a\.m\.|p\.m\.))?", window, re.I)
            if tm:
                start = max(0, m.start() - 45)
                return {
                    "kind": "hora",
                    "value": clean(tm.group(0)),
                    "context": clean((text or "")[start: m.end() + 70]),
                }
    return None


def extractive_answer(goal: str, text: str) -> dict[str, Any] | None:
    """Respuesta extractiva cuando el objetivo es una pregunta sin literal reconocible.

    "Tiene piscina", "que tipos de habitacion hay" o "cual es la direccion" no casan con ningún
    patrón de precio/hora/telefono, y sin esto el operador agotaba el presupuesto clicando pese a
    tener la frase en pantalla. Se puntúan frases por solape con el objetivo (puente es->en) y se
    devuelve la mejor: es evidencia literal de la página, no una invención del modelo.
    """
    # Exactitud aquí, recall en el ranking. `concepts()` añade el prefijo de 6 caracteres de cada
    # palabra (core.py:47) y con eso `'directions'[:6] == 'direct'`: un "Direct Booking" del menú
    # contestaba a "cual es la direccion". Para decidir que una frase ES la evidencia se comparan
    # palabras completas; para puntear candidatos el prefijo da igual, porque solo ordena.
    gt = tokens(expand_goal(goal)) - GENERIC_GOAL_WORDS
    if not gt:
        return None
    best: tuple[float, str] | None = None
    # Primero el cuerpo: los ~400 primeros caracteres de este sitio son navegación y title, y el
    # title ("USGAR Hotels | San Pedro, Cusco - Your gateway to the Andes") respondía `direccion`.
    for body in ((text or "")[400:], (text or "")):
        # Con offsets, no con re.split: hay que saber donde cae cada chunk dentro del cuerpo para
        # poder mirar su contexto antes de aceptarlo como evidencia.
        pieces: list[tuple[int, str]] = []
        pos = 0
        for sep in re.finditer(r"(?<=[.!?\n])\s+", body):
            pieces.append((pos, body[pos:sep.start()]))
            pos = sep.end()
        pieces.append((pos, body[pos:]))
        for start, chunk in pieces:
            s = clean(chunk)
            if len(s) < 25:
                continue
            # Guiones: en el texto real del sitio el wifi se escribe "Wi-Fi", y `tokens()` pide 3
            # caracteres o más, así que "wifi" no salía nunca y la pregunta del wifi no casaba.
            words = tokens(s.replace("-", ""))
            if len(words) < 5:
                continue
            hits = (gt & words) - GENERIC_GOAL_WORDS
            # Con el puente es->en casi siempre queda UN solo token común ("piscina"->"pool"), así
            # que el corte es 1 y la discriminación la hacen las comodines: sin restar "hotel",
            # cualquier title del sitio casaba con cualquier pregunta.
            if not hits:
                continue
            # Costura: el innerText del sitio pega en un mismo chunk de ~290 caracteres la cola de
            # una resena de Booking, el ticker de destinos y el dato propio ("CULINARY & COFFEE 04
            # AUKA RESTOBAR, An unforgettable culinary journey..."). Se candidatea por segmento, que
            # es lo que tambien acota el contexto con el que se le puede descalificar.
            segs = ornament_segments(chunk, start)
            a, b = segs[0][0], segs[0][1]
            if len(segs) > 1:
                a, b, _ = max(segs, key=lambda g: (len((gt & tokens(clean(g[2]).replace("-", "")))
                                                       - GENERIC_GOAL_WORDS), len(g[2])))
                s = clean(body[a:b])
                words = tokens(s.replace("-", ""))
                hits = (gt & words) - GENERIC_GOAL_WORDS
                if len(s) < 25 or len(words) < 4 or not hits:
                    continue
            # El sello que descalifica puede caer fuera del candidato: "We had to leave at 4am for
            # our tour - they packed breakfast for all of us." llega como chunk limpio, sin marca
            # dentro, y su "BOOKING.COM" esta unos metros antes en el cuerpo.
            scan = review_region(body, a, b).lower()
            if len(s) > 320:
                # El innerText de una SPA llega casi sin puntos: "Evening Cafeteria" vivía dentro
                # de un bloque de miles de caracteres que el viejo tope de 320 rechazaba entero, y
                # por eso `gastronomia` no contestaba teniendo el dato delante. Se ventana alrededor
                # del termino encontrado en vez de tirar el bloque.
                low = s.lower()
                positions = [low.find(h) for h in hits]
                positions = [p for p in positions if p >= 0]
                if not positions:
                    continue
                first = min(positions)
                lo = max(0, first - 120)
                hi = min(b - a, first + 190)
                s = clean(body[a + lo: a + hi])
                scan = review_region(body, a + lo, a + hi).lower()
                words = tokens(s.replace("-", ""))
                hits = (gt & words) - GENERIC_GOAL_WORDS
                if len(s) < 25 or len(words) < 4 or not hits:
                    continue
            # Una reseña citada no es un dato del hotel: "en que barrio esta el hotel" contestaba
            # con el texto de un huésped en Booking.com. Penalizar no arreglaba nada cuando no hay
            # otra candidata, así que se descarta y se prefiere `None` (y que el operador siga
            # buscando) antes que afirmar con palabras de un tercero.
            if any(marker in scan for marker in REVIEW_MARKERS):
                continue
            score = len(hits) / (1.0 + 0.03 * len(words))
            if best is None or score > best[0]:
                best = (score, s)
        if best is not None:
            break
    # Sin umbral sobre la puntuación: la longitud solo sirve para ORDENAR candidatos. Exigirle un
    # mínimo a una magnitud normalizada por palabras descarta justo la evidencia cross-lingüe de
    # un solo token ("wifi" en una frase de 30 palabras), que es el caso de uso del producto.
    if not best:
        return None
    value = best[1]
    if len(value) > 460:
        # Este sitio no corta frases y el trozo empezaba en la navegación: "Skip to main content
        # ROOMS SERVICES … ACCOMMODATIONS Our Rooms…". Se recorta el arranque solo si la evidencia
        # sigue estando presente después del recorte.
        trimmed = value[400:]
        if ((gt & tokens(trimmed.replace("-", ""))) - GENERIC_GOAL_WORDS):
            value = trimmed
    return {"kind": "extractiva", "value": clean(value), "context": clean(value)}


CURRENCY_RE = r"(?:US\$\s?|\$\s?|S/\s?|€\s?)\d[\d.,]*"


def anchored_price(goal: str, text: str) -> dict[str, Any] | None:
    """Precio de la habitacion que se pregunta, no del primer $ que aparece.

    Medido al asentar la SPA: la misma pregunta contesto primero `$90` (el "FROM $90 PER NIGHT"
    de la ficha de Doble Superior en /rooms/) y despues `$65` (la cotizacion de /book/ para las
    fechas de ejemplo). Los dos son ciertos y son cosas distintas; un extractor de precios que
    coge la primera moneda que ve no puede contar cual es la del huésped preguntado. Se ancla a
    los nombres de habitacion que el objetivo menciona, como se hizo con las horas.
    """
    low = (text or "").lower()
    subject = tokens(expand_goal(goal)) - GENERIC_GOAL_WORDS
    subject |= {w for w in tokens(goal) if len(w) >= 5}
    anchors = [m.start() for w in subject if len(w) >= 4 for m in re.finditer(re.escape(w), low)]
    if not anchors:
        return None
    matches = list(re.finditer(CURRENCY_RE, text or "", re.IGNORECASE))
    if not matches:
        return None
    # Cada precio "posee" el tramo que va desde el precio anterior hasta el suyo: con una ventana
    # de +-150 las tarjetas pegadas de /rooms/ empataban en puntuacion y la cercania ganaba la
    # erronea ($75 de la Superior Matrimonial en vez de $90 de la Doble Superior preguntada).
    best: tuple[int, int, str, str] | None = None
    # El tramo de un precio incluye SU unidad ("$90 PER NIGHT"): si no, el "PER NIGHT" de la
    # tarjeta anterior sube al segmento de la siguiente y la puntuacion se invierte (medido al
    # correr el test en el orden "Doble Superior primero", donde $75 ganaba 3-2).
    unit = re.compile(r"\s*(?:per|por)\s+(?:night|noche|noches)\b", re.IGNORECASE)
    prev_end = 0
    for m in matches:
        segment = (text or "")[prev_end: m.start()]
        score = len(subject & tokens(segment))
        tail = unit.match((text or "")[m.end():])
        prev_end = m.end() + (len(tail.group(0)) if tail else 0)
        if score < 2:
            continue
        dist = min(abs(m.start() - a) for a in anchors)
        cand = (score, -dist, clean(m.group(0)),
                clean(segment[-160:] + " " + m.group(0) + " " + (text or "")[m.end(): m.end() + 60]))
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    if not best:
        return None
    return {"kind": "precio", "value": best[2], "context": best[3]}


def pick_answer(goal: str, text: str) -> dict[str, Any] | None:
    """Literal si el objetivo pide un literal; extractiva solo si no pide ninguno.

    El fallback extractivo NO puede sustituir a un patrón: con el objetivo "a qué hora se sirve
    el desayuno" devolvió la frase `Family Superior Room FROM $150` porque era la que más
    solape tenía. Un asistente de hotel que contesta una hora con un precio es peor que uno que
    no contesta, así que si el objetivo es de tipo precio/hora/telefono/correo y ese literal no
    está en la página, la respuesta correcta es `None`.
    """
    kind_set = concepts(goal)
    triggered = False
    for kind, triggers, pattern in ANSWER_KINDS:
        if not (kind_set & set(triggers)):
            continue
        triggered = True
        if kind == "hora":
            return anchored_time(text)
        if kind == "precio":
            anchored = anchored_price(goal, text)
            if anchored:
                return anchored
        match = re.search(pattern, text or "", re.IGNORECASE)
        if not match:
            continue
        start = max(0, match.start() - 90)
        return {
            "kind": kind,
            "value": clean(match.group(0)),
            "context": clean(text[start:min(len(text), match.end() + 170)]),
        }
    return None if triggered else extractive_answer(goal, text)


def goal_terms(goal: str) -> list[str]:
    """Palabras de contenido del objetivo, con sus puentes al inglés, para saber si el sitio
    siquiera habla de eso. Sin esto, una pregunta sobre un dato que el sitio no publica
    (`password`/`address` no aparecen en ninguna de las 867 páginas observadas de usgarhoteles.com)
    se devuelve como un fallo del agente cuando es un hecho del sitio."""
    stop = {"cual", "cuales", "que", "tiene", "hay", "sono", "como", "donde", "cuando",
            "hotel", "hoteles", "sitio", "pagina", "web", "the", "and", "for", "with",
            "hay", "esta", "estan", "puedo", "ser", "era", "sobre"}
    english = {w for value in ES_EN.values() for w in value.split()}
    out = [w for w in sorted(tokens(expand_goal(goal)), key=lambda w: (w not in english, w))
           if len(w) >= 4 and w not in stop and w[:4] not in {"cual", "tien", "hote", "ques"}]
    return out[:12]


try:  # Conocimiento estructurado del sitio (JSON-LD + FAQ + <select>): suelo determinista.
    from structured_answers import answer_for as structured_answer_for, extract_from_url as structured_extract
except ImportError:  # El operador sigue funcionando sin él.
    structured_answer_for = None
    structured_extract = None


YES_NO_MARKERS = ("puedo", "tiene", "tienen", "hay", "ofrecen", "dan", "incluye", "aceptan",
                  "cuentan", "es posible", "como llego", "donde")
TEMPORAL_OR_NUMERIC_KINDS = {"hora", "check-in", "check-out", "precio", "telefono"}


def faq_answer_for_question(facts: dict[str, Any], goal: str) -> dict[str, Any] | None:
    """Respuesta de FAQ emparejada pregunta-con-pregunta, no por tipo de dato.

    Medido en el bench de 16: "puedo dejar el equipaje despues del check-out" devolvía `10:30`
    porque la FAQ "Can I store my luggage before check-in or after check-out?" se guarda bajo el
    tipo `check-out`. La respuesta buena estaba ahí mismo, en el mismo registro, con la pregunta
    en `source`. Comparar preguntas con palabras del objetivo (puente es->en mediante) la
    encuentra sin tocar el módulo de otra sesión.
    """
    gt = {w for w in tokens(expand_goal(goal)) - GENERIC_GOAL_WORDS if len(w) >= 4}
    if not gt:
        return None
    best: tuple[float, dict[str, Any]] | None = None
    for entries in (facts or {}).values():
        for entry in entries or []:
            source = str(entry.get("source") or "")
            if not source.startswith("faq:"):
                continue
            question = source[4:]
            hits = gt & tokens(question.lower())
            if not hits:
                continue
            score = len(hits) / (1.0 + 0.05 * len(tokens(question.lower())))
            if best is None or score > best[0]:
                best = (score, entry)
    return best[1] if best else None


def structured_shortcut(goal: str, urls: list[str]) -> dict[str, Any] | None:
    """Primera respuesta por datos estructurados, antes de gastar un solo paso de navegador.

    Aportado por la sesión bc04c579 y medido offline: resuelve 5 de las 8 preguntas del benchmark
    (check-in 12:00 desde `checkinTime` del JSON-LD, dirección, teléfono, desayuno, tipos de
    habitación desde el `<select name=room-type>`) y se niega a responder precio/wifi/piscina
    cuando el sitio no lo publica. Es más exacto que cualquier regex sobre el texto: el check-in
    servido por la política era `6:00` (la hora del desayuno) cuando el sitio dice `12:00`.

    Se prueban varias URLs porque el JSON-LD vive en la raíz: medido al relanzar con el atajo,
    `desayuno` se respondió en 0 pasos y 1 s, pero `check-in` seguía navegando y dando `6:00`
    porque solo se consultaba la página enrutada (`/book/`), que no lleva el bloque estructurado.
    """
    if structured_answer_for is None or structured_extract is None:
        return None
    saw_wifi = False
    for url in [u for u in urls if u]:
        try:
            facts = structured_extract(url)
            found = structured_answer_for(facts, goal)
        except Exception:  # un fallo de red o de parseo no puede tumbar la operación
            continue
        if found and found.get("value"):
            g = (goal or "").lower()
            kind = found.get("kind")
            # Preguntas de sí/no no se responden con una hora ni con un numero: medido en el
            # bench16, "puedo dejar el equipaje despues del check-out" contestaba `10:30` porque
            # el modulo emparej6 "check-out" antes que la FAQ. Es mejor no contestar por esta via
            # y dejar que el operador siga buscando, que dar un dato cierto pero que no es la
            # respuesta.
            if kind in TEMPORAL_OR_NUMERIC_KINDS and any(m in g for m in YES_NO_MARKERS):
                alt = faq_answer_for_question(facts, goal)
                if alt and alt.get("value"):
                    alt = dict(alt)
                    alt["_from"] = url
                    alt["kind"] = "faq"
                    return alt
                continue
            found["_from"] = url
            return found
        alt = faq_answer_for_question(facts, goal)
        if alt and alt.get("value"):
            alt = dict(alt)
            alt["_from"] = url
            alt["kind"] = "faq"
            return alt
        saw_wifi = saw_wifi or bool(facts.get("wifi"))
    if saw_wifi and structured_fact_about("wifi", goal):
        # "¿clave del wifi?" El sitio publica `Free High-Speed Wi-Fi` y NUNCA la clave (medido:
        # `password` solo aparece en /login/). El módulo devuelve None a propósito para no colar
        # "Free Wi-Fi" como si fuera la contraseña, pero eso no es "no lo encontré": es "no está
        # publicado", y decirlo es la respuesta correcta para el huésped.
        return {
            "kind": "wifi-clave",
            "value": "el sitio anuncia wifi gratis pero no publica la clave",
            "source": "structured:wifi_sin_clave",
        }
    return None


def structured_fact_about(kind_word: str, goal: str) -> bool:
    """True si el objetivo pregunta por ese tipo de dato (puente es->en mediante)."""
    return any(word in concepts(goal) for word in concepts(kind_word))


# Contenido del sitio que NO habla del hotel: excursiones, galería, blog. Medido en el bench16:
# "pool" solo aparece aquí ("terraced salt pools" de Maras), y "en qué barrio está el hotel"
# respondía "San Blas Neighborhood … 1.2 km 15m walk", que es una atracción cercana y no la
# respuesta (el hotel está en San Pedro / Centro de Cusco). Valen para informar de la zona, no
# para afirmar qué publica u omite el establecimiento.
EXCURSION_PATH_WORDS = ("explore", "gallery", "blog", "news", "tour", "atraccion", "excursion")


def is_hotel_content(url: str) -> bool:
    path = (urlparse(url or "").path or "").lower()
    return not any(word in path for word in EXCURSION_PATH_WORDS)


def load_site_graph(data_dir: Path, start_url: str) -> dict[str, Any] | None:
    """Grafo recorrido del MISMO host, si existe. Nada de adivinar entre hosts."""
    path = Path(data_dir) / "site_graph.json"
    if not path.exists():
        return None
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    origin = host(start_url)
    if not isinstance(graph, dict) or not origin or host(graph.get("start_url") or "") != origin:
        return None
    return graph


def route_start_url(goal: str, graph: dict[str, Any] | None, start_url: str) -> tuple[str, int] | None:
    """Página de entrada según el objetivo, con el grafo que ya se recorrió.

    Sin esto el operador arranca siempre en la home y tiene que adivinar la barra de navegación:
    medido el 25-sep, 4 de 8 tareas informativas morían saltando entre ROOMS/SERVICES/GALLERY sin
    llegar a la página del dato. El grafo no cuesta GPU ni entrenamiento: es conocimiento del
    sitio. Se exige >=2 palabras en común y se descarta la propia home, que casaría con todo.
    """
    if not graph:
        return None
    gt = tokens(expand_goal(goal)) - GENERIC_GOAL_WORDS
    if not gt:
        return None
    start_path = urlparse(start_url).path.rstrip("/")
    candidates: list[tuple[dict[str, Any], set[str]]] = []
    for node in (graph.get("nodes") or {}).values():
        url = node.get("url") or ""
        path = urlparse(url).path.rstrip("/")
        if not path or path == start_path or urlparse(url).fragment:
            continue
        # Las páginas legales repiten los labels del menú ("Terms", "Privacy", "Cancellation") y
        # ganaban por empate a `telefono` y a `room-types` al bajar el corte: no son la respuesta
        # de ninguna pregunta del huésped.
        if any(word in path.lower() for word in ("terms", "privacy", "policy", "cookie", "legal")):
            continue
        blob = " ".join(
            [url, node.get("title") or "", node.get("text_preview") or ""]
            + [str(a.get("label") or "") for a in (node.get("actions") or [])[:40]]
        )
        candidates.append((node, tokens(blob)))

    # Lo que rompía el enrutado con corte 1 no era el número: eran los términos de navegación
    # ("book", "contact", "direct", "room"), que salen en casi todas las páginas y empataban con
    # cualquier objetivo. Se descarta lo que aparece en más de un tercio del grafo y, hecho eso,
    # un solo término discriminatorio sí vale (`bilingual` está en una sola página y es la
    # respuesta a "el personal habla otros idiomas").
    df: dict[str, int] = {}
    for _, toks in candidates:
        for tok in toks:
            df[tok] = df.get(tok, 0) + 1
    ceiling = max(2, len(candidates) // 3)

    best: tuple[int, int, str] | None = None
    # Dos niveles: primero las páginas que hablan del establecimiento. Medido con `barrio`: el
    # grafo empata "neighborhood" en /explore/ (la guía de atracciones) y mandaba alli, mientras
    # /contact/ dice "The courtyard, San Pedro" — con el corpus de hechos ya filtrado, eso
    # costaba 8 pasos perdidos en vez de la respuesta.
    for tier in (True, False):
        for node, toks in candidates:
            if is_hotel_content(node.get("url") or "") != tier:
                continue
            hits = len({t for t in gt & toks if df.get(t, 0) <= ceiling})
            if not hits:
                continue
            depth = urlparse(node.get("url") or "").path.rstrip("/").count("/")
            if best is None or (hits, depth) > best[:2]:
                best = (hits, depth, node.get("url") or "")
        if best is not None:
            break
    return (best[2], best[0]) if best else None


def fillable(action: "Action", context: dict[str, Any]) -> bool:
    """Un campo que no se puede rellenar con lo que dio el usuario no va en el menú.

    La demo perdía dos de sus ocho pasos en `input_value_required` sobre fechas y selects:
    ofrecerlos es invitar al modelo a gastarse ahí.
    """
    if action.kind != "input":
        return True
    inputs = context.get("inputs") or {}
    files = context.get("files") or {}
    keys = (action.name, action.placeholder, action.label)
    if action.input_type == "file":
        return bool(context.get("file") or any(files.get(k) for k in keys if k))
    return any(inputs.get(k) is not None for k in keys if k)


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


# Los objetivos del usuario llegan en español; las etiquetas de las páginas reales y del train
# de Mind2Web están en inglés. "desayuno" nunca casaba con "Breakfast", así que el reintento del
# modelo y el bootstrap perdían la acción correcta antes de puntuarla.
ES_EN = {
    "desayuno": "breakfast", "comida": "food meal dining", "almuerzo": "lunch",
    "precio": "price rate cost", "tarifa": "rate price", "coste": "cost price",
    "habitacion": "room", "suite": "suite room", "cama": "bed", "bano": "bathroom bath",
    "reservar": "book reserve", "reserva": "booking reservation book",
    "disponibilidad": "availability available", "noche": "night stay",
    "wifi": "wifi internet wireless", "piscina": "pool", "gimnasio": "gym fitness",
    "estacionamiento": "parking", "aparcamiento": "parking", "mascota": "pet",
    "entrada": "checkin arrival entrance entry", "salida": "checkout departure exit",
    "checkin": "checkin arrival", "checkout": "checkout departure",
    "hora": "time schedule", "horario": "time schedule opening hours",
    "ubicacion": "location address map directions", "direccion": "address directions",
    "contacto": "contact", "correo": "mail email", "telefono": "phone tel call",
    "buscar": "search find", "ver": "view see show", "mas": "more",
    "informacion": "information info about", "detalles": "details detail",
    "enviar": "submit send", "confirmar": "confirm submit", "cancelar": "cancel",
    "cuenta": "account", "cliente": "customer client guest", "huesped": "guest",
    "tour": "tour", "visita": "visit tour", "evento": "event", "sala": "room hall",
    "aforo": "capacity guests", "gente": "guests people", "persona": "guests people",
    "niio": "child kids", "aire": "air", "desplegar": "show expand open",
    # Añadido tras el bench de 16 objetivos (25-sep): sin estas entradas el operador declaraba
    # "no publicado" para oxigeno/aeropuerto/gastronomia/equipaje/barrio, y los cinco datos SI
    # están en el sitio (comprobado en las 1556 trazas). Un puente incompleto fabrica ausencias.
    "oxigeno": "oxygen", "aeropuerto": "airport", "gastronomia": "restaurant dining cuisine cafeteria",
    "restaurante": "restaurant dining cafe cafeteria coffee", "cocina": "kitchen cuisine",
    "equipaje": "luggage baggage", "maleta": "luggage suitcase",
    "idioma": "language languages bilingual", "barrio": "neighborhood district courtyard",
    "centro": "center downtown", "ascensor": "elevator lift", "adaptador": "adapter",
    "traslado": "transfer shuttle", "azotea": "rooftop terrace", "terraza": "terrace rooftop",
    # "altura" en los Andes es la enfermedad de la altura: la FAQ del hotel dice
    # "How can I prevent altitude sickness (soroche)?" y sin estos puentes la pregunta de
    # "hay oxigeno para la altura" no casaba con ninguna FAQ.
    "altura": "altitude soroche sickness", "soroche": "altitude sickness",
    # `doble` -> `double` era el hueco que empataba dos tarjetas de /rooms/ que comparten
    # "Superior Room": solo "Double Superior Room" distingue la una de la otra, y sin esa
    # traduccion el desempate por cercania elegia el precio de la tarjeta vecina.
    # "matrimonial" se queda fuera a proposito: el sitio la usa como nombre propio
    # ("Superior Matrimonial Room", tarjeta distinta con precio propio), y traducirla a
    # "double" haria que preguntar por matrimonial conteste con el precio de otra habitacion.
    "doble": "double",
    "piso": "floor story", "mascotas": "pet pets",
}


def expand_goal(goal: str) -> str:
    """Añade al objetivo los equivalentes ingleses de las palabras reconocidas.

    Casa por forma morfológica (la palabra, o su singular/plural), NUNCA por prefijo: con
    `stem.startswith(word[:5])` el token `personal` casaba con el stem `persona` y arrastraba
    "guests people" a objetivos como "el personal habla otros idiomas" — medido: esa era la
    razón por la que ganaba una frase de GUEST REVIEWS. El prefijo también quedó detrás del
    `'directions'[:6] == 'direct'` de `concepts()`.
    """
    forms: set[str] = set()
    for stem in ES_EN:
        forms.add(stem)
        if len(stem) > 4 and stem.endswith("s"):
            forms.add(stem[:-1])
    lookup: dict[str, list[str]] = {}
    for stem, english in ES_EN.items():
        lookup.setdefault(stem, []).append(english)
        if len(stem) > 4 and stem.endswith("s"):
            lookup.setdefault(stem[:-1], []).append(english)
    extra: list[str] = []
    for word in tokens(goal):
        if len(word) < 4:
            continue
        # Se buscan las dos direcciones de la morfología: la palabra tal cual, y su singular si
        # viene en plural. Sin esto, "idiomas" no encontraba el stem "idioma" y se quedaba sin
        # traducir (lo pilló el test, no una corrida de benchmark).
        for form in (word, word[:-1] if word.endswith("s") and len(word) > 4 else word,
                     word[:-2] if word.endswith("es") and len(word) > 5 else word):
            if form in forms:
                extra.extend(lookup.get(form, []))
        # Y se permite el prefijo SOLO en el sentido seguro: el token del objetivo es prefijo del
        # stem. Hace falta porque "check-in" se trocea en "check", que debe llegar a "checkin"/
        # "checkout". La dirección contraria (stem prefijo del token) es la prohibida: era lo que
        # hacía que "personal" arrastrara "guests people" (medido 04:41, con el corte estricto
        # `check-in` y `desayuno` perdían su traducción y el enrutado se caía a None).
        if len(word) >= 5:
            for stem in ES_EN:
                if stem != word and stem.startswith(word):
                    extra.extend(lookup.get(stem, []))
    return (goal or "") + " " + " ".join(extra)


def rank_actions(goal: str, actions: list["Action"], cap: int = 10) -> list["Action"]:
    """Menú recortado para la política.

    El train tiene estados de 2-10 acciones y una página real expone 20-40: servir las 40
    desborda el contexto de 2048 (lo que reabre el truncado por la derecha) y saca al modelo
    de su distribución. Se puntúa por solape con el objetivo (expandido a inglés), se favorecen
    los campos editables y se devuelve el top en orden de DOM, que es el orden que vio en el
    train.
    """
    if len(actions) <= cap:
        return list(actions)
    goal_tokens = concepts(expand_goal(goal))
    scored: list[tuple[float, int, "Action"]] = []
    for position, action in enumerate(actions):
        text = " ".join((action.label, action.name, action.placeholder, action.href, action.role))
        overlap = float(len(goal_tokens & concepts(text)))
        prior = 0.9 if action.kind == "input" else (0.4 if action.role in {"button", "link", "tab"} else 0.1)
        if not clean(action.label):
            prior -= 0.5
        if len(action.label) > 60:
            prior -= 0.3
        scored.append((overlap * 3.0 + prior, position, action))
    scored.sort(key=lambda item: (-item[0], item[1]))
    keep = sorted(scored[:cap], key=lambda item: item[1])
    return [item[2] for item in keep]


def host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def allowed_url(url: str, allowed_hosts: set[str]) -> bool:
    current = host(url)
    return bool(current) and any(current == item or current.endswith("." + item) for item in allowed_hosts)


def page_key(observation: "Observation") -> str:
    """Clave de "página actual" para las acciones ya gastadas.

    El state_id es hash del texto visible, y en una SPA el texto cambia en cada click, así que
    la misma acción vuelve a parecer novelty y el operador se queda clicando ROOMS en bucle.
    """
    return urlparse(observation.url)._replace(fragment="").geturl()


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

    def settle(self, timeout_ms: int = 2500) -> None:
        """Da tiempo a que la SPA pinte, sin poder colgar el paso.

        Medido en usgarhoteles.com: el nodo `/book/` del grafo se capturó en estado "LOADING
        VIDEO…" y el listado de amenidades ("03 Cafeteria Nocturna UNTIL 22:00 PM", donde está la
        respuesta a "que restaurante tiene el hotel") no aparecía en el texto observado. Con
        `wait_until="networkidle"` a secas una página con analytics/websockets nunca llegaría a
        idle y el paso se caería, así que se espera acotado y se ignora el timeout.
        """
        assert self.page is not None
        try:
            self.page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            pass
        self.page.wait_for_timeout(250)

    def goto(self, url: str) -> None:
        if not allowed_url(url, self.allowed_hosts):
            raise ValueError("host no permitido: " + host(url))
        assert self.page is not None
        self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        self.settle()

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
                // Los skip-links y textos para lectores viven fuera del viewport (left:-9999px,
                // clip:rect(0,0,0,0)) o son inertes: clicarlos gasta un paso y no navega a nada.
                if (rect.right <= 0 || rect.bottom <= 0 || rect.left + rect.width <= 0 || rect.top + rect.height <= 0) continue;
                if (style.opacity === '0' || style.pointerEvents === 'none') continue;
                if (style.clip && style.clip.replace(/\s/g, '').indexOf('rect(0,0,0,0') === 0) continue;
                if (element.closest('[aria-hidden="true"], [hidden], [inert]')) continue;
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
            href = item.get("href", "")
            if href.startswith("http") and not allowed_url(href, self.allowed_hosts):
                continue
            signal = " ".join((item.get("label", ""), item.get("name", ""), item.get("placeholder", ""))).lower()
            item["dangerous"] = bool(tokens(signal) & DANGEROUS_WORDS)
            if href.startswith("http") and any(word in urlparse(href).path.lower() for word in WRITE_PATH_WORDS):
                item["dangerous"] = True
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
                try:
                    locator.click(timeout=10000)
                except PlaywrightTimeoutError:
                    # Widget que se re-renderiza sin estabilizarse: click() espera a que sea
                    # accionable y agota el timeout; force lo dispara igualmente.
                    locator.click(timeout=5000, force=True)
            self.settle(timeout_ms=2000)
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
                # 4000 en vez de 1000: el enrutador solo ve lo que guarda el grafo, y con 1000
                # caracteres estaba ciego a lo que vive a mitad de página (medido con
                # "que restaurante tiene el hotel": los nombres de las cafeterías están en /book/
                # fuera de los primeros 1000 caracteres, así que no había ruta posible).
                "text_preview": observation.text[:4000],
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


def corpus_term_evidence(terms, data_dir: Path, start_url: str) -> tuple[int, int]:
    """(fragmentos que contienen el termino, documentos cacheados examinados) en TODO el sitio.

    Se declara una ausencia despues de recorrer 5 paginas, pero el disco suele tener mucho mas
    recorrido ya: medido en usgarhoteles.com, "pool"/"piscina" aparece 0 veces en las 1105 paginas y
    observaciones cacheadas. Consultar el corpus cuesta milisegundos y da una afirmacion mucho mas
    fuerte que la del paseo actual, sin gastar un solo paso de navegacion.
    """
    docs = 0
    hits = 0
    textos: list[str] = []
    grafo = load_site_graph(data_dir, start_url) or {}
    for nodo in (grafo.get("nodes") or {}).values():
        textos.append(" ".join(str(nodo.get(c) or "") for c in ("url", "title", "text_preview")))
    trazas = data_dir / "traces.jsonl"
    if trazas.exists():
        for linea in trazas.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                registro = json.loads(linea)
            except ValueError:
                continue
            if registro.get("type") == "observation":
                textos.append(str(registro.get("text") or ""))
    patrones = [re.compile(r"(?<![a-z0-9])" + re.escape(str(t)) + r"(?![a-z0-9])", re.I)
                for t in set(terms or [])]
    for texto in textos:
        docs += 1
        compacto = texto.replace("-", "")
        if any(p.search(compacto) for p in patrones):
            hits += 1
    return hits, docs


def stamp_evidence(answer: dict[str, Any] | None,
                   pages: list[tuple[str, str]]) -> dict[str, Any] | None:
    """Deja constancia de en que pagina esta literalmente la evidencia de la respuesta.

    Sin esto, "cero respuestas inventadas" es una afirmacion que solo puede comprobar un humano
    leyendo el log: el valor extractivo no llevaba de donde salio. Aqui se exige que el texto
    devuelto aparezca tal cual (normalizando espacios) en el innerText de una pagina recorrida.
    Si no aparece en ninguna, `source_url` queda None y el benchmark cuenta la respuesta como
    procedencia no verificada, que es exactamente el caso que hay que ver y no esconder.
    """
    if not answer or not answer.get("value"):
        return answer
    needle = re.sub(r"\s+", " ", str(answer["value"])).strip().lower()
    for url, text in reversed(pages):
        if needle and needle in re.sub(r"\s+", " ", text or "").lower():
            answer["source_url"] = url
            answer["verbatim"] = True
            return answer
    answer["source_url"] = None
    answer["verbatim"] = False
    return answer


class UniversalOperator:
    def __init__(self, start_url: str, data_dir: Path, max_steps: int = 12,
                 allowed_hosts: Iterable[str] | None = None,
                 policy: ModelDecisionPolicy | None = None, menu_cap: int = 10):
        self.start_url = start_url
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.allowed_hosts = allowed_hosts
        self.policy = policy
        self.menu_cap = menu_cap

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
                and (page_key(observation), action.signature) not in used_actions
            ]
            if file_actions:
                return file_actions[0]
        goal_tokens = concepts(expand_goal(goal))
        best: tuple[float, Action] | None = None
        for action in observation.actions:
            if (page_key(observation), action.signature) in used_actions:
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
        failures = 0
        used_actions: set[tuple[str, str]] = set()
        visited: dict[str, int] = {}
        banned: set[str] = set()
        last_text = ""
        last_url = ""
        all_text = ""
        own_text = ""
        evidence_pages: list[tuple[str, str]] = []
        pages_seen: list[str] = []
        route = route_start_url(goal, load_site_graph(self.data_dir, self.start_url), self.start_url)
        entry_url = route[0] if route else self.start_url
        # Raíz + página enrutada + unas pocas páginas del grafo: el `<select name=room-type>` que
        # resuelve "tipos de habitación" está en /rooms/, y el JSON-LD en la raíz.
        graph = load_site_graph(self.data_dir, self.start_url) or {}
        candidates = [self.start_url, entry_url] + [
            n.get("url") for n in list((graph.get("nodes") or {}).values())[:4]
        ]
        seen_urls: list[str] = []
        for url in candidates:
            if url and url not in seen_urls:
                seen_urls.append(url)
        shortcut = structured_shortcut(goal, seen_urls[:6])
        if shortcut and shortcut.get("value"):
            result = {
                "status": "answered",
                "answer": {"kind": shortcut.get("kind"), "value": shortcut.get("value"),
                           "context": str(shortcut.get("source") or ""),
                           # Un dato estructurado no es una substring literal del innerText (la
                           # direccion viene de JSON-LD, parts sueltas), asi que su procedencia es
                           # el campo y la pagina de donde se leyeron, no `verbatim`.
                           "source_url": str(shortcut.get("_from") or self.start_url),
                           "source_field": str(shortcut.get("source") or ""),
                           "verbatim": False},
                "reason": "datos_estructurados_del_sitio",
                "url": entry_url,
                "steps": [],
                "structured": True,
            }
            JsonStore(self.data_dir).append("operations.jsonl", result)
            return result
        with BrowserSession(entry_url, self.data_dir, self.allowed_hosts,
                            trace_name="operation-" + str(int(time.time()))) as session:
            session.goto(entry_url)
            for index in range(self.max_steps):
                observation = session.observe()
                last_text = observation.text
                last_url = observation.url
                all_text = (all_text + " " + observation.text)[-40000:]
                if is_hotel_content(observation.url):
                    own_text = (own_text + " " + observation.text)[-40000:]
                    evidence_pages.append((observation.url, observation.text))
                pages_seen.append(observation.url)
                state_key = page_key(observation)
                visited[state_key] = visited.get(state_key, 0) + 1
                if visited[state_key] >= 2 and is_hotel_content(observation.url):
                    answer = stamp_evidence(pick_answer(goal, observation.text), evidence_pages)
                    if answer:
                        result = {
                            "status": "answered",
                            "answer": answer,
                            "reason": "estado_repetido_con_respuesta",
                            "url": observation.url,
                            "steps": steps,
                            "evidence": observation.text[:3000],
                        }
                        JsonStore(self.data_dir).append("operations.jsonl", result)
                        return result
                # La política no ve las 20-40 acciones de una página real: ve un top del tamaño
                # del que había en el train (2-10). Los índices del modelo se interpretan contra
                # este menú, no contra observation.actions.
                available = [
                    item for item in observation.actions
                    if item.enabled and fillable(item, context)
                    and item.signature not in banned
                    and (page_key(observation), item.signature) not in used_actions
                ]
                menu = rank_actions(goal, available, self.menu_cap)
                menu_observation = replace(observation, actions=menu)
                policy_decision: dict[str, Any] | None = None
                if self.policy is not None:
                    try:
                        policy_decision = self.policy.decide(
                            goal, context, menu_observation, history,
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
                    # El guard está pensado en envíos (form/uploads). En una tarea informativa
                    # el modelo sí ve la respuesta en la página sin que aparezca ninguna de esas
                    # palabras: devolverla marcada como sin verificar es mejor que tirar la
                    # evidencia y acabar en "blocked".
                    result = {
                        "status": "finished_unverified",
                        "url": observation.url,
                        "steps": steps,
                        "evidence": observation.text[:3000],
                        "policy": policy_decision,
                    }
                    JsonStore(self.data_dir).append("operations.jsonl", result)
                    return result
                if policy_decision and policy_decision.get("decision") in {"act", "request_confirmation"}:
                    action_index = policy_decision.get("action_index")
                    if isinstance(action_index, int) and 0 <= action_index < len(menu):
                        candidate = menu[action_index]
                        file_already_uploaded = candidate.kind == "input" and candidate.input_type == "file" and uploaded
                        same_page_link = candidate.kind == "click" and candidate.href and candidate.href == observation.url
                        if not file_already_uploaded and not same_page_link:
                            action = candidate
                        elif self.policy is not None:
                            # El modelo ha elegido una acción estéril (enlace a la propia página o
                            # file ya subido): se le vuelve a preguntar sin esa acción, para que el
                            # "siguiente mejor" salga del modelo y no del emparejamiento por tokens
                            # del bootstrap (que es monolingue).
                            remaining = rank_actions(
                                goal, [item for item in menu if item is not candidate], self.menu_cap
                            )
                            if remaining:
                                retry = self.policy.decide(
                                    goal, context, replace(observation, actions=remaining), history,
                                    write_confirmed=bool(confirm and allow_writes),
                                )
                                if retry and retry.get("decision") in {"act", "request_confirmation"}:
                                    index2 = retry.get("action_index")
                                    if isinstance(index2, int) and 0 <= index2 < len(remaining):
                                        action = remaining[index2]
                                        retry["_attempt"] = "second_best"
                                        policy_decision = retry
                if action is None:
                    action = self.choose(goal, menu_observation, context, uploaded, used_actions)
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
                    "goal": goal,
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
                used_actions.add((page_key(observation), action.signature))
                if not act_result.get("ok"):
                    # Un <select> o un input sin valor en el contexto no es un fallo del sitio:
                    # se marca como gastada y se reintenta con otra cosa. Sin archivo local si.
                    fatal = act_result.get("reason") in {"disabled", "file_required"}
                    failures += 1
                    if fatal or failures >= 3:
                        result = {"status": "error", "error": act_result, "consecutive_failures": failures, "steps": steps}
                        JsonStore(self.data_dir).append("operations.jsonl", result)
                        return result
                    # El siguiente paso re-observa el mismo estado: como la acción fallada ya
                    # está en used_actions, eso es exactamente el reintento con la siguiente
                    # mejor candidata.
                    continue
                failures = 0
                if action.kind == "input" and action.input_type == "file":
                    uploaded = True
                after = session.observe()
                if action.kind == "click" and page_key(after) in visited:
                    # El clic lleva a un estado ya visto: es un enlace de la barra de navegación
                    # que devuelve al mismo sitio. Sin vetarlo, las tareas que no tienen literal
                    # en pantalla gastan los 8 pasos en ROOMS/SERVICES/GALLERY (medido en la
                    # demo del 25-sep: 4 de 8 tareas acababan en budget_exhausted así).
                    banned.add(action.signature)
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
            answer = stamp_evidence(
                (pick_answer(goal, last_text) if is_hotel_content(last_url) else None)
                or pick_answer(goal, own_text), evidence_pages)
            result: dict[str, Any] = {
                "status": "budget_exhausted",
                "steps": steps,
                "evidence": last_text[:3000],
                "pages_visited": pages_seen,
            }
            if answer:
                result["answer"] = answer
                result["status"] = "answered"
                result["reason"] = "respuesta_en_alguna_pagina_visitada"
            else:
                terms = goal_terms(goal)
                # Conservador por diseño: solo se declara "no encontrado en esta encuesta" si el
                # término no aparece en NINGUNA parte de lo recorrido ni del grafo. Contar
                # "contextos distintos" para descartar el relleno repetido (el `salt pools` de las
                # excursiones) suena mejor pero convierte una mención única legítima — un hotel que
                # nombra la piscina una vez en amenities — en un falso "aquí no está publicado", y
                # afirmar ausencias es peor que gastar pasos. Piscina queda sin resolver a
                # propósito; la salida buena es un dato estructurado, no un heurístico de strings.
                hay = re.sub(r"[^a-z0-9]+", "", (own_text + " " + " ".join(
                    str((n.get("url"), n.get("title"), n.get("text_preview")))
                    for n in ((load_site_graph(self.data_dir, self.start_url) or {}).get("nodes") or {}).values()
                    if is_hotel_content(n.get("url") or "")
                )).lower())
                if terms and not any(term.replace("-", "") in hay for term in terms):
                    # El bench de 16 objetivos (25-sep) pilló este bug: con el puente es->en sin
                    # "oxigeno"/"aeropuerto"/"gastronomia", los terminos buscados eran solo
                    # espanoles, no casaban con un sitio en ingles, y el operador afirmaba
                    # "aqui no esta publicado" cuando el dato SI estaba (oxygen/airport/restaurant
                    # presentes en las 1556 trazas). Sin cobertura del puente no se puede afirmar
                    # ausencia: solo que no se encontró.
                    english = {w for value in ES_EN.values() for w in value.split()}
                    bridged = any(term in english for term in terms)
                    result["status"] = "not_found_in_survey" if bridged else "no_encontrado"
                    result["reason"] = ("termino_ausente_en_paginas_recorridas_y_en_el_grafo"
                                        if bridged else "puente_es_en_sin_cobertura_para_el_termino")
                    result["bridge_covered"] = bridged
                    result["searched_terms"] = terms
                    result["pages_surveyed"] = sorted(set(pages_seen))
                    # El alcance del recorrido es parte de la afirmacion: "no esta publicado"
                    # después de 2 paginas no vale lo mismo que despues de 8 y 40.000 caracteres.
                    result["survey_pages"] = len(set(pages_seen))
                    result["survey_chars"] = len(own_text)
                    # Y se apoya en lo que ya esta en disco: recorrer el sitio cuesta pasos, consultar
                    # el corpus cacheado no. Si el termino tampoco aparece ahi, la ausencia deja de ser
                    # "no lo vi en estas paginas" y pasa a ser "el sitio no lo publica".
                    if bridged:
                        hits, docs = corpus_term_evidence(terms, self.data_dir, self.start_url)
                        result["corpus_docs_checked"] = docs
                        result["corpus_term_hits"] = hits
                        if hits == 0 and docs:
                            result["reason"] = ("termino_no_publicado_en_el_sitio_"
                                                "ausente_en_corpus_completo")
            JsonStore(self.data_dir).append("operations.jsonl", result)
            return result
