"""Prueba sin GPU del menú recortado: el objetivo en español debe traer la acción inglesa."""
import sys
from dataclasses import dataclass, field


@dataclass
class Action:
    action_id: str = ""
    kind: str = "click"
    label: str = ""
    role: str = "link"
    input_type: str = ""
    name: str = ""
    placeholder: str = ""
    href: str = ""
    dangerous: bool = False
    enabled: bool = True
    signature: str = field(default="")


from site2tools.core import expand_goal, rank_actions, concepts, tokens

# Enunciado real de la demo del hotel: desayuno en inglés, pregunta en español.
menu = [Action(action_id="s2t-%d" % i, label=l, role="link") for i, l in enumerate([
    "Home", "Rooms", "Gallery", "Location", "Contact", "Booking.com", "Facebook", "Instagram",
    "Double Superior Room", "Deluxe Room", "Standard Room", "Suite", "Amenities", "Services",
    "Restaurant", "Bar", "Breakfast", "Pool", "Gym", "Spa", "Parking", "WiFi", "Events",
    "Weddings", "Conference", "Terms", "Privacy", "Cancellation", "FAQ", "Newsletter",
])]
forbidden = {"Booking.com", "Facebook", "Instagram", "Privacy", "Terms", "Cancellation"}

ranked = rank_actions("¿A qué hora es el desayuno?", menu, 10)
labels = [a.label for a in ranked]
print("goal expandido ->", sorted(concepts(expand_goal("¿A qué hora es el desayuno?")) & {"breakfast", "time", "hours"}))
print("menu(10) =", labels)
assert "Breakfast" in labels, "falla el puente es->en: Breakfast no llega al menu"

# Un objetivo de precio debe priorizar tarifas/rooms sobre la navegacion generica.
priced = [a.label for a in rank_actions("precio de la habitacion Double Superior", menu, 10)]
assert "Double Superior Room" in priced, priced
print("menu precio =", priced)

# Regresion del presupuesto: con 40 acciones el payload JSON no debe desbordar 2048 tokens.
big = [Action(action_id="s2t-%d" % i, label="Option %d description text" % i, role="link") for i in range(40)]
print("cap aplicado:", len(rank_actions("abrir caso", big, 10)), "de", len(big))
assert len(rank_actions("abrir caso", big, 10)) == 10

# Orden de DOM preservado dentro del top (los indices del modelo se leen contra este orden).
dom = [Action(action_id="s2t-%d" % i, label="noise %d" % i) for i in range(12)]
dom[3].label = "breakfast menu"; dom[9].label = "breakfast hours"
kept = [a.action_id for a in rank_actions("hora del desayuno", dom, 4)]
assert kept == sorted(kept, key=lambda s: int(s.split("-")[1])), kept
print("orden DOM ok:", kept)

# --- pick_answer: un objetivo de hora nunca se responde con un precio ---
from site2tools.core import pick_answer

PAGINA_HABITACIONES = ("All Rooms 2 Guests 8 GUESTS Family Superior Room FROM $150 PER NIGHT "
                       "BREAKFAST INCLUDED BOOK YOUR STAY")
assert pick_answer("A que hora se sirve el desayuno", PAGINA_HABITACIONES) is None, \
    "devolvió un precio donde se pedía una hora"
assert pick_answer("Cual es el precio por noche", PAGINA_HABITACIONES)["value"] == "$150"
CON_HORAS = PAGINA_HABITACIONES + " Daily Buffet Breakfast 06:00 AM - 09:00 AM"
got = pick_answer("A que hora se sirve el desayuno", CON_HORAS)
assert got["kind"] == "hora" and got["value"] == "06:00 AM", got

# Sin patrón de literal (piscina/direccion) la extractiva sí puede hablar.
assert pick_answer("Tiene piscina el hotel", "Our facilities include an outdoor Swimming Pool and a Fitness Center.") is not None
print("pick_answer ok: hora no se confunde con precio")

# El title+nav del sitio real no puede pasar por respuesta de `direccion`.
TITLE_NAV = ("Skip to main content ROOMS SERVICES EXPLORE CUSCO GALLERY CONTACT EN Sign In Book Now "
             "LOADING VIDEO USGAR Hotels | San Pedro, Cusco - Your gateway to the Andes Book Your "
             "Stay Explore Rooms CHECK-IN - CHECK-OUT Sep 26 Sep 29 2 Guests Direct Booking Best Rate")
assert pick_answer("Cual es la direccion del hotel", TITLE_NAV) is None, "el title contesto la direccion"
print("title no responde direccion: ok")

# Texto literal de /contact/ guardado en las trazas del 25-sep: la dirección se publica así.
CONTACTO = ("Skip to main content ROOMS SERVICES EXPLORE CUSCO GALLERY CONTACT EN Sign In Book Now "
            "759 CALLE HOSPITAL, CUSCO Contact Us & Plan Your Trip We would love to hear from you. "
            "WhatsApp Direct +51992559943 Call Front Desk +51 992 559 943 The courtyard, San Pedro "
            "CONTACT 759 Calle Hospital, Centro de Cusco, Cusco info@usgarhoteles.com")
got = pick_answer("Cual es la direccion del hotel", CONTACTO)
assert got and got["kind"] == "direccion" and "CALLE HOSPITAL" in got["value"].upper(), got
print("direccion leida de /contact/:", got["value"])

# --- horas ancladas a su palabra, con señuelo delante (texto real de /book/) ---
BOOK = ("Sale ends at 6:00 today. Complimentary Coca/Muna tea Free High-Speed Wi-Fi 24-Hour "
        "Front Desk Assistance Check-in: 12:00 hrs | Check-out: 10:30 hrs GUARANTEED BOOKING "
        "Free cancellation up to 48 hours before arrival.")
got = pick_answer("A que hora es el check-in", BOOK)
assert got and got["value"].startswith("12:00"), got
print("check-in anclado:", got["value"], "|", got["context"][:70])

# --- "Wi-Fi" con guion: tokens() pide 3+ caracteres y 'wifi' desaparecía ---
got = pick_answer("Cual es la clave del wifi", BOOK)
assert got is not None and "wi" in str(got["value"]).lower().replace("-", ""), got
print("wifi leido pese al guion:", str(got["value"])[:90])

# --- la extractiva no puede devolver la navegación por delante ---
NAVLARGA = ("Skip to main content " + "ROOMS SERVICES EXPLORE CUSCO GALLERY CONTACT EN Sign In Book Now " * 6
            + "ACCOMMODATIONS Our Rooms Each room is uniquely designed with hand-painted Andean murals. ")
got = pick_answer("Que tipos de habitacion hay", NAVLARGA)
assert got is not None and not got["value"].startswith("Skip to main content"), got
print("extractiva sin navegacion:", got["value"][:90])

# --- cobertura del puente: sin traduccion no se puede afirmar ausencia ---
from site2tools.core import goal_terms

assert "oxygen" in goal_terms("Hay oxigeno para la altura"), goal_terms("Hay oxigeno para la altura")
assert "airport" in goal_terms("Como llego desde el aeropuerto"), goal_terms("Como llego desde el aeropuerto")
assert "restaurant" in goal_terms("Que restaurante tiene el hotel")
assert "luggage" in goal_terms("Puedo dejar el equipaje despues del check-out")
print("puente cubre oxigeno/aeropuerto/restaurante/equipaje: OK")

# --- el puente no debe arrastrar comodines por prefijo ---
from site2tools.core import expand_goal

exp = expand_goal("El personal habla otros idiomas").lower()
assert "language" in exp and "bilingual" in exp, exp
assert "guests" not in exp and "people" not in exp, \
    "personal casa con el stem 'persona' y mete guests/people: " + exp
print("puente sin contaminacion por prefijo: OK")

# --- una reseña citada no es un dato del hotel ---
RESENA = ("Skip to main content ROOMS SERVICES GALLERY CONTACT BOOKING.COM reviews below. "
          'Guests say "It is a nice older building with a walk-around courtyard and the staff '
          'speaks multiple languages." End of reviews section here.')
assert pick_answer("En que barrio esta el hotel", RESENA) is None, \
    "la reseña de un tercero contesto el barrio"
print("resena no contestada: OK")

# --- tabla de enrutado medida en el grafo real del hotel (25-sep 03:00 UTC) ---
# Sirve de regresión porque el `expand_goal` nuevo y el corte del enrutador ya rompieron esta
# tabla una vez (04:41: `check-in` y `desayuno` se cayeron a None al perder la traducción).
import json
from pathlib import Path
from urllib.parse import urlparse
from site2tools.core import route_start_url, GENERIC_GOAL_WORDS

graph_path = next((p for p in ("data/site_graph.json", "data-local/site_graph.json")
                   if Path(p).exists()), None)
if graph_path is None:
    print("AVISO: sin grafo del hotel, se omite la tabla de enrutado")
else:
    graph = json.load(open(graph_path, encoding="utf-8"))
    GOALS = ["Cual es la direccion del hotel", "Cual es el telefono de contacto",
             "A que hora es el check-in",
             "Cual es el precio por noche de la habitacion Doble Superior"]
    for goal in GOALS:
        got = route_start_url(goal, graph, "https://usgarhoteles.com/")
        if got is None:
            print("  ruta %-52s -> None (el operador arranca en la home)" % goal[:50])
            continue
        node = next(n for n in graph["nodes"].values() if n["url"] == got[0])
        other = []
        for n in graph["nodes"].values():
            p = urlparse(n.get("url") or "").path.rstrip("/")
            if not p or p == "/" or n.get("url") == got[0]:
                continue
            other.append(tokens(" ".join([n.get("url") or "", n.get("title") or "",
                                          n.get("text_preview") or ""])))
        toks = tokens(" ".join([node.get("url") or "", node.get("title") or "",
                                node.get("text_preview") or ""]))
        shared = (tokens(expand_goal(goal)) - GENERIC_GOAL_WORDS) & toks
        # Invariante: si enruta, al menos un termino compartido tiene que ser DISCRIMINATORIO,
        # o sea no puede aparecer en mas de un tercio de las paginas del grafo. Es la condicion
        # que hace legitimo el corte en 1; ">=2 terminos" era una proxy peor de lo mismo.
        ceiling = max(2, len(other) // 3)
        distinctive = {t for t in shared if sum(1 for b in other if t in b) <= ceiling}
        assert distinctive, (goal, got[0], sorted(shared))
        print("  ruta %-52s -> %s (unicos: %s)" % (goal[:50], urlparse(got[0]).path, sorted(distinctive)[:3]))
    assert "guests" not in expand_goal("El personal habla otros idiomas").lower()

# --- el contenido de excursiones no habla por el hotel ---
from site2tools.core import is_hotel_content

assert not is_hotel_content("https://x.com/explore/"), "excursiones contadas como dato del hotel"
assert not is_hotel_content("https://x.com/gallery/")
assert is_hotel_content("https://x.com/contact/") and is_hotel_content("https://x.com/rooms/")
print("excursiones excluidas del corpus de hechos: OK")

# --- bloques enormes sin puntuacion (SPA): se ventana, no se rechazan ---
SIN_PUNTOS = ("Skip to main content ROOMS SERVICES EXPLORE GALLERY " + "amenities list " * 40
              + "03 Evening Cafeteria UNTIL 22:00 PM SERVING LOCAL PRODUCTS " + "tail " * 120)
got = pick_answer("Que restaurante tiene el hotel", SIN_PUNTOS)
assert got and "cafeteria" in got["value"].lower(), got
assert len(got["value"]) <= 340, len(got["value"])
print("bloque sin puntos ventana el dato: OK")

# --- precio anclado a la habitacion preguntada, no a la primera moneda ---
DOBLE_PRIMERO = ("01 / 04 Double Superior Room FROM $90 PER NIGHT BREAKFAST INCLUDED "
                 "02 / 04 Superior Matrimonial Room FROM $75 PER NIGHT BOOK YOUR STAY")
OTRA_PRIMERO = ("Superior Matrimonial Room FROM $75 PER NIGHT then further down "
                "Double Superior Room FROM $90 PER NIGHT Breakfast included")
for nombre, pagina in (("dobles primero", DOBLE_PRIMERO), ("otra primero", OTRA_PRIMERO)):
    got = pick_answer("Cual es el precio por noche de la habitacion Doble Superior", pagina)
    assert got and got["value"] == "$90", (nombre, got)
print("precio anclado a la habitacion preguntada: OK")

# --- costura del carrusel: cola de resena + ticker + dato propio pegados (texto real) ---
# innerText del sitio entrega esto como UN chunk de ~290 caracteres: la ultima frase de una
# resena, el ticker de destinos separado por ✦ y, pegado despues del ultimo glifo, el nombre
# propio del restaurante. Antes respondia con la frase del huesped.
COSTURA = ('The service provided is exceptional." T Team MEXICO / 2026 ✦ San Pedro ✦ Cusco '
           '✦ Machu Picchu ✦ Sacred Valley ✦ CULINARY & COFFEE 04 AUKA RESTOBAR An unforgettable '
           'culinary journey with fresh Andean ingredients in the historic heart of Cusco.')
assert len(COSTURA) < 320, len(COSTURA)
got = pick_answer("Que restaurante tiene el hotel", COSTURA)
assert got and "auka" in got["value"].lower(), got
assert "exceptional" not in got["value"].lower(), got
print("costura de resena separada del dato propio: OK")

# --- y la marca que descalifica puede estar a dos chunks de la cita ---
# El splitter parte por ". " asi que "We had to leave at 4am..." llega como chunk limpio, sin
# Booking.com dentro; con el descarte solo por chunk se colaba como horario del desayuno.
CITA_LEJOS = ('BOOKING.COM "The hospitality of the staff. The cleanliness and warmth of the room. '
              'We had to leave at 4am for our tour - they packed breakfast for all of us. '
              'That was so impressive." C Chul BRAZIL / 2026')
assert pick_answer("A que hora se sirve el desayuno", CITA_LEJOS) is None, CITA_LEJOS
print("cita cuya marca cae fuera del chunk tambien se descarta: OK")

# --- interrogativos: "que" y "tiene" no son informacion ---
# Medido en el diagnostico de `gastronomia`: los terminos del objetivo salian
# ['cafe','cafeteria','coffee','dining','que','restaurant','restaurante','tiene']. Con "que"
# contando como solape, cualquier frase con un "que" era evidencia valida y podia ganar el ranking.
gt = goal_terms("Que restaurante tiene el hotel")
assert "que" not in gt and "tiene" not in gt, gt
assert "restaurant" in gt, gt
RUIDO = ("Esto es una frase cualquiera que habla de otra cosa totalmente distinta, sin ningun dato "
         "relevante para la pregunta que trae el visitante del hotel.")
assert pick_answer("Que restaurante tiene el hotel", RUIDO) is None, RUIDO
print("interrogativos fuera del solape de evidencia: OK")
print("TODO OK")
