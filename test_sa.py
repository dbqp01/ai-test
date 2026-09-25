import json
import sys
import urllib.request

sys.path.insert(0, "/root/sa_test")
import structured_answers as sa

pages = ["https://usgarhoteles.com/", "https://usgarhoteles.com/rooms/"]
facts = {}
for url in pages:
    try:
        req = urllib.request.Request(url, headers={"user-agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")
    except Exception as exc:
        print(f"{url}: {type(exc).__name__} {exc}")
        continue
    got = sa.extract(html)
    print(f"--- {url}: {len(got)} clases de hecho")
    for kind, entries in got.items():
        print(f"  {kind}: {entries[0]['value'][:90]}  [{entries[0]['source'][:40]}]")
    for kind, entries in got.items():
        facts.setdefault(kind, []).extend(
            e for e in entries if all(e["value"] != f["value"] for f in facts[kind])
        )

GOALS = [
    ("precio-doble-superior", "Cual es el precio por noche de la habitacion Doble Superior"),
    ("desayuno-hora", "A que hora se sirve el desayuno"),
    ("checkin-hora", "A que hora es el check-in"),
    ("direccion", "Cual es la direccion del hotel"),
    ("telefono", "Cual es el telefono de contacto"),
    ("wifi-clave", "Cual es la clave del wifi"),
    ("piscina", "Tiene piscina el hotel"),
    ("room-types", "Que tipos de habitacion hay"),
]
print("\n=== respuesta estructurada por objetivo ===")
hit = 0
for label, goal in GOALS:
    ans = sa.answer_for(facts, goal)
    if ans:
        hit += 1
    print(f"  {label:24s} -> {json.dumps(ans, ensure_ascii=False) if ans else 'SIN DATO ESTRUCTURADO'}")
print(f"=== cubiertos {hit}/{len(GOALS)} con solo JSON-LD + FAQ + selects ===")
