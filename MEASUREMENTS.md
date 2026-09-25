# Measurements

Todo lo de aqui abajo esta reproducido con el codigo de este repositorio y los logs de `logs/`.
Los numeros de exactitud son **accion exacta** (acertar el indice del candidato), no "respuesta
correcta": es la metrica que permite comparar base vs LoRA sobre las tres distribuciones.

## 1. El LoRA gana a la base en las tres distribuciones (control obligatorio)

| distribucion | base Qwen3-1.7B | LoRA v4 | azar (1/k) |
|---|---|---|---|
| sintetica (juguetes, k=2-3) | 0.322 | 0.753 | 0.42 |
| Mind2Web real (plana, k=7-10) | 0.191 | 0.586 | 0.12 |
| `m2w2` realista (k=20, split por sitio) | 0.068 | 0.781 | 0.05 |

Leido del log con el adapter desactivado (`--adapter none`), no de memoria: la objecion central de
la revision era que el gain podia ser de la base, y el control lo descarta en las tres.
Ver `logs/eval-v4full.log`, `logs/eval-base2.log`, `logs/eval-r2.log`.

## 2. Fuga train/valid
Auditoria con `leak_audit.py` sobre hashes normalizados (url + texto + indices): **0 %** de
solapamiento entre `m2w2.train.jsonl` y `m2w2.valid.jsonl`. El split agrupa por `website`, no por
`domain` (en ese subconjunto `domain` trae la categoria, y agrupar por ello dejaba 3 grupos y un
valid mayor que el train).

## 3. Operator end-to-end en el hardware objetivo (RTX 3050 6 GB)
16 objetivos reales de un sitio de hoteles propio, lectura sola, escrituras apagadas, 8 pasos de
techo, modelo v4 cargado en la GPU local: ver `deploy-3050/bench16-local.json`.

- **16/16 concluyen** (ninguno se queda colgado ni agota el presupuesto sin veredicto).
- **15/16 con respuesta**, y la unica ausencia es honesta y **acotada a lo recorrido**:
  `not_found_in_survey` para "tiene piscina" declarado sobre **5 paginas distintas y 14.786
  caracteres leidos** (`survey_pages`/`survey_chars` en la fila del JSON). El numero pequeno es
  informativo a proposito: una ausencia afirmada sobre poco texto vale menos que una afirmada sobre
  el sitio entero, y el operador prefiere no gastar pasos inventando cobertura.
- **0 respuestas inventadas, y eso esta comprobado a maquina**: cada respuesta lleva
  `source_url`, y `stamp_evidence()` exige que el valor devuelto aparezca tal cual en el
  `innerText` de una pagina recorrida. Desglose de esa cifra, porque los dos grados no son
  intercambiables: **5 literales en pagina** (`verbatim: true`) + **10 leidas de un campo
  publicado** (`jsonld`, `faq:...`, `select:room-type`). El desglose esta en
  `deploy-3050/bench16-local.json`, objetivo a objetivo, y `run-local16.py` lo imprime.

VRAM medida en el mismo cacharro (`deploy-3050/test-3050.py`): 3.58-3.92 GB de pico a 4.9 tok/s.
Cuanto no gana tiempo, si no memoria; la via para latencia seria GGUF/llama.cpp, no probada.

## 4. Lo que se midio para no repetir errores
- Prompt por la izquierda: `policy.py` devuelve `_error: prompt_over_budget` si no cabe, en vez de
  truncar el observador por la derecha (el bug que hacia que el modelo viera un menu incompleto).
- Respuestas extractivas: se descarta texto de terceros (caruseles de resenas) aunque contenga la
  palabra buscada, y se separa la "costura" que innerText pega entre si. Ver `test_menu.py`.
- Precio anclado a la tarjeta preguntada, no a la primera moneda de la pagina.

## 5. Limitacion conocida, medida y abierta: `ubicacion-barrio` responde con basura pegada
**RESUELTA el 25-sep (commit `949135e`).** El dato llega a nivel de linea ("The courtyard, San
Pedro"), y lo que impedia partirla era el suelo del extractor (>=25 caracteres y >=5 palabras),
calibrado para el megachunk colapsado. Simulado sobre el /contact/ real: de las cinco piezas del
bloque solo una trae el termino distintivo del objetivo, y el relleno de navegacion ninguno. Con
dos vistas del innerText (prompt colapsado / extractor con fronteras de bloque) y suelo de 12
caracteres y 3 palabras, la corrida queda en **16/16, 15 respuestas, 15 con procedencia** y
`ubicacion-barrio` = "The courtyard, San Pedro".

Abierto en su lugar, y mas pequeno: `gastronomia` contesto "Cafeteria open until 10:00 pm" en vez de
nombrar **AUKA RESTOBAR**, porque el puntuador prefiere la pieza mas densa entre dos con un solo
termino comun. Desempatar por cantidad de nombre propio (`proper_tokens`, 0 vs 6 en las dos piezas
candidatas) **se midio y fallo**: sobre la pagina completa eligio la descripcion del desayuno, que
tambien trae "coffee" y un capitalizado.

Resuelto el ranking con una senal mas especifica: `names_venue()` mira mayusculas que compartan
principio con el tipo de local preguntado (`RESTOBAR` ~ `restaurante`), y solo actua como desempate
cuando dos piezas traen el mismo numero de terminos. Sobre 60 observaciones reales cacheadas la pieza
ganadora pasa a ser **AUKA RESTOBAR en 9 de 9** paginas que lo contienen (congelado en `test_menu.py`).

Lo que queda, y ya no es ranking sino politica de parada: en la corrida de 16 objetivos `gastronomia`
sigue contestando "Cafeteria open until 10:00 pm" **porque el operador se detuvo en la home, cuya
seccion de servicios no publica el nombre** (lo publicado ahi es el horario de la cafeteria). Seguir
buscando teniendo una respuesta valida cuesta pasos y puede empeorar la respuesta; es decision
de politica, no un bug, y esta medida en `logs/bench16-venue.log` (16/16, 15 respuestas, 15 con
procedencia, resto de valores identicos).
