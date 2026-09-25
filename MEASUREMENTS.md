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

## 5. Precision FUERA de dominio (el numero que el benchmark del hotel no podia dar)
Medido con `eval_precision_ooc.py` sobre 400-500 paginas **reales** de otros sitios (split Travel de
Mind2Web, que ya estaba en disco): son objetivos de ACCION, asi que cualquier afirmacion de la capa
extractiva es falsa por definicion. Antes de tocar nada, la heuristica afirmaba algo en el **65,4 %**
de los casos, y los arranques mas frecuentes lo decian todo: "skip to main content menu" (23),
"skip to content cart my bo" (14), "upgrade your browser" (9), "welcome to united.com skip" (7),
"my profile sign out" (5). Era la **cabecera de la pagina** tomada como evidencia, porque el scraper
reescanea el texto entero en el segundo paso y la cabecera llega densisima de palabras comunes, que es
exactamente lo que premia la densidad.

Despues de tres reglas estructurales:
- `is_navigation_text()` — cromo inequivoco descarta; los marcadores ambiguos (menu, english, cart)
  hacen falta dos. **65,4 % -> 19,5 %**.
- Filtro de repeticion sobre la pieza intacta (listas de enlaces y calendarios repiten fichas;
  `attorneys near X, NJ pedicure salon near Y...`). **19,5 % -> 11,8 %**.
- Filtro de **listas de etiquetas**: proporcion de tokens que empiezan en mayuscula fuera del
  primero. Medido sobre lo que quedaba: cromo con mediana 0,69 y maximo 1,00, mientras que las cinco
  evidencias buenas del hotel estan entre 0,00 y 0,36. **11,8 % -> 4,5 % a 400 paginas** (y 8,1 % a
  2000, que es el numero que vale: ver abajo).

Los dos umbrales los fijo el gate, no la intuicion: aplicar el filtro de repeticion tambien a las
ventanas recortadas mataba evidencia valida dentro de bloques enormes sin puntuacion, y el umbral
0,50 de mayusculas se comia "Our facilities include an outdoor Swimming Pool and a Fitness Center"
(los nombres de instalacion van en mayuscula: justo 0,50). Con 0,60 queda fuera el cromo y dentro la
frase, y las dos cosas estan congeladas como test.

**Limitacion del instrumento, dicha en seco:** `convert_mind2web_v2.py` escribe la CATEGORIA en el
campo url de la fila, asi que desde este artefacto no se pueden contar dominios distintos: lo que se
puede afirmar es "paginas reales del split Travel de Mind2Web" (2000/2000 en la categoria `Travel`).
No es un corpus de sitios cualesquiera, es un dominio tematico, y por eso el 8,1 % es una cota sobre
ese tipo de pagina, no una promesa universal. Re-verificado con el denominador corregido (el script
dividia por N en vez de por las filas evaluadas: resulto ser `evaluadas == N`, o sea que la cifra no
cambiaba, pero el bug estaba ahi).

**El numero se sostiene fuera de la categoria Travel (medido despues de escribir la limitacion de
arriba):** con `data/m2w2.train.jsonl`, que es **Shopping (999) + Entertainment (501)**, la
sobreafirmacion sale **7,7 %** (116/1500) frente al 8,1 % de Travel. Misma magnitud en tres
categorias, asi que el 8 % es una propiedad del heuristico y no un artefacto del muestreo. El desglose
por tipo ademas dice donde mirar: en Travel el residuo era casi todo `extractiva` (prosa de UI),
mientras que en Shopping/Entertainment se cuelan los **patrones** — `precio` 23, `hora` 11,
`correo` 9. En e-commerce el cromo tambien lleva precios, horarios y correos de boletin, y esos tres
patrones no exigen el contexto que si pide `anchored_price`. Siguiente paso concreto, sin hacer.

El benchmark del hotel, re-medido despues de todo: **16/16 concluyen, 15 con respuesta, 15 con
procedencia verificada** (5 literales + 10 de campo publicado) — ninguna respuesta perdida.

**El numero honesto es 8,1 %, no 4,5 %.** Con 400 paginas salio 4,5 %, pero al ampliar a **2000**
paginas la tasa sube a **8,1 %** (162/2000): el subconjunto de 400 era optimista por varianza de
muestreo, y lo anoto porque es exactamente el tipo de numero que se tiende a reportar del lado
favorable. El guard del gate usa 400 paginas por rapidez con tope 12 %, no 8 %: un tope pegado a lo
observado fallaria por muestreo, no por regresion. Lo que queda dentro del 8,1 % es cromo de
formulario de alta densidad ("Yes No Continue Male Female") y precios de widget leidos por
`anchored_price` en paginas de reservas.

## 6. `ubicacion-barrio` resuelto y lo que queda de `gastronomia`
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
