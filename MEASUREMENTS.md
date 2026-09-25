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
- **15/16 con respuesta**, y la unica ausencia es honesta y acotada a lo recorrido
  (`not_found_in_survey` para "tiene piscina": el sitio no publica piscina).
- **0 respuestas inventadas**: toda respuesta lleva `source` con la pagina o el campo estructurado
  del que salio.

VRAM medida en el mismo cacharro (`deploy-3050/test-3050.py`): 3.58-3.92 GB de pico a 4.9 tok/s.
Cuanto no gana tiempo, si no memoria; la via para latencia seria GGUF/llama.cpp, no probada.

## 4. Lo que se midio para no repetir errores
- Prompt por la izquierda: `policy.py` devuelve `_error: prompt_over_budget` si no cabe, en vez de
  truncar el observador por la derecha (el bug que hacia que el modelo viera un menu incompleto).
- Respuestas extractivas: se descarta texto de terceros (caruseles de resenas) aunque contenga la
  palabra buscada, y se separa la "costura" que innerText pega entre si. Ver `test_menu.py`.
- Precio anclado a la tarjeta preguntada, no a la primera moneda de la pagina.
