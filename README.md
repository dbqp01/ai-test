# Site2Tools MVP

Prototipo para convertir un website en un único operador universal:

    objetivo en lenguaje natural -> site_operator -> navegador -> verificación

La primera versión construye los datos que necesitaremos para entrenar un modelo: estados, acciones, transiciones, errores y resultados verificables. La decisión inicial usa una heurística pequeña para funcionar desde el primer día; luego se sustituye por un modelo local.

## Ejecutar en el MI300X

    cd ~/site2tools-mvp
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m playwright install chromium
    python -m site2tools.server --host 0.0.0.0 --port 8080 --data ./data

Comprobar:

    curl http://127.0.0.1:8080/health
    curl -X POST http://127.0.0.1:8080/explore -H 'content-type: application/json' -d '{"start_url":"https://example.com","max_states":8,"max_depth":2}'

El contrato público siempre es operate, aunque internamente existan explorador, grafo, ejecutor y verificadores.

## Seguridad

- Cada ejecución usa un contexto de navegador aislado.
- Solo se permiten hosts declarados por la URL inicial y allowed_hosts.
- La exploración bloquea acciones con efectos externos.
- operate devuelve confirmation_required antes de una acción sensible.
- La decisión del modelo nunca se considera prueba de éxito: la verificación la hace el código.
- Usa una cuenta de prueba y una plataforma propia o autorizada.

## Archivos generados

- data/site_graph.json: estados y transiciones descubiertos.
- data/traces.jsonl: observación por paso.
- data/operations.jsonl: objetivos ejecutados y evidencia.
- data/traces/*.zip: trazas Playwright.

## Entrenar la política (LoRA)

`ml/generate_dataset.py` fabrica escenarios reproducibles de educación, reportes, soporte, compras y
cloud; `convert_mind2web_v2.py` y `build_v3_splits.py` construyen los conjuntos reales (`data/m2w2.*`
agrupado por sitio, `data/v3.*`). En la MI300X, el champion actual:

    .venv/bin/python train_lora.py --model Qwen/Qwen3-1.7B \
      --train data/v3.train.jsonl --valid data/v3.valid.jsonl \
      --output artifacts/site2tools-student-17b-v4 --epochs 2

`eval_action_acc.py` mide accción exacta por distribución, y **con control**: `--adapter none` deja
el modelo base delante, que es la única forma de saber si el gain es del LoRA o del fundamental.
Los números están en `MEASUREMENTS.md`; `leak_audit.py` audita el solapamiento train/valid.

## Correrlo en la RTX 3050 (hardware objetivo)

    uv venv .venv --python 3.11 && .venv/Scripts/python.exe -m pip install -r deploy-3050/requirements-3050.txt
    cp -r artifacts/site2tools-student-17b-v4 adapter-v4     # policy.py carga el tokenizer de ahi
    bash deploy-3050/gate-bench16.sh                          # tests CPU y, si salen verdes, los 16 objetivos

El gate existe porque una vez se lanzo una medicion de cinco minutos con los tests en rojo por leer
mal el estado de un pipe. Pico medido: 3,92 GB de VRAM y 4,9 tok/s sin cuantizar.

El modelo solo propone la siguiente decisión (`act`, `request_confirmation`, `finish` o `block`); el
navegador, los permisos y el verificador siguen siendo código determinista: comprueba que el índice
exista, que la acción esté habilitada, que las escrituras tengan confirmación y que `finish` traiga
evidencia observable. Si la salida no se interpreta, vuelve a la heurística. Una buena métrica en
datos sintéticos no demuestra generalización a cualquier sitio real: después hay que registrar sitios
autorizados y medir por dominio.
