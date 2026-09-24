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

## Entrenamiento de la política local

ml/generate_dataset.py fabrica escenarios reproducibles de educación, reportes, soporte, compras y cloud. Cada ejemplo contiene el objetivo, contexto, observación de acciones, historial y la decisión segura esperada. Incluye confirmaciones, requisitos faltantes, sesión expirada y evidencia de éxito.

En la MI300X:

    python ml/generate_dataset.py --output data/policy.jsonl --count 12000
    pip install -r requirements-ml.txt
    python ml/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 2
    python ml/evaluate_policy.py --adapter artifacts/site2tools-policy-0.5b-lora

Para probarlo como política del operador cuando exista el adaptador:

    python -m site2tools.server --host 127.0.0.1 --port 8080 --data ./data --policy-adapter ./artifacts/site2tools-policy-0.5b-lora

El modelo propone act, request_confirmation, finish o block. El runtime comprueba que el índice exista, que la acción esté habilitada, que las escrituras tengan confirmación y que finish tenga evidencia observable. Si la salida no se puede interpretar, vuelve a la heurística.

## Entrenamiento de la política local

ml/generate_dataset.py fabrica escenarios reproducibles de educación, reportes, soporte, compras y cloud. Cada ejemplo contiene el objetivo, contexto, observación de acciones, historial y la decisión segura esperada. Incluye confirmaciones, requisitos faltantes, sesión expirada y evidencia de éxito.

En la MI300X:

    python ml/generate_dataset.py --output data/policy.jsonl --count 12000
    pip install -r requirements-ml.txt
    python ml/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 2
    python ml/evaluate_policy.py --adapter artifacts/site2tools-policy-lora

El entrenamiento usa LoRA sobre un modelo causal pequeño. El modelo solo propone la siguiente decisión; el navegador, los permisos y el verificador siguen siendo código determinista. Una buena métrica en datos sintéticos no demuestra generalización a cualquier sitio real: después hay que registrar sitios autorizados y medir por dominio.
