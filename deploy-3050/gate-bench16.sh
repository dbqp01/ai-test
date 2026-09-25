#!/bin/bash
# Corre la bateria de tests CPU y, solo si sale verde, lanza la medicion de 16 objetivos.
# El gate existe porque una vez lancé el bench con los tests en rojo por leer $? despues de
# un pipe: el estado que se mira es siempre el del python de test, sin intermediarios.
cd "$(dirname "$0")"
# Vale en las dos posiciones donde vive el script: en el portatil esta en la raiz del proyecto, y
# en el repo esta dentro de deploy-3050/, asi que hay que subir un nivel para encontrar site2tools.
[ -d site2tools ] || cd ..
# La copia solo aplica en el portatil, donde se edita fuera del arbol. En un clone limpio del repo
# no existe esa ruta y el gate no debe fallar por eso.
[ -f ../site2tools-backup/snap0042/site2tools/core.py ] && cp ../site2tools-backup/snap0042/site2tools/core.py site2tools/core.py
# El test tambien: si no se copia, el gate puede ponerse verde con una bateria de tests vieja
# mientras el canon ha cambiado, que es justo el fallo que ya cometi subiendo codigo suelto.
[ -f ../site2tools-backup/test_menu.py ] && cp ../site2tools-backup/test_menu.py test_menu.py
mkdir -p logs
sha256sum site2tools/core.py | tee -a logs/code-hash.txt
./.venv/Scripts/python.exe test_menu.py > logs/gate.log 2>&1
rc=$?
tail -6 logs/gate.log
if [ $rc -ne 0 ]; then
  echo "GATE ROJO rc=$rc - no se lanza el bench"
  exit $rc
fi
echo "GATE VERDE - lanzando bench16"
./.venv/Scripts/python.exe run-local16.py > logs/bench16.log 2>&1
echo "BENCH rc=$?"
tail -24 logs/bench16.log
