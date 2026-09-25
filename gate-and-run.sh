cd /root/site2tools-mvp || exit 1
sha256sum site2tools/core.py site2tools/policy.py structured_answers.py 2>/dev/null | cut -c1-72 | tee logs/code-hash.txt

# 1) tests unitarios del harness (sin GPU, sin red)
.venv/bin/python test_menu.py > logs/test_menu.log 2>&1
rc=$?
echo "EXITO_TEST_UNITARIO=$rc"
tail -3 logs/test_menu.log

# 2) el atajo estructurado, contra el sitio real (solo lectura)
.venv/bin/python - > logs/test_structured.log 2>&1 <<PY
import structured_answers as sa
facts = sa.extract_from_url("https://usgarhoteles.com/")
got = sa.answer_for(facts, "A que hora es el check-in")
print("check-in:", got)
assert got and "12:00" in str(got["value"]), got
bad = sa.answer_for(facts, "Cual es la clave del wifi")
print("wifi-clave (debe ser None):", bad)
assert bad is None, bad
# El operador si debe responderla: "no esta publicada", no un presupuesto agotado.
import site2tools.core as c
got = c.structured_shortcut("Cual es la clave del wifi", ["https://usgarhoteles.com/"])
print("atajo wifi-clave:", got)
assert got and got["kind"] == "wifi-clave" and "no publica" in got["value"], got
print("TODO ESTRUCTURADO OK")
PY
rc2=$?
echo "EXITO_TEST_ESTRUCTURADO=$rc2"
tail -4 logs/test_structured.log

if [ "$rc" -ne 0 ] || [ "$rc2" -ne 0 ]; then
  echo "NO_SE_LANZA: tests en rojo"
  exit 1
fi
setsid nohup bash run-hotel9.sh > logs/hotel17.log 2>&1 < /dev/null &
echo "BENCHMARK_LANZADO hotel17"
