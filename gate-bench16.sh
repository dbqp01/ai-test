cd /root/site2tools-mvp || exit 1
.venv/bin/python test_menu.py > logs/test_menu.log 2>&1
rc=$?
echo "EXITO_TEST=$rc"
tail -3 logs/test_menu.log
if [ "$rc" -ne 0 ]; then
  echo "NO_SE_LANZA_BENCH: tests en rojo"
  exit 1
fi
bash bench16.sh > logs/bench16b.log 2>&1
echo "BENCH16b_EXITO=$?"
grep -aE "^#####|^   R:|BENCH16" logs/bench16b.log | tr -d '\r' | tail -22
