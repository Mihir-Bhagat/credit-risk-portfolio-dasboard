#!/usr/bin/env bash
# Simulates a reviewer: copy what would be in a clean checkout, create a fresh virtualenv, run ONLY the README commands,
# then load every view in a real browser. Usage: scripts/fresh_venv_check.sh [python-for-playwright]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBROWSER="${1:-python3}"
TMP="$(mktemp -d)"
trap '[ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null; rm -rf "$TMP"; true' EXIT
mkdir -p "$TMP/repo"
cd "$ROOT"
git ls-files -co --exclude-standard -z | while IFS= read -r -d "" f; do [ -e "$f" ] && cp --parents "$f" "$TMP/repo/"; done
cd "$TMP/repo"
# Remove every generated artifact: the README commands must recreate them from the raw CSVs.
rm -rf data/processed reports models outputs docs/architecture.pdf docs/solution_overview.pdf docs/data_quality_log.md
mkdir -p outputs
PYTHON_BIN="${FRESH_PYTHON:-python3}"
"$PYTHON_BIN" -m venv "$TMP/venv"
export PATH="$TMP/venv/bin:$PATH"
# Extract the run commands from the first fenced bash block under "Run it" in README.md
CMDS="$(awk '/^## Run it/{f=1} f&&/^```bash/{b=1;next} b&&/^```/{exit} b{print}' README.md | sed 's/[[:space:]]*#.*$//' | grep -v '^[[:space:]]*$')"
N="$(printf '%s\n' "$CMDS" | wc -l)"
echo "README run commands ($N):"; printf '%s\n' "$CMDS"
[ "$N" -le 3 ] || { echo "FAIL: more than 3 run commands"; exit 1; }
INSTALL="$(printf '%s\n' "$CMDS" | sed -n 1p)"
BUILD="$(printf '%s\n' "$CMDS" | sed -n 2p)"
START="$(printf '%s\n' "$CMDS" | sed -n 3p)"
echo ">> $INSTALL"; eval "$INSTALL" > "$TMP/install.log" 2>&1 || { tail -30 "$TMP/install.log"; exit 1; }
echo ">> $BUILD";   eval "$BUILD"   > "$TMP/build.log" 2>&1   || { tail -30 "$TMP/build.log"; exit 1; }
tail -3 "$TMP/build.log"
for f in data/processed/portfolio.parquet reports/metrics.json docs/architecture.pdf docs/solution_overview.pdf docs/data_quality_log.md; do
  [ -s "$f" ] || { echo "FAIL: build did not create $f"; exit 1; }
done
PORT="$("$PYBROWSER" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')"
echo ">> $START   (port chosen via STREAMLIT_SERVER_PORT=$PORT)"
( export STREAMLIT_SERVER_PORT="$PORT" STREAMLIT_SERVER_HEADLESS=true; exec bash -c "exec $START" ) > "$TMP/app.log" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 90); do
  if curl -fsS http://localhost:$PORT/_stcore/health 2>/dev/null | grep -q ok; then HEALTH=ok; break; fi
  sleep 1
done
[ "${HEALTH:-}" = "ok" ] || { echo "FAIL: app did not become healthy"; tail -30 "$TMP/app.log"; exit 1; }
echo "app healthy"
mkdir -p "$ROOT/tests/reports/screenshots"
"$PYBROWSER" "$ROOT/scripts/browser_qa.py" http://localhost:$PORT "$ROOT/tests/reports/screenshots"
echo "FRESH VENV CHECK: PASS"
