#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-/Users/zhangziling/Documents/Project_King_County_Safety_Rating}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_HTML="${OUT_HTML:-$ROOT/outputs/dashboard/index.html}"
MAX_EVENTS_PER_BIZ="${MAX_EVENTS_PER_BIZ:-200}"
MAX_VIOLATIONS_PER_BIZ="${MAX_VIOLATIONS_PER_BIZ:-300}"

echo "[html-dashboard] root=$ROOT"
echo "[html-dashboard] output=$OUT_HTML"
echo "[html-dashboard] max_events_per_business=$MAX_EVENTS_PER_BIZ"
echo "[html-dashboard] max_violations_per_business=$MAX_VIOLATIONS_PER_BIZ"

"$PYTHON_BIN" "$ROOT/scripts/export_html_dashboard.py" \
  --root "$ROOT" \
  --output-html "$OUT_HTML" \
  --max-events-per-business "$MAX_EVENTS_PER_BIZ" \
  --max-violations-per-business "$MAX_VIOLATIONS_PER_BIZ"

echo "[html-dashboard] open file:"
echo "  $OUT_HTML"
