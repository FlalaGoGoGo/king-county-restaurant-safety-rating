#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-/Users/zhangziling/Documents/Project_King_County_Safety_Rating}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PAGE_SIZE="${PAGE_SIZE:-50000}"
FETCHER="${FETCHER:-auto}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-3}"

# Default policy:
# - Sunday: full refresh
# - Other days: incremental with lookback window
if [[ "${FORCE_FULL:-0}" == "1" ]]; then
  MODE="full"
else
  DOW="$(date +%u)" # 1..7 (Mon..Sun)
  if [[ "$DOW" == "7" ]]; then
    MODE="full"
  else
    MODE="incremental"
  fi
fi

echo "[daily] root=$ROOT"
echo "[daily] mode=$MODE"
echo "[daily] fetcher=$FETCHER"

run_pipeline() {
  local run_mode="$1"
  if [[ "$run_mode" == "full" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/run_food_inspection_pipeline.py" \
      --root "$ROOT" \
      --mode full \
      --page-size "$PAGE_SIZE" \
      --fetcher "$FETCHER"
  else
    "$PYTHON_BIN" "$ROOT/scripts/run_food_inspection_pipeline.py" \
      --root "$ROOT" \
      --mode incremental \
      --lookback-days "$LOOKBACK_DAYS" \
      --page-size "$PAGE_SIZE" \
      --fetcher "$FETCHER"
  fi
}

if [[ "$MODE" == "incremental" ]]; then
  if run_pipeline incremental; then
    echo "[daily] incremental run succeeded"
  else
    echo "[daily] incremental run failed, falling back to full run"
    run_pipeline full
  fi
else
  run_pipeline full
fi
