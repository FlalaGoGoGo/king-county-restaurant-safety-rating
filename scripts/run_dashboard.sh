#!/usr/bin/env zsh
set -euo pipefail

ROOT="${ROOT:-/Users/zhangziling/Documents/Project_King_County_Safety_Rating}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8501}"
HOME_DIR="${DASHBOARD_HOME:-$ROOT}"
CONFIG_DIR="$HOME_DIR/.streamlit"
CREDS_FILE="$CONFIG_DIR/credentials.toml"
CONFIG_FILE="$CONFIG_DIR/config.toml"

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
  echo "[dashboard] streamlit is not installed."
  echo "[dashboard] install dependencies first:"
  echo "  $PYTHON_BIN -m pip install -r $ROOT/requirements.txt"
  exit 1
fi

mkdir -p "$CONFIG_DIR"
if [[ ! -f "$CREDS_FILE" ]]; then
  cat > "$CREDS_FILE" <<EOF
[general]
email = ""
EOF
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  cat > "$CONFIG_FILE" <<EOF
[browser]
gatherUsageStats = false
EOF
fi

export HOME="$HOME_DIR"

echo "[dashboard] root=$ROOT"
echo "[dashboard] port=$PORT"
echo "[dashboard] app=$ROOT/app/dashboard_app.py"
echo "[dashboard] home=$HOME"

printf "\n" | "$PYTHON_BIN" -m streamlit run "$ROOT/app/dashboard_app.py" \
  --server.port "$PORT" \
  --server.address "127.0.0.1"
