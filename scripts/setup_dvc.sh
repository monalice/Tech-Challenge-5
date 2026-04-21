#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x "$ROOT_DIR/.venv/Scripts/dvc.exe" ]; then
  DVC_CMD="$ROOT_DIR/.venv/Scripts/dvc.exe"
  DVC_CMD_IS_WINDOWS_EXE=1
elif [ -x "$ROOT_DIR/.venv/bin/dvc" ]; then
  DVC_CMD="$ROOT_DIR/.venv/bin/dvc"
  DVC_CMD_IS_WINDOWS_EXE=0
elif command -v dvc >/dev/null 2>&1; then
  DVC_CMD="$(command -v dvc)"
  DVC_CMD_IS_WINDOWS_EXE=0
else
  echo "[ERROR] dvc não encontrado no PATH. Instale as dependências com 'pip install -r requirements.txt'." >&2
  exit 1
fi

run_dvc() {
  if [ "${DVC_CMD_IS_WINDOWS_EXE:-0}" -eq 1 ]; then
    if command -v py >/dev/null 2>&1; then
      py -m dvc "$@"
      return
    fi
  fi

  "$DVC_CMD" "$@"
}

if [ ! -d ".dvc" ]; then
  echo "[INFO] Inicializando DVC no repositório..."
  run_dvc init
else
  echo "[INFO] DVC já inicializado."
fi

REMOTE_URL="${DVC_REMOTE_URL:-}"

if [ -z "$REMOTE_URL" ]; then
  REMOTE_NAME="localremote"
  REMOTE_URL="./.dvc/localstore"
else
  REMOTE_NAME="s3remote"
fi

if run_dvc remote list | grep -q "^${REMOTE_NAME}[[:space:]]"; then
  echo "[INFO] Atualizando remote padrão ${REMOTE_NAME}..."
  run_dvc remote modify "$REMOTE_NAME" url "$REMOTE_URL"
else
  echo "[INFO] Configurando remote padrão ${REMOTE_NAME}..."
  run_dvc remote add -d "$REMOTE_NAME" "$REMOTE_URL"
fi

if git ls-files --error-unmatch models/btc_hourly_cache.csv >/dev/null 2>&1; then
  echo "[INFO] Removendo models/btc_hourly_cache.csv do rastreio do Git para que o DVC possa gerenciá-lo..."
  git rm -r --cached --quiet models/btc_hourly_cache.csv
fi

if [ -f "dvc.yaml" ] && grep -q "models/btc_hourly_cache.csv" dvc.yaml; then
  echo "[INFO] models/btc_hourly_cache.csv já é rastreado pelo pipeline em dvc.yaml; pulando dvc add."
  if [ ! -f "models/btc_hourly_cache.csv" ]; then
    echo "[INFO] O cache ainda não existe localmente. Execute 'dvc repro' para gerá-lo."
  fi
else
  if [ ! -f "models/btc_hourly_cache.csv" ]; then
    echo "[ERROR] O arquivo models/btc_hourly_cache.csv não existe. Gere ou copie o cache antes de executar o setup do DVC." >&2
    exit 1
  fi
  echo "[INFO] Adicionando models/btc_hourly_cache.csv ao rastreio do DVC..."
  run_dvc add models/btc_hourly_cache.csv
fi

echo "[INFO] Setup do DVC concluído."
