#!/usr/bin/env bash
set -euo pipefail

directories=(
  "data/raw"
  "data/processed"
  "src/features"
  "src/models"
  "src/agent"
  "src/serving"
  "monitoring"
  "security"
  "tests"
  "evaluation"
)

for dir in "${directories[@]}"; do
  if [ -d "$dir" ]; then
    echo "[OK] Já existe: $dir"
  else
    mkdir -p "$dir"
    echo "[CRIADO] $dir"
  fi
done

echo "Estrutura de pastas validada com sucesso."
