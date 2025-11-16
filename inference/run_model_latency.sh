#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Script: run_model_latency.sh
#
# Purpose:
#   This script evaluates the full model_latency dataset,
#   generating IR outputs.
#
# Runtime:
#   Full execution requires substantial computation. On an
#   NVIDIA RTX 3090 ti GPU, the expected runtime is approximately
#   9–12 hours. Slower or smaller GPUs may require significantly
#   longer time or may encounter out-of-memory errors.
#
#
# Usage:
#   ./run_model_latency.sh
#
# ============================================================
LIMIT=null
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILES=(
  "configs/model_latency.yaml"
)
# ============================================================

echo "[INFO] Starting batch inference with LIMIT = ${LIMIT}"

for cfg in "${CONFIG_FILES[@]}"; do
  FULL_CFG="${SCRIPT_DIR}/${cfg}"

  if [ ! -f "$FULL_CFG" ]; then
    echo "[WARN] Config file not found, skipping: $FULL_CFG"
    continue
  fi

  echo "============================================================"
  echo "[INFO] Running inference using config: $FULL_CFG"
  echo "============================================================"

  # The evaluation script always reads from 'inference_config.yaml'.
  TMP_CFG="${SCRIPT_DIR}/inference_config.yaml"

  # Copy the selected config into the temporary config file.
  cp "$FULL_CFG" "$TMP_CFG"

  if [ "$LIMIT" = "null" ]; then
      sed -i 's/^limit: .*/limit: null/' "$TMP_CFG" || echo "limit: null" >> "$TMP_CFG"
  else
      sed -i "s/^limit: .*/limit: ${LIMIT}/" "$TMP_CFG" || echo "limit: ${LIMIT}" >> "$TMP_CFG"
  fi

  # Run inference with the active configuration.
  python "${SCRIPT_DIR}/inference_demo.py"
done

echo "[ALL DONE]"
