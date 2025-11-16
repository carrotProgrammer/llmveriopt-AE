#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Adjust the number of test samples to run for each config.
# Set LIMIT to an integer (e.g., 8, 16, 24) or to 'null'
# to disable sample limiting.

# Note: The full evaluation test set used in the paper contains
# a maximum of **4386 samples**. Evaluators may choose a smaller
# LIMIT to reduce inference time.
# ============================================================
LIMIT=1
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# NOTE FOR ARTIFACT EVALUATION:
# It is strongly recommended to run *only the 3B models*
# during evaluation. Models of 7B or larger typically require
# **at least 32GB of GPU memory** to perform inference
# (depending on precision). Running these larger models on CPU
# is possible but will be **extremely slow**.
# ============================================================

# ============================================================
# List of configuration files to evaluate.
# Comment out any configs you do not want to run.
# Paths are relative to this script's directory.
# ============================================================
CONFIG_FILES=(
  "configs/model_latency.yaml"
  # "configs/sft_qwen_32b.yaml"
  # "configs/sft_qwen_7b.yaml"
  "configs/sft_qwen_3b.yaml"
  "configs/sft_llama3_3b.yaml"
  # "configs/sft_llama3_8b.yaml"
  "configs/model_correctness.yaml"
  # "configs/sft_codellama_7b.yaml"
  "configs/model_zero.yaml"
  "configs/warm_up_model.yaml"
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

  # ------------------------------------------------------------
  # Update the 'limit' field inside the temporary YAML config.
  # If LIMIT=null, write "limit: null". Otherwise, write the
  # integer value.
  # ------------------------------------------------------------
  if [ "$LIMIT" = "null" ]; then
      sed -i 's/^limit: .*/limit: null/' "$TMP_CFG" || echo "limit: null" >> "$TMP_CFG"
  else
      sed -i "s/^limit: .*/limit: ${LIMIT}/" "$TMP_CFG" || echo "limit: ${LIMIT}" >> "$TMP_CFG"
  fi

  # Run inference with the active configuration.
  python "${SCRIPT_DIR}/inference_demo.py"
done

echo "[ALL DONE]"
