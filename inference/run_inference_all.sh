#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# IMPORTANT (Artifact Evaluation Notice):
# This script (`run_inference_all.sh`) executes inference across *all* models
# and the *full* evaluation test sets (up to 4386 samples per task).
#
# ❗ It is **NOT recommended** to run this script during artifact evaluation.
#
# Models of 7B, 8B, and 32B typically require **≥32GB GPU memory** to run
# efficiently. On single-GPU systems with less memory, inference may:
#   • fail due to insufficient VRAM,
#   • fall back to CPU execution,
#   • take **multiple days to a full week** to finish.
#
# For practical and quick evaluation, please use:
#       👉 `run_inference_demo.sh`
# which performs small-sample inference and completes quickly.
# ============================================================================

# ============================================================
# Adjust the number of test samples to run for each config.
# Set LIMIT to an integer (e.g., 8, 16, 24) or to 'null'
# to disable sample limiting.
#
# Note: The full evaluation test set used in the paper contains
# a maximum of **4386 samples**. Evaluators may choose a smaller
# LIMIT to reduce inference time.
# ============================================================
LIMIT=null
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# List of configuration files to evaluate.
# Comment out any configs you do not want to run.
# Paths are relative to this script's directory.
# ============================================================
CONFIG_FILES=(
  "configs/model_latency.yaml"
  "configs/sft_qwen_32b.yaml"
  "configs/sft_qwen_7b.yaml"
  "configs/sft_qwen_3b.yaml"
  "configs/sft_llama3_3b.yaml"
  "configs/sft_llama3_8b.yaml"
  "configs/model_correctness.yaml"
  "configs/sft_codellama_7b.yaml"
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

  TMp_CFG="${SCRIPT_DIR}/inference_config.yaml"
  cp "$FULL_CFG" "$TMP_CFG"

  if [ "$LIMIT" = "null" ]; then
      sed -i 's/^limit: .*/limit: null/' "$TMP_CFG" || echo "limit: null" >> "$TMP_CFG"
  else
      sed -i "s/^limit: .*/limit: ${LIMIT}/" "$TMP_CFG" || echo "limit: ${LIMIT}" >> "$TMP_CFG"
  fi

  python "${SCRIPT_DIR}/inference_demo.py"
done

echo "[ALL DONE]"
