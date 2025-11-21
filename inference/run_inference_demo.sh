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
LIMIT=32
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
  # "configs/sft_llama3_3b.yaml"
  # "configs/sft_llama3_8b.yaml"
  "configs/model_correctness.yaml"
  # # "configs/sft_codellama_7b.yaml"
  # "configs/model_zero.yaml"
  # "configs/warm_up_model.yaml"
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

  TMP_CFG="${SCRIPT_DIR}/inference_config.yaml"
  cp "$FULL_CFG" "$TMP_CFG"

  if [ "$LIMIT" = "null" ]; then
      sed -i 's/^limit: .*/limit: null/' "$TMP_CFG" || echo "limit: null" >> "$TMP_CFG"
  else
      sed -i "s/^limit: .*/limit: ${LIMIT}/" "$TMP_CFG" || echo "limit: ${LIMIT}" >> "$TMP_CFG"
  fi

  python "${SCRIPT_DIR}/inference_demo.py"
done

echo "============================================================"
echo "[INFO] Inference finished. Starting verification..."
echo "============================================================"


# ============================================================
# NEW SECTION: verify all generated results
# ============================================================

RESULT_ROOT="${SCRIPT_DIR}/output/new_result"

if [ ! -d "$RESULT_ROOT" ]; then
    echo "[ERROR] Directory not found: $RESULT_ROOT"
    exit 1
fi

find "$RESULT_ROOT" -type f -name "results.csv" | while read -r csvfile; do
    echo "[INFO] Verifying: $csvfile"

    model_dir="$(dirname "$csvfile")"
    metrics_out="${model_dir}/metrics.json"

    TOOLS_DIR="${SCRIPT_DIR}/tools"

    # Ensure latency tool is executable
    chmod +x "${TOOLS_DIR}/aarch64_tti_latency"
    chmod +x "${TOOLS_DIR}/llvm/build/bin/opt"
    chmod +x "${TOOLS_DIR}/llvm/build/bin/llc"
    chmod +x "${TOOLS_DIR}/llvm/build/bin/llvm-size"
    chmod +x "${TOOLS_DIR}/alive-tv"

    # Run verify.py and capture exit status
    if python "${SCRIPT_DIR}/verify.py" \
        --input "$csvfile" \
        --output "$metrics_out" \
        --alive "${TOOLS_DIR}/alive-tv" \
        --latency "${TOOLS_DIR}/aarch64_tti_latency" \
        --llvm-bin "${TOOLS_DIR}/llvm/build/bin" \
        --instcount "${TOOLS_DIR}/llvm/build/lib/InstCount.so"; then

        echo "[INFO] Metrics saved to: $metrics_out"

    else
        echo "[ERROR] Verification FAILED for $csvfile"
        # Remove possibly corrupted JSON file
        rm -f "$metrics_out"
    fi

done

echo "[INFO] Generating summary plots..."
python "${SCRIPT_DIR}/plot_metrics.py" --root "$RESULT_ROOT"


echo "[ALL DONE]"