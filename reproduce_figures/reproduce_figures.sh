#!/bin/bash
set -e
cd "$(dirname "$0")" 

# function to run one config
run_cfg () {
  local cfg_name=$1
  shift
  echo ">>> Running $cfg_name ..."
  cat > artifact_viz/config/config.yaml <<EOF
$@
EOF
  python3 -m artifact_viz.cli
}

# --- 1. Only Model_Correctness_3B ---
run_cfg cfg1 "
mode: bar
models:
  - Model_Correctness_3B
metrics:
  - correctness
dataset_path: ./summary_table
output_dir: outputs
log_level: INFO
"

# --- 2. Only Qwen_3B ---
run_cfg cfg2 "
mode: bar
models:
  - Qwen_3B
metrics:
  - correctness
dataset_path: ./summary_table
output_dir: outputs
log_level: INFO
"

# --- 3. Latency/Size/ICount with 3 models ---
run_cfg cfg3 "
mode: bar
models:
  - Model_Latency_3B
  - Model_Correctness_3B
  - Qwen_3B
metrics:
  - latency
  - BinSize
  - ICount
dataset_path: ./summary_table
output_dir: outputs
log_level: INFO
"

# --- 4. Big set, correctness ---
run_cfg cfg4 "
mode: bar
models:
  - Model_Latency_3B
  - Model_Correctness_3B
  - Qwen_3b_SFT
  - Llama_3B_SFT
  - Qwen_7b_SFT
  - CodeLlama_7B_SFT
  - LLM_Compiler_7B
  - Llama_8b_SFT
  - Qwen_32b_SFT
metrics:
  - correctness
dataset_path: ./summary_table
output_dir: outputs
log_level: INFO
"

# --- 5. Same models, latency only ---
run_cfg cfg5 "
mode: bar
models:
  - Model_Latency_3B
  - Model_Correctness_3B
  - Qwen_3b_SFT
  - Llama_3B_SFT
  - Qwen_7b_SFT
  - CodeLlama_7B_SFT
  - LLM_Compiler_7B
  - Llama_8b_SFT
  - Qwen_32b_SFT
metrics:
  - latency
dataset_path: ./summary_table
output_dir: outputs
log_level: INFO
"

# --- 6. Same models, BinSize ---
run_cfg cfg6 "
mode: bar
models:
  - Model_Latency_3B
  - Model_Correctness_3B
  - Qwen_3b_SFT
  - Llama_3B_SFT
  - Qwen_7b_SFT
  - CodeLlama_7B_SFT
  - LLM_Compiler_7B
  - Llama_8b_SFT
  - Qwen_32b_SFT
metrics:
  - BinSize
dataset_path: ./summary_table
output_dir: outputs
log_level: INFO
"

# --- 7. Same models, ICount ---
run_cfg cfg7 "
mode: bar
models:
  - Model_Latency_3B
  - Model_Correctness_3B
  - Qwen_3b_SFT
  - Llama_3B_SFT
  - Qwen_7b_SFT
  - CodeLlama_7B_SFT
  - LLM_Compiler_7B
  - Llama_8b_SFT
  - Qwen_32b_SFT
metrics:
  - ICount
dataset_path: ./summary_table
output_dir: outputs
log_level: INFO
"

# --- 8. Mixed set with Model_Zero & Warmup ---
run_cfg cfg8 "
mode: bar
models:
  - Model_Zero
  - Warmup_Model
  - Model_Correctness_3B
  - Model_Latency_3B
metrics:
  - correctness
  - latency
  - BinSize
  - ICount
dataset_path: ./summary_table
output_dir: outputs
log_level: INFO
"

# --- 9. Violin plot run ---
run_cfg cfg9 "
mode: violin
models:
  - Model_Latency_3B
  - LLVM_O0
  - LLVM_instcombine
metrics:
  - latency
  - BinSize
  - ICount
dataset_path: ./summary_table
output_dir: outputs
log_level: INFO
"

echo ">>> All configs finished!"
