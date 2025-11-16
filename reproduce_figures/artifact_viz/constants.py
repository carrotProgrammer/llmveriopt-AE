# -*- coding: utf-8 -*-

# Allowed model and metric names (enumerations)
ALLOWED_MODELS = {
    "LLVM_O0",
    "LLVM_instcombine",
    "Qwen_3B",
    "Model_Latency_3B",
    "Model_Correctness_3B",
    "CodeLlama_7B_SFT",
    "Llama_3B_SFT",
    "Llama_8b_SFT",
    "Qwen_3b_SFT",
    "Qwen_7b_SFT",
    "Qwen_32b_SFT",
    "LLM_Compiler_7B",
    "Model_Zero",
    "Warmup_Model",
}

ALLOWED_METRICS = {
    "correctness",
    "latency",
    "ICount",
    "BinSize",
    "real_latency",  # not implemented yet; placeholder
}

# The four non-correctness metrics (drive Table III / comparisons)
NON_CORR_METRICS = {"latency", "ICount", "BinSize", "real_latency"}

ALLOWED_PLOT_MODES = {"bar", "violin"}

BASELINE_MODELS = {"LLVM_O0", "LLVM_instcombine"}
VIOLIN_TARGET_MODELS = {"Model_Latency_3B"}


class Scenario:
    """Output scenarios"""
    A_TABLE_I = "A_TABLE_I"                 # Single model + correctness -> Table I
    B_TABLE_III = "B_TABLE_III"             # Single model + any of the non-correctness metrics -> Table III
    C_FIG5 = "C_FIG5"                       # Multiple models + single metric -> Fig5
    D_TABLEIII_FIG7 = "D_TABLEIII_FIG7"     # Multiple models + multiple metrics -> Table III (extended) + Fig7
