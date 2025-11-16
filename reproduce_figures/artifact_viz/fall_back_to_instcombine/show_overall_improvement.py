import pandas as pd
from datasets import load_from_disk

# Path
DATA_PATH = "statistic_table"

# Load dataset
dataset = load_from_disk(DATA_PATH)
df = dataset.to_pandas()

# Clean numeric values
cols = [
    "LLVM_instcombine_latency", "Model_Latency_3B_latency",
    "LLVM_instcombine_icount",  "Model_Latency_3B_icount",
    "LLVM_instcombine_size",    "Model_Latency_3B_size"
]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=cols)

# Define a general fallback function
def fallback(model_col, label_col):
    return df.apply(
        lambda row: row[label_col]
        if row[model_col] > row[label_col]
        else row[model_col],
        axis=1
    )

# Compute fallback results
df["final_latency"] = fallback("Model_Latency_3B_latency", "LLVM_instcombine_latency")
df["final_icount"]  = fallback("Model_Latency_3B_icount",  "LLVM_instcombine_icount")
df["final_size"]    = fallback("Model_Latency_3B_size",    "LLVM_instcombine_size")

# Unified print function
def print_stats(name, label_col, final_col):
    improvement = (df[label_col] - df[final_col]).sum()
    improvement_pct = 100 * improvement / df[label_col].sum()
    print(f"[{name}]")
    print(f"  InstCombine avg {name}: {df[label_col].mean():.4f}")
    print(f"  Model+fallback avg {name}: {df[final_col].mean():.4f}")
    print(f"  Total improvement: {improvement:.2f}")
    print(f"  Relative improvement ratio: {improvement_pct:.2f}%\n")

# Print results
print_stats("Latency", "LLVM_instcombine_latency", "final_latency")
print_stats("Icount",  "LLVM_instcombine_icount",  "final_icount")
print_stats("Size",    "LLVM_instcombine_size",    "final_size")

