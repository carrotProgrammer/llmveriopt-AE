# README

## Overview

This artifact contains all materials required to reproduce the experimental results presented in the paper.  
It includes trained SFT and GRPO LoRA models, evaluation datasets, inference pipelines, configuration files, figure-generation scripts, and a pre-computed summary table aggregating IR statistics, latency, instruction counts, and correctness metrics.  
Both full evaluation and lightweight sampling-based evaluation are supported.

---

## Platform Requirement

This artifact is developed and tested primarily on Linux.  
All evaluation scripts are POSIX-compatible shell scripts (`.sh`), and the pipeline relies on tools commonly available in Linux environments.

Other Unix-like systems (e.g., macOS, WSL2) may also work **as long as**:

- `.sh` scripts can be executed,
- Python + CUDA (optional for GPU) are correctly installed,
- the required Python packages are available for the platform.

Native Windows is **not recommended**, but WSL2 typically works.

---

## Hardware & Software Requirements

### Hardware Requirements

- **Recommended:** Nvidia GPU ≥ **32 GB** (needed for 7B/8B/32B model evaluation).  
- **Minimum:** Nvidia GPU ≥ **16 GB** (sufficient for 3B models).  
- **CPU-only mode:** Supported but **extremely slow** (may take days).  
- We performed all measurements on an **NVIDIA RTX 3090 Ti (24 GB)**.  
  Models larger than 3B may **trigger OOM** on GPUs with < 24 GB.

### Software Requirements

- Linux environment  
- Python **3.10+**  
- PyTorch  
- Transformers  
- PEFT  
- datasets  
- PyYAML  

(Exact tested versions are listed in `requirements.txt`.)

### Estimated Runtime

The following estimates are based on experiments run on an **NVIDIA RTX 3090 Ti (24 GB)**:

- **Sampling evaluation:** ~1 hour  
- **Full evaluation (model_latency_3b):** 9–12 hours  
- **Full 3B/7B/8B/32B evaluation:** multiple days on large-memory GPUs  

---

## Installation

### 1. Download the dataset from Zenodo

Download the dataset package from:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17625556.svg)](https://doi.org/10.5281/zenodo.17625556)


After downloading and extracting, extract it using:

```bash
unzip llmveriopt-datasets.zip
```
place the entire `dataset/` directory directly under the `llmveriopt-AE/` folder:

unzip llmveriopt-datasets.zip

```text
llmveriopt-AE/
└── dataset/
    └── <dataset_files_here>
```

### 2. Clone this repository

```bash
git clone https://github.com/carrotProgrammer/llmveriopt-AE
cd llmveriopt-AE
```

You should now see a structure similar to:

```text
models/
inference/
reproduce_figures/
requirements.txt
README.md
```

### 3. Install dependencies

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Main packages: `torch` with cuda, `transformers`, `peft`, `datasets`, `pyyaml`.

### Install Z3 (required for Alive2)

Alive2 requires Z3. Please install:

```bash
sudo apt update
sudo apt install z3 libz3-dev
```

This artifact already includes the necessary LLVM .so libraries under:
```text
inference/tools/llvm-project/build/lib/
```

If you need to access gated models (e.g., Llama family), authenticate first:

```bash
huggingface-cli login
```

---

## Contents

- Trained LoRA adapters (SFT + GRPO variants)
- All evaluation datasets
- Unified inference scripts
- Config files for every model variant
- Scripts to regenerate all figures in the paper
- Author-provided reference outputs
- A complete summary table used to generate plots

---

## Directory Structure

```text
llmveriopt-AE/
├── models/
├── dataset/
├── inference/
│   ├── tools 
│   ├── output
│   ├── run_inference_demo.sh
│   ├── run_inference_all.sh
│   ├── run_model_latency.sh
│   └── configs/*.yaml
├── reproduce_figures/
│   ├── summary_table/
│   └── reproduce_figures.sh
└── README.md
```

---

## Quick Start

### Sampling-based evaluation (recommended)

Install Z3 (required for Alive2)

```bash
sudo apt update
sudo apt install z3 libz3-dev
```

```bash
cd inference
chmod +x run_inference_demo.sh
./run_inference_demo.sh
```
This script runs a sampling-based evaluation on a **small subset** of the test data.  
It typically completes **within one hour** on common GPUs.

### Expected Output

Results will be created under:

```
llmveriopt-AE/inference/output/new_result/
```

Each model will have its own subdirectory containing:

```
<model_name>/
  ├── results.csv
  └── metrics.json
```

Reviewers should compare:

```
llmveriopt-AE/inference/output/new_result/summary.png
```

**against** the reference version:

```
llmveriopt-AE/inference/output/reference_results/summary.png
```

to confirm reproducibility of the evaluation pipeline.  
All IR outputs, Alive2 verification logs, and detailed metrics are stored under each `<model_name>` directory.


### Full evaluation (large GPU required)

```bash
cd inference
chmod +x run_inference_all.sh
./run_inference_all.sh
```

This script executes **all models on the full test set**.

- Requires **≥ 32GB GPU memory** (recommended: A100 / H100 GPUs).
- Smaller GPUs may encounter **out-of-memory** failures.
- Running time is significantly longer than the demo evaluation.

> Note: this script performs **inference only**.  
> It does **not** generate metrics or summary figures.

### Final model evaluation (model_latency)

```bash
cd inference
chmod +x run_model_latency.sh
./run_model_latency.sh
```

This script reproduces the inference output for the **primary model (Model_Latency) used in the paper**.

- Only performs model inference.
- Does **not** compute metrics or generate plots.

---

## Reproducing Figures

All plots in the paper can be regenerated using:

```bash
cd reproduce_figures
chmod +x reproduce_figures.sh
./reproduce_figures.sh
```

The script reads the included summary table and writes all outputs to:

```text
reproduce_figures/outputs/
```

Reviewers may compare these newly generated figures with the reference versions stored in:

```text
artifact_final/llmveriopt-AE/reproduce_figures/outputs/reference_results/
```
to confirm reproducibility of all plots in the paper.

---

## Notes for Evaluators

- **Linux** is the recommended and tested platform.
- All decoding uses **greedy decoding** → outputs are **deterministic**.
- CPU-only execution is supported but **extremely slow**.
- GPU memory below the recommended threshold will likely cause OOM errors.
- Only `run_inference_demo.sh` generates:
  - `metrics.json`
  - `summary.png`
  - Alive2 verification logs
  - IR-generation artifacts  

  The other two scripts perform **inference only**.

