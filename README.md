# README

## Overview

This artifact contains all materials required to reproduce the experimental results presented in the paper.  
It includes trained SFT and GRPO LoRA models, evaluation datasets, inference pipelines, configuration files, figure-generation scripts, and a pre-computed summary table aggregating IR statistics, latency, instruction counts, and correctness metrics.  
Both full evaluation and lightweight sampling-based evaluation are supported.

---

## Platform Requirement

This artifact is **developed and tested primarily on Linux**.  
All evaluation scripts are standard POSIX-compatible shell scripts (`.sh`), and the pipeline relies on common Linux utilities.

Other Unix-like systems (e.g., macOS, WSL2) may also work **as long as**:

- `.sh` scripts can be executed,  
- Python + CUDA (optional for GPU) are correctly installed,  
- required packages (PyTorch, Transformers, etc.) are available for that platform.

Windows (native) is **not recommended**, but running through WSL2 generally works.

---

## Installation

### 1. Download the dataset from Zenodo

Download the dataset package from:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17625555.svg)](https://doi.org/10.5281/zenodo.17625555)

After downloading and extracting, place the entire `dataset/` directory directly under the `artifact/` folder:

```text
artifact/
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
artifact/
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

Main packages: `torch`, `transformers`, `peft`, `datasets`, `pyyaml`.

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
artifact/
├── models/
├── dataset/
├── inference/
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

```bash
cd inference
chmod +x run_inference_demo.sh
./run_inference_demo.sh
```

This runs a small subset of the test data and completes within minutes on common GPUs.

### Full evaluation (large GPU required)

```bash
cd inference
chmod +x run_inference_all.sh
./run_inference_all.sh
```

Running all models across the entire test set requires ≥ 32GB GPU memory.  
Smaller devices may fail with out-of-memory errors.

### Final model evaluation (model_latency)

```bash
cd inference
chmod +x run_model_latency.sh
./run_model_latency.sh
```

This reproduces the full evaluation for the primary model used in the paper.

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

---

## Expected Outputs

Running inference scripts produces:

```text
inference/output/<model_name>/results.csv
```

The figure-generation script reproduces all plots and tables from the paper.

---

## Notes for Evaluators

- **Linux OS is required**; other platforms are not supported.
- GPU memory below the required threshold will result in OOM errors.
- All generation uses greedy decoding → deterministic output.
- CPU-only execution is possible but extremely slow.
