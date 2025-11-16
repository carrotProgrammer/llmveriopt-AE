# -*- coding: utf-8 -*-
from typing import Any,Tuple,List, Dict
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def render_table_I(data: Any, model: str, output_dir: str ) -> str:
    """
    Scenario A: single model + correctness -> Table I
    Show Correctness
    Save as a Markdown table file.
    Returns the file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"table_I_{model}.md")

    # --- 1. Map status values to paper categories ---
    status_col = f"status_{model}"
    if status_col not in data.columns:
        raise KeyError(f"Column {status_col} not found in dataset")

    mapping = {
        "correct": "Correct (Alive2 verified)",
        "semantic error": "Semantic Error (Not Equivalent)",
        "syntactic error": "Syntax Error (Invalid IR)",
        "alive2 can't prove": "Inconclusive",
    }

    # --- 2. Count occurrences ---
    total = len(data)
    counts = data[status_col].value_counts().to_dict()

    rows = []
    for key, label in mapping.items():
        cnt = counts.get(key, 0)
        prop = (cnt / total * 100.0) if total > 0 else 0.0
        rows.append((label, cnt, f"{prop:.1f}"))

    # --- 3. Compute “Copy of input (no optimization)” ---
    col1 = "LLVM_O0_IR"
    col2 = f"{model}_IR"
    if col1 in data.columns and col2 in data.columns:
        match = (data[col1].astype(str).str.strip() == data[col2].astype(str).str.strip())
        matched = match.sum()
        prop_match = matched / total * 100 if total > 0 else 0.0

        # Insert as sub-row under Correct
        for i, (label, cnt, prop) in enumerate(rows):
            if label.startswith("Correct"):
                rows.insert(
                    i + 1,
                    ("  └─ Copy of input (no optimization)",
                     f"({matched})",
                     f"({prop_match:.1f})")
                )
                break

    # --- 4. Render Markdown table ---
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("| Category | Count | Proportion (%) |\n")
        f.write("|----------|-------|----------------|\n")
        for label, cnt, prop in rows:
            f.write(f"| {label} | {cnt} | {prop} |\n")

    return output_file


def render_table_III(data: Any, model: str, metrics: List[str], output_dir: str = "outputs") -> str:
    """
    Scenario B: single model + one or more of {latency, ICount, BinSize, real_latency} -> Table III
    Save results as a Markdown table file.

    For each requested metric:
      - Compare the model column vs. the LLVM -O0 baseline column.
      - Count per-sample outcomes:
          Better  = model < baseline  (smaller is better)
          Worse   = model > baseline
          Tie     = model == baseline
      - Total   = number of valid samples
      - Mean Δ vs -O0:
          Algorithm: ratio of means, negative = improvement
          Formula:
              b_mean = mean(baseline values)
              v_mean = mean(model values)
              Δ = (v_mean - b_mean) / b_mean
          Interpretation:
              Δ < 0  => model is better (smaller is better)
              Δ > 0  => model is worse

    If 'real_latency' is requested, it is not implemented yet,
    and a placeholder row with 'N/A' is written.

    Returns:
        str: path to the generated Markdown file
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"table_III_{model}.md")

    col_suffix = {
        "latency": ("LLVM_O0_latency", f"{model}_latency", "Latency"),
        "ICount":  ("LLVM_O0_icount",  f"{model}_icount",  "ICount"),
        "BinSize": ("LLVM_O0_size",    f"{model}_size",    "Size"),
        # real_latency -> N/A
    }

    rows = []
    for m in metrics:
        if m == "real_latency":
            rows.append(("real_latency", model, "N/A", "N/A", "N/A", "N/A", "N/A"))
            continue
        if m not in col_suffix:
            raise ValueError(f"Unsupported metric '{m}' for Table III.")

        base_col, model_col, pretty = col_suffix[m]
        missing = [c for c in (base_col, model_col) if c not in data.columns]
        if missing:
            raise KeyError(f"Missing required column(s) for {m}: {missing}")

        b = pd.to_numeric(data[base_col], errors="coerce")
        v = pd.to_numeric(data[model_col], errors="coerce")
        mask = b.notna() & v.notna()

        total = int(mask.sum())
        if total == 0:
            rows.append((pretty, model, 0, 0, 0, 0, "N/A"))
            continue

        bb = b[mask]
        vv = v[mask]

        better = int((vv < bb).sum())
        worse  = int((vv > bb).sum())
        tie    = int((vv == bb).sum())

        b_mean = bb.mean()
        v_mean = vv.mean()
        if pd.isna(b_mean) or b_mean == 0:
            delta_str = "N/A"
        else:
            # Outcome counts: smaller values are considered better (improvement),
            # larger values are worse (regression), equal values are ties
            mean_delta = (v_mean - b_mean) / b_mean 
            delta_str = f"{mean_delta * 100:+.2f}%"

        rows.append((pretty, model, better, worse, tie, total, delta_str))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| Metric | Model | Better | Worse | Tie | Total | Mean Δ vs -O0 |\n")
        f.write("|--------|-------|--------|-------|-----|-------|----------------|\n")
        for metric_label, mdl_name, better, worse, tie, total, delta in rows:
            f.write(f"| {metric_label} | {mdl_name} | {better} | {worse} | {tie} | {total} | {delta} |\n")

    return out_path



def render_fig5(data: Any, models: List[str], metric: str, output_dir: str = "outputs") -> Tuple[str, str]:
    """
    Scenario C: multiple models + single metric -> Fig5
    Saves PNG and PDF files, returns (png_path, pdf_path).

    Supported metrics: 'latency', 'ICount', 'BinSize', 'correctness', 'real_latency'

    Rules:
      - For ratios:
          latency  -> ratio = O0 / Model (larger is better)
          ICount   -> ratio = Model / O0 (smaller is better)
          BinSize  -> ratio = Model / O0 (smaller is better)
      - Error samples (status != 'correct'): model value is replaced by its O0 value (i.e., counted as O0).
      - Geometric mean across per-sample ratios.
      - real_latency: not implemented yet -> creates a tiny placeholder file noting N/A.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- normalize metric key and pick columns/labels ----
    mkey = metric.strip()
    if mkey.lower() in ("latency",):
        base_col = "LLVM_O0_latency"
        model_suffix = "_latency"
        ylabel = "Geomean (O0 / Model)"
        ratio_mode = "latency"  # ratio = base / model
    elif mkey.lower() in ("icount", "ic", "i-count"):
        base_col = "LLVM_O0_icount"
        model_suffix = "_icount"
        ylabel = "Geomean (Model / O0)"
        ratio_mode = "smaller"  # ratio = model / base
        mkey = "ICount"
    elif mkey.lower() in ("binsize", "size", "bin_size"):
        base_col = "LLVM_O0_size"
        model_suffix = "_size"
        ylabel = "Geomean (Model / O0)"
        ratio_mode = "smaller"  # ratio = model / base
        mkey = "BinSize"
    elif mkey.lower() in ("correctness",):
        # Delegate to a correctness bar (percentage of 'correct')
        return _render_fig5_correctness(data, models, output_dir)
    elif mkey.lower() in ("real_latency", "real-latency"):
        # Not implemented yet: write a small placeholder and return paths
        base = os.path.join(output_dir, f"fig5_{mkey}_placeholder.txt")
        with open(base, "w", encoding="utf-8") as f:
            f.write("Fig5 (real_latency) is not implemented yet. N/A.\n")
        return (base, base)
    else:
        raise ValueError(f"Unsupported metric for Fig5: {metric}")

    # Validate required columns exist
    missing = [base_col] if base_col not in data.columns else []
    for mdl in models:
        mcol = f"{mdl}{model_suffix}"
        if mcol not in data.columns:
            missing.append(mcol)
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    # Build per-model geomean ratios with error-sample adjustment
    labels: List[str] = []
    values: List[float] = []

    base_vals_full = pd.to_numeric(data[base_col], errors="coerce")

    for mdl in models:
        mcol = f"{mdl}{model_suffix}"
        scol = f"status_{mdl}"

        model_vals_full = pd.to_numeric(data[mcol], errors="coerce")

        # Intersection mask (finite on both)
        mask = base_vals_full.notna() & model_vals_full.notna()

        if not mask.any():
            continue

        base_vals = base_vals_full[mask].astype(float).copy()
        model_vals = model_vals_full[mask].astype(float).copy()

        # Error samples counted as O0 (replace model with base when status != 'correct')
        if scol in data.columns:
            status = data.loc[mask, scol].astype(str).str.lower().str.strip()
            not_correct = (status != "correct")
            model_vals.loc[not_correct] = base_vals.loc[not_correct]

        # Compute ratios
        if ratio_mode == "latency":
            # ratio = base / model, larger is better
            ratio = base_vals.values / model_vals.values
        else:
            # ratio = model / base, smaller is better
            ratio = model_vals.values / base_vals.values

        # Keep finite and positive
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size == 0:
            continue

        # Geometric mean
        geomean = float(np.exp(np.mean(np.log(ratio))))
        labels.append(mdl)
        values.append(geomean)
    

    if not values:
        # Nothing to plot; write placeholder file
        placeholder = os.path.join(output_dir, f"fig5_{mkey}_empty.txt")
        with open(placeholder, "w", encoding="utf-8") as f:
            f.write(f"No valid data to plot for Fig5 metric={mkey}.\n")
        return (placeholder, placeholder)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = ax.bar(labels, values)

    for i, lab in enumerate(labels):
        if lab == "Model_Latency_3B":
            bars[i].set_color("#C44E52")      # red fill
            bars[i].set_edgecolor("#C44E52")  # red edge

    # Bar labels like "1.23×"
    bar_texts = [f"{v:.2f}×" for v in values]
    ax.bar_label(bars, labels=bar_texts, padding=3, fontsize=12)

    ax.set_ylabel(ylabel)
    # Reasonable y-limits
    ymax = max(values) * 1.15
    if mkey == "latency":
        ymin = 1.0  # latency ratios typically >= 1 when improved
        if ymax < 1.1:
            ymax = 1.1
    else:
        ymin = 0.0
        if ymax < 1.0:
            ymax = 1.0
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"fig5_{mkey.lower()}_{len(labels)}models.png")
    pdf_path = os.path.join(output_dir, f"fig5_{mkey.lower()}_{len(labels)}models.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return (png_path, pdf_path)


def _render_fig5_correctness(data: Any, models: List[str], output_dir: str) -> Tuple[str, str]:
    """
    Correctness bar chart: percent of 'correct' per model (on rows where that model's status is present).
    Saves PNG/PDF and returns the paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    labels: List[str] = []
    values: List[float] = []

    for mdl in models:
        scol = f"status_{mdl}"
        if scol not in data.columns:
            continue
        s = data[scol].astype(str).str.lower().str.strip()
        valid = s.notna()
        denom = int(valid.sum())
        if denom == 0:
            continue
        pct = float((s == "correct").sum()) / denom * 100.0
        labels.append(mdl)
        values.append(pct)

    if not values:
        placeholder = os.path.join(output_dir, f"fig5_correctness_empty.txt")
        with open(placeholder, "w", encoding="utf-8") as f:
            f.write("No valid correctness data to plot for Fig5.\n")
        return (placeholder, placeholder)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = ax.bar(labels, values)

    # === highlight Model-Latency-3B ===
    for i, lab in enumerate(labels):
        if lab == "Model_Latency_3B":    
            bars[i].set_color("#C44E52")
            bars[i].set_edgecolor("#C44E52")

    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=3, fontsize=12)

    ax.set_ylabel("Correctness (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"fig5_correctness_{len(labels)}models.png")
    pdf_path = os.path.join(output_dir, f"fig5_correctness_{len(labels)}models.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return (png_path, pdf_path)

def render_tableIII_and_fig7(
    data: Any,
    models: List[str],
    metrics: List[str],
    output_dir: str = "outputs",
) -> Dict[str, str]:
    """
    Scenario D: multiple models + multiple metrics → Table III (extended) + Fig. 7

    Changes:
    - Allow passing Model_Zero / Warmup_Model; they can be plotted even if their columns
        do not exist in `data`.
    - If `models` contains any special model (Model_Zero or Warmup_Model), then:
        * Skip table output (table_md returns "").
        * In Fig. 7: special models use built-in constants (geomean × relative to -O0, larger is better;
            correctness uses the given percentage); non-special models are still computed from `data`
            (including the “error samples fall back to O0” logic).
    - If `models` contains no special model, behavior remains the same as the original version:
        output both the table and the figure.

    Outputs:
    - If special models are present: Fig. 7 only (PNG/PDF); table_md = "".
    - If no special models are present: Table III (Markdown) + Fig. 7 (PNG/PDF).

    Column conventions in `data` (pandas.DataFrame):
    Baseline (LLVM -O0):
        - LLVM_O0_latency, LLVM_O0_icount, LLVM_O0_size
    Per-model:
        - f"{model}_latency", f"{model}_icount", f"{model}_size", f"status_{model}"

    Rules:
    - Outcome counts (table; smaller = better):
        Better = model < baseline; Worse = model > baseline; Tie = equal
    - Mean Δ vs -O0 (table):
        Δ = (mean(model) − mean(baseline)) / mean(baseline)
    - Geomean ratio (Fig. 7, all performance metrics):
        ratio = O0 / Model  (larger = better; bars start at 1)
    - Error samples: if status != "correct", replace that sample’s model value with the baseline value
    - 'real_latency' not implemented: skip
    - Correctness (right axis) is plotted only when 'correctness' is requested

    Returns:
    dict with keys: {"table_md", "fig7_png", "fig7_pdf"} (file paths or empty string)
    """
    from matplotlib.patches import Patch

    os.makedirs(output_dir, exist_ok=True)

    # ---- Supported perf metrics → (baseline_col, model_suffix, pretty_label)
    METRIC_MAP = {
        "latency": ("LLVM_O0_latency", "_latency", "Latency"),
        "ICount":  ("LLVM_O0_icount",  "_icount",  "ICount"),
        "BinSize": ("LLVM_O0_size",    "_size",    "Size"),
    }

    # ---- Normalize requested metrics
    norm_metrics: List[str] = []
    want_correctness = False
    for m in metrics:
        mk = str(m).strip()
        lk = mk.lower()
        if lk == "latency":
            norm_metrics.append("latency")
        elif lk in ("icount", "ic", "i-count"):
            norm_metrics.append("ICount")
        elif lk in ("binsize", "size", "bin_size"):
            norm_metrics.append("BinSize")
        elif lk == "correctness":
            want_correctness = True
        elif lk in ("real_latency", "real-latency"):
            # Not implemented; skip.
            continue
        else:
            raise ValueError(f"Unsupported metric in Scenario D: {mk}")

    seen = set()
    norm_metrics = [m for m in norm_metrics if not (m in seen or seen.add(m))]

    def _norm_name(s: str) -> str:
        return str(s).strip().lower().replace(" ", "_")

    SPECIALS_CANONICAL = {
        "model_zero": "Model_Zero",
        "warmup_model": "Warmup_Model",
    }
    SPECIAL_VALUES = {
        "Model_Zero":   {"latency": 1.381, "ICount": 1.373, "BinSize": 1.128, "correctness": 50.1},
        "Warmup_Model": {"latency": 1.668, "ICount": 1.629, "BinSize": 1.180, "correctness": 66.6},
    }

    norm_models: List[str] = []
    any_special = False
    for mdl in models:
        key = _norm_name(mdl)
        if key in SPECIALS_CANONICAL:
            canon = SPECIALS_CANONICAL[key]
            norm_models.append(canon)
            any_special = True
        else:
            norm_models.append(mdl)

    if any_special:
        table_md_path = ""
    else:
        table_rows = []
        for m in norm_metrics:
            base_col, suffix, pretty = METRIC_MAP[m]
            if base_col not in data.columns:
                raise KeyError(f"Missing baseline column for {m}: {base_col}")

            b_full = pd.to_numeric(data[base_col], errors="coerce")

            for mdl in norm_models:
                mdl_col = f"{mdl}{suffix}"
                if mdl_col not in data.columns:
                    raise KeyError(f"Missing model column for {m}: {mdl_col}")

                v_full = pd.to_numeric(data[mdl_col], errors="coerce")
                mask = b_full.notna() & v_full.notna()
                total = int(mask.sum())
                if total == 0:
                    table_rows.append((pretty, mdl, 0, 0, 0, 0, "N/A"))
                    continue

                bb = b_full[mask]
                vv = v_full[mask]

                # Outcome counts（smaller = better）
                better = int((vv < bb).sum())
                worse  = int((vv > bb).sum())
                tie    = int((vv == bb).sum())

                # Mean Δ vs -O0
                b_mean = float(bb.mean())
                v_mean = float(vv.mean())
                if np.isnan(b_mean) or b_mean == 0.0:
                    delta_str = "N/A"
                else:
                    mean_delta = (v_mean - b_mean) / b_mean
                    delta_str = f"{mean_delta * 100:+.2f}%"

                table_rows.append((pretty, mdl, better, worse, tie, total, delta_str))

        table_md_path = os.path.join(output_dir, "table_III_extended.md")
        with open(table_md_path, "w", encoding="utf-8") as f:
            f.write("| Metric | Model | Better | Worse | Tie | Total | Mean Δ vs -O0 |\n")
            f.write("|--------|-------|--------|-------|-----|-------|----------------|\n")
            for metric_label, mdl_name, better, worse, tie, total, delta in table_rows:
                f.write(f"| {metric_label} | {mdl_name} | {better} | {worse} | {tie} | {total} | {delta} |\n")

    perf_metrics = norm_metrics[:]  
    if not perf_metrics and not want_correctness:
        placeholder = os.path.join(output_dir, "fig7_empty.txt")
        with open(placeholder, "w", encoding="utf-8") as fp:
            fp.write("No metrics to plot for Fig7.\n")
        return {"table_md": table_md_path, "fig7_png": placeholder, "fig7_pdf": placeholder}

    values_x: Dict[str, List[float]] = {}
    correctness_pct: Dict[str, float] = {}

    for mdl in norm_models:
        if mdl in SPECIAL_VALUES:
            vals_per_model = [float(SPECIAL_VALUES[mdl][m]) for m in perf_metrics]
            values_x[mdl] = vals_per_model
            if want_correctness:
                correctness_pct[mdl] = float(SPECIAL_VALUES[mdl]["correctness"])
            continue

        vals_per_model: List[float] = []
        for m in perf_metrics:
            base_col, suffix, _pretty = METRIC_MAP[m]
            if base_col not in data.columns:
                continue
            vcol = f"{mdl}{suffix}"
            if vcol not in data.columns:
                continue

            base_vals = pd.to_numeric(data[base_col], errors="coerce")
            model_vals = pd.to_numeric(data[vcol], errors="coerce")
            mask = base_vals.notna() & model_vals.notna()
            if not mask.any():
                continue

            b = base_vals[mask].astype(float).copy()
            v = model_vals[mask].astype(float).copy()

            # fallback to O0
            scol = f"status_{mdl}"
            if scol in data.columns:
                s = data.loc[mask, scol].astype(str).str.lower().str.strip()
                v.loc[s != "correct"] = b.loc[s != "correct"]

            ratio = b.values / v.values  # O0 / Model
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            if ratio.size == 0:
                continue

            geomean = float(np.exp(np.mean(np.log(ratio))))
            vals_per_model.append(geomean)

        if vals_per_model:
            values_x[mdl] = vals_per_model

        if want_correctness:
            scol = f"status_{mdl}"
            if scol in data.columns:
                s = data[scol].astype(str).str.lower().str.strip()
                denom = int(s.notna().sum())
                if denom > 0:
                    correctness_pct[mdl] = float((s == "correct").sum()) / denom * 100.0

    any_vals = any(values_x.get(m) for m in values_x)
    if not any_vals and not correctness_pct:
        placeholder = os.path.join(output_dir, "fig7_empty.txt")
        with open(placeholder, "w", encoding="utf-8") as fp:
            fp.write("No valid data to plot for Fig7.\n")
        return {"table_md": table_md_path, "fig7_png": placeholder, "fig7_pdf": placeholder}

    # ---------------- Plot ----------------
    n_perf = len(perf_metrics)
    bar_w = 0.25
    gap = bar_w
    x_metrics = np.array([i + i * gap for i in range(n_perf)], dtype=float)
    x_correct = np.array([n_perf + n_perf * gap], dtype=float) if want_correctness else np.array([])

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax2 = ax.twinx() if want_correctness else None

    base_palette = [
        "#4C72B0", "#55A868", "#C44E52", "#8172B3",
        "#64B5CD", "#8C8C8C", "#CCB974", "#64A6A6",
    ]
    color_map: Dict[str, str] = {}
    for i, mdl in enumerate(norm_models):
        if (not any_special) and mdl == "Model_Latency_3B":
            color_map[mdl] = "#C44E52"  # red highlight
        else:
            color_map[mdl] = base_palette[i % len(base_palette)]

    # left-axis：performance
    for i_metric, metric_key in enumerate(perf_metrics):
        heights: List[float] = []
        labels_local: List[str] = []
        for mdl in norm_models:
            vals = values_x.get(mdl, [])
            if len(vals) <= i_metric:
                continue
            heights.append(vals[i_metric])
            labels_local.append(mdl)
        if not heights:
            continue

        x_base = x_metrics[i_metric]
        n_local = len(labels_local)
        for j, (mdl, h) in enumerate(zip(labels_local, heights)):
            x = x_base + (j - (n_local - 1) / 2.0) * bar_w
            ax.bar(x, h, width=bar_w, color=color_map[mdl])
            ax.annotate(f"{h:.3f}×", xy=(x, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # right-axis：correctness(optional)
    if want_correctness and ax2 is not None and len(correctness_pct) > 0:
        labels_corr = [mdl for mdl in norm_models if mdl in correctness_pct]
        n_corr = len(labels_corr)
        x_base = x_correct[0]
        for j, mdl in enumerate(labels_corr):
            h = correctness_pct[mdl]
            x = x_base + (j - (n_corr - 1) / 2.0) * bar_w
            ax2.bar(x, h, width=bar_w, color=color_map[mdl], alpha=0.90, hatch="//")
            ax2.annotate(f"{h:.1f}%", xy=(x, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("Correctness (%)")
        ax2.set_ylim(0, 100)

    ax.set_ylabel("Geomean vs -O0 (×)")
    pretty_names = [METRIC_MAP[m][2] for m in perf_metrics]
    if want_correctness:
        xticks = np.concatenate([x_metrics, x_correct])
        xticklabels = pretty_names + ["Correctness"]
    else:
        xticks = x_metrics
        xticklabels = pretty_names
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    try:
        left_max = 1.2 * max([max(values_x[m]) for m in values_x if values_x[m]])
    except ValueError:
        left_max = 2.0
    ax.set_ylim(1.0, left_max)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # legend
    legend_handles = [Patch(facecolor=color_map[m], label=m) for m in norm_models if m in color_map]
    if legend_handles:
        fig.legend(legend_handles, [h.get_label() for h in legend_handles],
                   loc="lower center", ncol=min(4, len(legend_handles)),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    fig7_pdf = os.path.join(output_dir, "fig7.pdf")
    fig7_png = os.path.join(output_dir, "fig7.png")
    plt.savefig(fig7_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(fig7_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"table_md": table_md_path, "fig7_png": fig7_png, "fig7_pdf": fig7_pdf}


def render_violin(
    data: Any,
    model: str,
    metrics: List[str],
    output_dir: str = "outputs",
) -> Tuple[str, str]:
    """
    Render violin plots comparing distributions of per-sample ratios (log2-scaled).

    Layout:
      - 3 subplots horizontally:
          [ instcombine vs -O0 ]  [ <model> vs -O0 ]  [ <model> vs instcombine ]
      - X-axis categories are the selected metrics (Latency / ICount / BinSize).

    Column conventions (must match your dataset):
      - Base columns (O0):           "LLVM_O0_<suffix>"
      - Model columns:               "<ModelName>_<suffix>"
      - Instcombine columns:         "LLVM_instcombine_<suffix>"
      - Status columns (optional):   "status_<ModelName>" with values like "correct"/"..."
        * Error samples are counted as O0: replace that model's value with the O0 value.

    Supported metrics (case-insensitive):
      - 'latency'  -> suffix "_latency"
      - 'icount'   -> suffix "_icount"  (label "ICount")
      - 'binsize'  -> suffix "_size"    (label "BinSize")

    Ratio definition for plotting (generic):
      For a comparison "A vs B", ratio = A / B and we plot log2(ratio).
      (Hence 0 on y-axis == equal; <0 means A is smaller/better when "smaller is better".)

    Returns:
      (png_path, pdf_path)

    If no valid data exists, writes a small placeholder .txt and returns its path twice.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 0) Normalize and filter metrics ----------
    # Map from various user inputs to canonical (suffix, pretty label)
    METRIC_MAP: Dict[str, Dict[str, str]] = {
        "latency":  {"suffix": "_latency", "pretty": "Latency"},
        "icount":   {"suffix": "_icount",  "pretty": "ICount"},
        "binsize":  {"suffix": "_size",    "pretty": "BinSize"},
    }

    # Normalize incoming metrics and keep only supported ones (preserve order preference)
    canonical_order = ["latency", "icount", "binsize"]
    req = []
    for m in metrics:
        mk = str(m).strip().lower()
        if mk in ("latency",):
            req.append("latency")
        elif mk in ("icount", "ic", "i-count"):
            req.append("icount")
        elif mk in ("binsize", "bin_size", "size"):
            req.append("binsize")
        else:
            logging.info("[render_violin] Ignore unsupported metric: %s", m)

    # Deduplicate while preserving original order among the known canon order
    filtered = []
    for k in canonical_order:
        if k in req and k not in filtered:
            filtered.append(k)

    if not filtered:
        placeholder = os.path.join(output_dir, f"violin_{model}_empty.txt")
        with open(placeholder, "w", encoding="utf-8") as f:
            f.write("render_violin: no supported metrics to plot.\n")
        return (placeholder, placeholder)

    # ---------- 1) Basic column names ----------
    base_prefix = "LLVM_O0"
    inst_prefix = "LLVM_instcombine"
    base_cols = {k: f"{base_prefix}{METRIC_MAP[k]['suffix']}" for k in filtered}
    inst_cols = {k: f"{inst_prefix}{METRIC_MAP[k]['suffix']}" for k in filtered}
    mdl_cols  = {k: f"{model}{METRIC_MAP[k]['suffix']}"       for k in filtered}

    # Status columns (optional)
    status_inst = f"status_{inst_prefix}"
    status_mdl  = f"status_{model}"

    # ---------- 2) Validate required columns exist (at least those present in data) ----------
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    missing = []
    for k in filtered:
        if base_cols[k] not in df.columns:
            missing.append(base_cols[k])
        if inst_cols[k] not in df.columns:
            missing.append(inst_cols[k])
        if mdl_cols[k] not in df.columns:
            missing.append(mdl_cols[k])
    if missing:
        raise KeyError(f"[render_violin] Missing required column(s): {sorted(set(missing))}")

    # ---------- 3) Helpers ----------
    def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        """Return a/b with inf handled as NaN."""
        r = a / b
        return r.replace([np.inf, -np.inf], np.nan)

    def _wlog2(s: pd.Series) -> pd.Series:
        """log2 with inf handled as NaN (no truncation)."""
        return np.log2(s.replace([np.inf, -np.inf], np.nan))

    def _apply_error_as_o0(values: pd.Series, status_col: str, base_values: pd.Series) -> pd.Series:
        """
        If status column exists and sample != 'correct', replace the model's value
        with the base (O0) value for that sample.
        """
        if status_col in df.columns:
            status = df.loc[values.index, status_col].astype(str).str.lower().str.strip()
            mask_bad = (status != "correct")
            # Align indexes and replace only where both present
            aligned_base = base_values.reindex(values.index)
            values = values.copy()
            values[mask_bad] = aligned_base[mask_bad]
        return values

    def _outcome_counts(a: pd.Series, b: pd.Series) -> Tuple[float, float, float]:
        """Share of (a<b), (a==b), (a>b) after dropping NaNs (on aligned indices)."""
        mask = a.notna() & b.notna()
        if not mask.any():
            return 0.0, 0.0, 0.0
        aa = a[mask].astype(float)
        bb = b[mask].astype(float)
        better = (aa < bb).sum()
        equal  = (aa == bb).sum()
        worse  = (aa > bb).sum()
        total  = float(better + equal + worse)
        if total <= 0:
            return 0.0, 0.0, 0.0
        return better/total, equal/total, worse/total

    # ---------- 4) Build long-form table for plotting ----------
    # Three comparisons as in your legacy figure
    comp_order = [
        "instcombine vs -O0",
        f"{model} vs -O0",
        f"{model} vs instcombine",
    ]
    metric_labels = [METRIC_MAP[k]["pretty"] for k in filtered]

    frames = []
    outcome_rows = []  # to place top/bottom % for each (Comparison, Metric)

    for k, pretty in zip(filtered, metric_labels):
        # Raw columns
        base_col = base_cols[k]
        inst_col = inst_cols[k]
        mdl_col  = mdl_cols[k]

        # Cast to numeric and build finite masks jointly for each pair we compute
        base_vals = pd.to_numeric(df[base_col], errors="coerce")
        inst_vals = pd.to_numeric(df[inst_col], errors="coerce")
        mdl_vals  = pd.to_numeric(df[mdl_col], errors="coerce")

        # Apply "error sample counted as O0" to *model* and *instcombine* where applicable
        inst_vals_adj = _apply_error_as_o0(inst_vals, status_inst, base_vals)
        mdl_vals_adj  = _apply_error_as_o0(mdl_vals,  status_mdl,  base_vals)

        # 3 pairwise comparisons for this metric
        pairs = {
            "instcombine vs -O0": (inst_vals_adj, base_vals),
            f"{model} vs -O0":    (mdl_vals_adj,  base_vals),
            f"{model} vs instcombine": (mdl_vals_adj, inst_vals_adj),
        }

        for comp, (a, b) in pairs.items():
            # Align finite
            mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
            if not mask.any():
                continue
            a2 = a[mask].astype(float)
            b2 = b[mask].astype(float)

            ratios = _safe_ratio(a2, b2)
            log2r  = _wlog2(ratios)

            # Store for violin
            frames.append(pd.DataFrame({
                "Comparison": comp,
                "Metric":     pretty,
                "Log2Ratio":  log2r.values,  # already aligned
            }))

            # Outcome proportions (on original scale a vs b)
            bc, ec, wc = _outcome_counts(a2, b2)
            outcome_rows.append({
                "Comparison": comp,
                "Metric":     pretty,
                "Better":     bc,
                "Equal":      ec,
                "Worse":      wc,
            })

    if not frames:
        placeholder = os.path.join(output_dir, f"violin_{model}_empty.txt")
        with open(placeholder, "w", encoding="utf-8") as f:
            f.write("render_violin: no valid data after filtering (NaN/inf/<=0 removed).\n")
        return (placeholder, placeholder)

    plot_df    = pd.concat(frames, ignore_index=True)
    outcome_df = pd.DataFrame(outcome_rows)

    # ---------- 5) Figure setup ----------
    sns.set_style("whitegrid")

    # Compute a sensible global y range (include 0 = same performance)
    all_vals   = plot_df["Log2Ratio"].dropna()
    global_min = float(np.nanmin(all_vals))
    global_max = float(np.nanmax(all_vals))
    global_min = min(global_min, 0.0)
    global_max = max(global_max, 0.0)

    # y tick formatter: show as "×" on original (non-log) scale
    def _fmt_y(val, _):
        try:
            return f"{2 ** val:.2f}×"
        except Exception:
            return ""

    # Colors per subplot (can be tweaked)
    palette_each = ["#4E79A7", "#59A14F", "#9C755F"]

    # Figure size: width scales mildly with number of metrics
    n_metrics = len(metric_labels)
    fig_w = max(9.0, 2.4 * n_metrics + 2.5)  # empirical; feel free to adjust
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 3.2), sharey=True)

    # First metric will receive "Better/Worse" labels; others show percentages only
    first_metric = metric_labels[0]

    # ---------- 6) Draw three violins ----------
    for ax, comp, col in zip(axes, comp_order, palette_each):
        sub = plot_df[plot_df["Comparison"] == comp]
        if sub.empty:
            ax.axis("off")
            continue

        out = outcome_df[outcome_df["Comparison"] == comp].set_index("Metric")

        # Violin plot
        sns.violinplot(
            data=sub, x="Metric", y="Log2Ratio",
            order=metric_labels, cut=0, inner="quartile", ax=ax,
            palette=[col] * len(metric_labels), zorder=2
        )

        # Unified y range + midline at 0 (1× on original scale)
        ax.set_ylim(global_min, global_max)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, zorder=3)

        # Green below / red above background bands
        ax.axhspan(global_min, 0, facecolor="#6BAE60", alpha=0.08, zorder=0)
        ax.axhspan(0, global_max, facecolor="#E15759", alpha=0.06, zorder=0)

        # Ticks, labels, title
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_y))
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_title(comp, fontsize=12, pad=8)

        # Top/Bottom percentage annotations for each metric
        ymin, ymax = ax.get_ylim()
        yspan = ymax - ymin
        y_bot = ymin + 0.02 * yspan   # bottom 2%
        y_top = ymax - 0.02 * yspan   # top 2%

        x_positions = ax.get_xticks()
        for xi, metric_label in zip(x_positions, metric_labels):
            bc = float(out.at[metric_label, "Better"]) * 100 if (metric_label in out.index and "Better" in out.columns) else 0.0
            wc = float(out.at[metric_label, "Worse"])  * 100 if (metric_label in out.index and "Worse"  in out.columns) else 0.0

            label_better = f"Better↓ {bc:.1f}%" if metric_label == first_metric else f"{bc:.1f}%"
            label_worse  = f"Worse↑ {wc:.1f}%"  if metric_label == first_metric else f"{wc:.1f}%"

            # Bottom (Better, green-ish)
            ax.text(
                xi, y_bot, label_better,
                ha="center", va="bottom", fontsize=9, color="#2E7D32",
                zorder=6, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
            )
            # Top (Worse, red-ish)
            ax.text(
                xi, y_top, label_worse,
                ha="center", va="top", fontsize=9, color="#B71C1C",
                zorder=6, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
            )

    # Only left-most y-label
    axes[0].set_ylabel("Relative Performance (log2 scale)")

    # Layout & save
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12, wspace=0.25)

    png_path = os.path.join(output_dir, f"violin_{model}_{n_metrics}metrics.png")
    pdf_path = os.path.join(output_dir, f"violin_{model}_{n_metrics}metrics.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return (png_path, pdf_path)

