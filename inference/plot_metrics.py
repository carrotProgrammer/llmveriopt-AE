#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import matplotlib.pyplot as plt
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Path to RESULT_ROOT directory")
    return parser.parse_args()
# ===========================================
# Utility: geometric mean
# ===========================================
def geomean(values):
    values = [v for v in values if v is not None and v > 0]
    if not values:
        return 1.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


# ===========================================
# Parse all metrics.json
# ===========================================
def load_all_models():
    models = {}  # model_name -> list of metrics dict

    for root, dirs, files in os.walk(RESULT_ROOT):
        if "metrics.json" in files:
            model_name = os.path.basename(root)
            path = os.path.join(root, "metrics.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            models[model_name] = data
    return models


# ===========================================
# Process each model: compute correctness + geomean ratios
# ===========================================
def compute_stats(model_data):
    total = len(model_data)
    correct = 0

    lat_ratios = []
    size_ratios = []
    inst_ratios = []

    for _, row in model_data.items():
        if row.get("status") == "correct":
            correct += 1

        # latency ratio = O0 / Model
        s = row.get("src_latency")
        t = row.get("tgt_latency")
        if s and t and s > 0 and t > 0:
            lat_ratios.append(s / t)

        # size ratio = O0 / Model
        s = row.get("src_size")
        t = row.get("tgt_size")
        if s and t and s > 0 and t > 0:
            size_ratios.append(s / t)

        # inst ratio = O0 / Model
        s = row.get("src_inst")
        t = row.get("tgt_inst")
        if s and t and s > 0 and t > 0:
            inst_ratios.append(s / t)

    correctness_pct = correct / total * 100 if total > 0 else 0

    return {
        "correctness": correctness_pct,
        "latency_geomean": geomean(lat_ratios),
        "size_geomean": geomean(size_ratios),
        "inst_geomean": geomean(inst_ratios),
    }


# ===========================================
# Plot figure: one figure for all models
# ===========================================
def plot_results(model_stats):
    models = list(model_stats.keys())
    lat_vals = [model_stats[m]["latency_geomean"] for m in models]
    size_vals = [model_stats[m]["size_geomean"] for m in models]
    inst_vals = [model_stats[m]["inst_geomean"] for m in models]
    correctness = [model_stats[m]["correctness"] for m in models]

    x = list(range(len(models)))
    bar_width = 0.2

    # 每个 model 下 4 组柱子，位置对称展开
    lat_pos  = [i - 1.5 * bar_width for i in x]
    size_pos = [i - 0.5 * bar_width for i in x]
    inst_pos = [i + 0.5 * bar_width for i in x]
    corr_pos = [i + 1.5 * bar_width for i in x]

    # 图稍微缩小一点
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # 左轴：三个 geomean ratio
    bars_lat = ax1.bar(lat_pos,  lat_vals,  width=bar_width, label="Latency Geomean")
    bars_size = ax1.bar(size_pos, size_vals, width=bar_width, label="Size Geomean")
    bars_inst = ax1.bar(inst_pos, inst_vals, width=bar_width, label="Inst Geomean")

    ax1.set_ylabel("Geomean Ratio vs O0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha='right')  # 稍微少转一点

    # 右轴：正确率柱状
    ax2 = ax1.twinx()
    bars_corr = ax2.bar(
        corr_pos,
        correctness,
        width=bar_width,
        label="Correctness (%)",
        color="tab:red",
    )
    ax2.set_ylabel("Correctness (%)")
    ax2.set_ylim(0, 110)  # 给顶部数字留点空间

    # 在柱子顶部标数值
    def add_bar_labels(bars, axis, fmt="{:.2f}", dy=0.02):
        for bar in bars:
            h = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                h + dy,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    add_bar_labels(bars_lat,  ax1, fmt="{:.2f}", dy=0.03)
    add_bar_labels(bars_size, ax1, fmt="{:.2f}", dy=0.03)
    add_bar_labels(bars_inst, ax1, fmt="{:.2f}", dy=0.03)
    add_bar_labels(bars_corr, ax2, fmt="{:.1f}", dy=3.0)

    # 先自动排版，再手动多留底部和左侧空间
    plt.title("Model Performance Summary")
    plt.tight_layout()
    fig.subplots_adjust(left=0.14, bottom=0.35, top=0.88)

    # legend 放在整个图的最下方，和 x 轴文字错开
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),  # 明显在坐标轴区域外面
        ncol=4,
        frameon=False,
    )

    out_path = os.path.join(RESULT_ROOT, "summary.png")
    plt.savefig(out_path, dpi=200)
    print("[PLOT] Saved summary figure to:", out_path)


# ===========================================
# Main
# ===========================================
def main():
    models = load_all_models()

    if not models:
        print("[ERROR] No metrics.json found.")
        return

    model_stats = {
        m: compute_stats(data)
        for m, data in models.items()
    }

    plot_results(model_stats)


if __name__ == "__main__":
    args = parse_args()
    RESULT_ROOT = args.root
    main()
