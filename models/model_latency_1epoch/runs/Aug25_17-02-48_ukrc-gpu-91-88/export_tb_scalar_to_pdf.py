import os, glob, csv
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# ========= 配置区域 =========
LOGDIR = "/workspace/GRPO/grpo_vllm_train/outputs/deepseek_r1_distill_qwen_3b/v10_latency_reward_and_prompt_without_CoT/grpo_round1/runs/Aug25_17-02-48_ukrc-gpu-91-88"
TAG = "train/rewards/latency_speedup_reward_func/mean"   # 你的曲线名
OUTPUT_PDF = "latency_reward_curve.pdf"
OUTPUT_CSV = "latency_reward_curve.csv"

# 可选：指数滑动平均平滑窗口（0 表示不平滑，常用 0.9）
EMA_ALPHA = 0.0

# ========= 工具函数 =========
def load_scalars_from_events(logdir, tag):
    event_files = sorted(glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True),
                         key=lambda p: os.path.getmtime(p))
    records = []
    for f in event_files:
        try:
            ea = event_accumulator.EventAccumulator(f)
            ea.Reload()
            if tag in ea.Tags().get("scalars", []):
                for ev in ea.Scalars(tag):
                    records.append((ev.wall_time, ev.step, ev.value))
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")
    records.sort(key=lambda x: x[0])  # 按 wall_time 排序
    return records

def stitch_steps(records):
    stitched = []
    step_offset = 0
    last_step_global = -1

    for wt, step, val in records:
        step_global = step + step_offset
        if step_global < last_step_global:
            step_offset = last_step_global + 1 - step
            step_global = step + step_offset
        stitched.append((wt, step_global, val))
        if step_global > last_step_global:
            last_step_global = step_global

    dedup = {}
    for wt, s, v in stitched:
        dedup[s] = (wt, v)
    steps = sorted(dedup.keys())
    vals = [dedup[s][1] for s in steps]
    return steps, vals

def ema_smooth(values, alpha):
    if alpha <= 0 or alpha >= 1:
        return values
    y = []
    m = None
    for v in values:
        m = v if m is None else alpha * m + (1 - alpha) * v
        y.append(m)
    return y

# ========= 主流程 =========
def main():
    recs = load_scalars_from_events(LOGDIR, TAG)
    if not recs:
        raise RuntimeError(f"No scalar data found for tag: {TAG}")

    steps, values = stitch_steps(recs)
    values_smooth = ema_smooth(values, 0.95)

    # 导出 CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "value_raw", "value_ema_0.95"])
        for s, v_raw, v_smooth in zip(steps, values, values_smooth):
            w.writerow([s, v_raw, v_smooth])
    print(f"[OK] CSV saved -> {os.path.abspath(OUTPUT_CSV)}")

    # 画图
    plt.figure(figsize=(6.0, 3.0))
    plt.plot(steps, values, linestyle="--", color="gray", alpha=0.3, label="Raw Latency Speedup Reward")
    plt.plot(steps, values_smooth, linewidth=2, color="C0", label="EMA (0.95) Latency Speedup Reward")

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Latency Speedup Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF)
    print(f"[OK] PDF saved -> {os.path.abspath(OUTPUT_PDF)}")

if __name__ == "__main__":
    main()
