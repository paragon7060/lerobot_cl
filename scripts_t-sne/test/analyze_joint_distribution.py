#!/usr/bin/env python3
"""
Raw action space에서 joint별 분포를 task별로 시각화.
action_encoder_cache.npz의 gt_actions를 사용.

CW/CCW 같은 패턴이 실제 action space에서 구분되는지 확인용.

출력:
  1. per-joint box plot (task별)
  2. per-joint delta trajectory (task별 평균 ± std)
  3. per-joint delta histogram (task별)
  4. joint 분리도 summary heatmap (task pair별 KS-test p-value)
  5. interactive HTML

Usage:
  python analyze_joint_distribution.py \
      --cache_path /home/seonho/ws3/outputs/action_emb_vis_weighted_1/action_encoder_cache.npz \
      --output_dir /home/seonho/ws3/outputs/joint_analysis \
      --joints 6 7

  # 특정 joint만, 특정 task만:
  python analyze_joint_distribution.py \
      --cache_path ... --joints 6 --tasks 5a 5b 5c 5d 5e 5f 5g 5h
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASK_TO_CATEGORY = {
    "1ext": "cabinet",
    "3a": "door",  "3b": "door",  "3c": "door",  "3d": "door",
    "5a": "bottle", "5b": "bottle", "5c": "bottle", "5d": "bottle",
    "5e": "bottle", "5f": "bottle", "5g": "bottle", "5h": "bottle",
}


def parse_args():
    p = argparse.ArgumentParser(description="Per-joint distribution analysis")
    p.add_argument("--cache_path", type=str, required=True,
                   help="action_encoder_cache.npz path")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir. Defaults to <cache_dir>/joint_analysis/")
    p.add_argument("--joints", type=int, nargs="*", default=None,
                   help="Joint indices to analyze. Default: all")
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="Task labels to include. Default: all")
    return p.parse_args()


def main():
    args = parse_args()

    cache_path = Path(args.cache_path)
    out_dir = Path(args.output_dir) if args.output_dir else cache_path.parent / "joint_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"[1/4] Loading cache: {cache_path}")
    npz = np.load(cache_path, allow_pickle=True)
    gt_actions = npz["gt_actions"]          # (N, T, action_dim)
    task_labels = [str(x) for x in npz["task_labels"]]
    N, T, D = gt_actions.shape
    print(f"  N={N}, T={T}, action_dim={D}")

    # filter tasks
    if args.tasks:
        mask = np.array([t in args.tasks for t in task_labels])
        gt_actions = gt_actions[mask]
        task_labels = [t for t, m in zip(task_labels, mask) if m]
        N = len(task_labels)
        print(f"  Filtered to tasks={args.tasks}: N={N}")

    joint_indices = args.joints if args.joints else list(range(D))
    task_ids = sorted(set(task_labels))
    task_arr = np.array(task_labels)
    print(f"  Tasks: {task_ids}")
    print(f"  Joints: {joint_indices}")

    # ── Compute deltas ───────────────────────────────────────────────────────
    # per-step velocity: (N, T-1, D)
    deltas = gt_actions[:, 1:, :] - gt_actions[:, :-1, :]
    # delta_total: (N, D)
    delta_total = gt_actions[:, -1, :] - gt_actions[:, 0, :]
    # delta_mean: (N, D)
    delta_mean = deltas.mean(axis=1)

    # ── 2/4: Per-joint box plots ─────────────────────────────────────────────
    print(f"\n[2/4] Per-joint box plots ...")
    n_joints = len(joint_indices)

    # delta_mean box plot
    fig, axes = plt.subplots(1, n_joints, figsize=(5 * n_joints, 6), squeeze=False)
    axes = axes[0]
    for ax_i, ji in enumerate(joint_indices):
        ax = axes[ax_i]
        data = [delta_mean[task_arr == tid, ji] for tid in task_ids]
        bp = ax.boxplot(data, labels=task_ids, patch_artist=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(task_ids)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(f"joint {ji}\ndelta_mean", fontsize=11)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Per-joint delta_mean distribution by task", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_delta_mean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved boxplot_delta_mean.png")

    # delta_total box plot
    fig, axes = plt.subplots(1, n_joints, figsize=(5 * n_joints, 6), squeeze=False)
    axes = axes[0]
    for ax_i, ji in enumerate(joint_indices):
        ax = axes[ax_i]
        data = [delta_total[task_arr == tid, ji] for tid in task_ids]
        bp = ax.boxplot(data, labels=task_ids, patch_artist=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(task_ids)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(f"joint {ji}\ndelta_total", fontsize=11)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Per-joint delta_total distribution by task", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_delta_total.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved boxplot_delta_total.png")

    # ── 3/4: Per-joint trajectory (mean ± std over time) ─────────────────────
    print(f"\n[3/4] Per-joint trajectories ...")
    cmap = plt.cm.tab20(np.linspace(0, 1, len(task_ids)))
    task_colors = {tid: cmap[i] for i, tid in enumerate(task_ids)}

    for ji in joint_indices:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: raw value over time
        ax = axes[0]
        for tid in task_ids:
            vals = gt_actions[task_arr == tid, :, ji]  # (n_samples, T)
            m = vals.mean(axis=0)
            s = vals.std(axis=0)
            x = np.arange(T)
            ax.plot(x, m, label=tid, color=task_colors[tid], linewidth=1.5)
            ax.fill_between(x, m - s, m + s, color=task_colors[tid], alpha=0.15)
        ax.set_title(f"joint {ji} — raw value", fontsize=11)
        ax.set_xlabel("timestep")
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(True, alpha=0.2)

        # Right: per-step delta over time
        ax = axes[1]
        for tid in task_ids:
            d = deltas[task_arr == tid, :, ji]  # (n_samples, T-1)
            m = d.mean(axis=0)
            s = d.std(axis=0)
            x = np.arange(T - 1)
            ax.plot(x, m, label=tid, color=task_colors[tid], linewidth=1.5)
            ax.fill_between(x, m - s, m + s, color=task_colors[tid], alpha=0.15)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"joint {ji} — per-step delta", fontsize=11)
        ax.set_xlabel("timestep")
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(True, alpha=0.2)

        fig.suptitle(f"Joint {ji} trajectory by task", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectory_joint{ji}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved trajectory_joint{ji}.png")

    # ── 4/4: Per-joint histogram + KS-test ───────────────────────────────────
    print(f"\n[4/4] Per-joint histograms & separation metrics ...")
    from scipy import stats

    # KS-test p-value matrix: for each joint, all task pairs
    n_tasks = len(task_ids)
    for ji in joint_indices:
        n_cols = min(4, n_tasks)
        n_rows = (n_tasks + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        for ti, tid in enumerate(task_ids):
            ax = axes_flat[ti]
            vals = delta_mean[task_arr == tid, ji]
            ax.hist(vals, bins=40, alpha=0.7, color=task_colors[tid], edgecolor="black",
                    linewidth=0.3)
            ax.set_title(f"{tid} (n={len(vals)})", fontsize=10)
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel(f"joint {ji} delta_mean")
            ax.grid(True, alpha=0.2)
        for ti in range(len(task_ids), len(axes_flat)):
            axes_flat[ti].set_visible(False)

        fig.suptitle(f"Joint {ji} delta_mean histogram by task", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"histogram_joint{ji}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved histogram_joint{ji}.png")

    # KS-test summary
    print(f"\n  ── KS-test separation summary (delta_mean) ──")
    print(f"  {'joint':>6}  {'task_A':>6} vs {'task_B':>6}  {'KS-stat':>8}  {'p-value':>10}  sep?")
    print(f"  {'─'*60}")

    summary_rows = []
    for ji in joint_indices:
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                a = delta_mean[task_arr == task_ids[i], ji]
                b = delta_mean[task_arr == task_ids[j], ji]
                ks_stat, p_val = stats.ks_2samp(a, b)
                sep = "✓" if p_val < 0.01 else "✗"
                summary_rows.append((ji, task_ids[i], task_ids[j], ks_stat, p_val, sep))
                if ks_stat > 0.3:  # only print notable separations
                    print(f"  {ji:>6}  {task_ids[i]:>6} vs {task_ids[j]:>6}"
                          f"  {ks_stat:>8.3f}  {p_val:>10.2e}  {sep}")

    # Save summary CSV
    csv_path = out_dir / "ks_test_summary.csv"
    with open(csv_path, "w") as f:
        f.write("joint,task_a,task_b,ks_stat,p_value,separated\n")
        for ji, ta, tb, ks, pv, sep in summary_rows:
            f.write(f"{ji},{ta},{tb},{ks:.4f},{pv:.2e},{sep}\n")
    print(f"\n  saved ks_test_summary.csv")

    print(f"\n✅ Done. All outputs → {out_dir}")


if __name__ == "__main__":
    main()
