#!/usr/bin/env python3
"""
Joint visualization v2: action_encoder vs VLM backbone (same N samples).

Features:
  - Two-click distance mode:
      1st click → select point A (blue ring) + play video A
      2nd click → select point B (green ring) + play video B
                  → show cosine / L2 distance in:
                    * full feature space  (action encoder latent, backbone latent)
                    * PCA-50 space        (both spaces)
      3rd click / same point → reset
  - Pairwise distance matrices precomputed in Python (upper triangle, float32, base64)
    and embedded in HTML for O(1) lookup in browser
  - Episode progress displayed instead of raw frame index
  - A/B video clips side-by-side per camera
  - --no_zscore: skip z-score before PCA (preserves inter-joint variance scale)

Usage:
  python src/lerobot/scripts/visualize_joint_embedding_v2.py \
      --cache_dir /home/seonho/ws3/outputs/action_emb_vis_2048 \
      --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
      --action_agg flatten \
      --no_zscore \
      --output_dir /home/seonho/ws3/outputs/action_emb_vis_2048/joint_v2_flatten_noz
"""

import argparse
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

TASK_TO_CATEGORY = {
    "1ext": "cabinet",
    "3a": "door",  "3b": "door",  "3c": "door",  "3d": "door",
    "5a": "bottle", "5b": "bottle", "5c": "bottle", "5d": "bottle",
    "5e": "bottle", "5f": "bottle", "5g": "bottle", "5h": "bottle",
}

CATEGORY_COLORS = {
    "cabinet": "#1f77b4",
    "door":    "#ff7f0e",
    "bottle":  "#2ca02c",
}

DEFAULT_CAM_KEYS = [
    "observation.images.right_shoulder",
    "observation.images.wrist",
]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir",      type=str, required=True)
    p.add_argument("--dataset_root",   type=str, default=None)
    p.add_argument("--output_dir",     type=str, default=None)
    p.add_argument("--action_agg",     type=str, default="delta_mean",
                   choices=["mean", "max", "last", "first_last", "delta_total", "delta_mean",
                            "flatten", "first_mid_last", "raw_flatten", "raw_delta_mean",
                            "raw_delta_total"])
    p.add_argument("--backbone_agg",   type=str, default="mask_mean",
                   choices=["mask_mean", "mean", "max", "first", "last"])
    p.add_argument("--perplexity",     type=int, default=30)
    p.add_argument("--tsne_n_iter",    type=int, default=1000)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--cam_keys",       type=str, nargs="*", default=DEFAULT_CAM_KEYS)
    p.add_argument("--clip_size",      type=str, default="320x240")
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--skip_clips",     action="store_true")
    p.add_argument("--no_zscore",      action="store_true",
                   help="Skip z-score normalization before PCA (preserves inter-joint variance scale)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def agg_action(feats: np.ndarray, method: str,
               gt_actions: np.ndarray = None) -> np.ndarray:
    if method.startswith("raw_"):
        if gt_actions is None:
            raise ValueError(f"method='{method}' requires gt_actions")
        t = torch.from_numpy(gt_actions)
        N, T, D = t.shape
        if method == "raw_flatten":    out = t.reshape(N, T * D)
        elif method == "raw_delta_mean": out = (t[:, 1:, :] - t[:, :-1, :]).mean(dim=1)
        elif method == "raw_delta_total": out = t[:, -1, :] - t[:, 0, :]
        else: raise ValueError(method)
        return out.numpy()

    t = torch.from_numpy(feats)
    N, T, D = t.shape
    if method == "mean":           out = t.mean(dim=1)
    elif method == "max":          out = t.max(dim=1).values
    elif method == "last":         out = t[:, -1, :]
    elif method == "first_last":   out = (t[:, 0, :] + t[:, -1, :]) / 2.0
    elif method == "delta_total":  out = t[:, -1, :] - t[:, 0, :]
    elif method == "delta_mean":   out = (t[:, 1:, :] - t[:, :-1, :]).mean(dim=1)
    elif method == "flatten":      out = t.reshape(N, T * D)
    elif method == "first_mid_last":
        mid = T // 2
        out = torch.cat([t[:, 0, :], t[:, mid, :], t[:, -1, :]], dim=-1)
    else: raise ValueError(method)
    return out.numpy()


def agg_backbone(feats_list: list, masks_list: list, method: str) -> np.ndarray:
    parts = []
    for f, m in zip(feats_list, masks_list):
        if method == "mask_mean":
            m_f = m.unsqueeze(-1).float()
            agg = (f * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)
        elif method == "mean":  agg = f.mean(dim=1)
        elif method == "max":
            m_f = m.unsqueeze(-1).float()
            agg = (f * m_f + (1.0 - m_f) * (-1e9)).max(dim=1).values
        elif method == "first": agg = f[:, 0, :]
        elif method == "last":
            lengths = m.sum(dim=1).long().clamp(min=1)
            idx = (lengths-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, f.shape[-1])
            agg = f.gather(dim=1, index=idx).squeeze(1)
        else: raise ValueError(method)
        parts.append(agg)
    return torch.cat(parts, dim=0).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Dimensionality reduction
# ══════════════════════════════════════════════════════════════════════════════

def pca_preprocess(X: np.ndarray, n_components: int = 50, zscore: bool = True):
    from sklearn.decomposition import PCA
    var = X.var(axis=0)
    X_c = X[:, var > 1e-10]
    if X_c.shape[1] == 0:
        return None
    if zscore:
        mean = X_c.mean(axis=0); std = X_c.std(axis=0).clip(min=1e-10)
        X_c = (X_c - mean) / std
    n = min(n_components, X_c.shape[0] - 1, X_c.shape[1])
    out = PCA(n_components=n, random_state=0).fit_transform(X_c)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def run_tsne(X: np.ndarray, perplexity: int, n_iter: int, seed: int,
             label: str = "", zscore: bool = True):
    """Returns (X_pca50, X_2d) or (None, None) on failure."""
    from sklearn.manifold import TSNE
    import sklearn
    X_pca = pca_preprocess(X, zscore=zscore)
    if X_pca is None:
        print(f"  [skip] {label}: all features constant")
        return None, None
    if X_pca.std(axis=0)[0] < 1e-12:
        X_pca = X_pca + np.random.RandomState(seed).randn(*X_pca.shape) * 1e-6
    perplexity = min(perplexity, max(5.0, (X_pca.shape[0]-1)/3.0 - 1))
    kw = dict(n_components=2, perplexity=perplexity, random_state=seed, verbose=0)
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
        kw["max_iter"] = n_iter
    else:
        kw["n_iter"] = n_iter
    zscore_str = "zscore" if zscore else "no_zscore"
    print(f"  [t-SNE] {label}  shape={X_pca.shape}  perp={perplexity:.0f}  [{zscore_str}] ...")
    X_2d = TSNE(**kw).fit_transform(X_pca)
    return X_pca, X_2d


# ══════════════════════════════════════════════════════════════════════════════
# Pairwise distance matrices
# ══════════════════════════════════════════════════════════════════════════════

def _upper_tri_b64(mat: np.ndarray) -> str:
    """(N, N) float → upper triangle (i<j) → float32 bytes → base64 string."""
    N = mat.shape[0]
    idx = np.triu_indices(N, k=1)
    return base64.b64encode(mat[idx].astype(np.float32).tobytes()).decode("ascii")


def compute_dist_matrices(
    X_action_full:    np.ndarray,   # (N, D_action)  — full aggregated action features
    X_backbone_full:  np.ndarray,   # (N, D_bb)      — full aggregated backbone features
    X_pca_action:     np.ndarray,   # (N, 50)
    X_pca_backbone:   np.ndarray,   # (N, 50)
) -> dict:
    """
    Compute pairwise cosine & L2 distances for full-space and PCA-50 space.
    Returns dict of base64-encoded upper-triangle float32 arrays.

    Keys: action_full_cos, action_full_l2,
          action_pca_cos,  action_pca_l2,
          bb_full_cos,     bb_full_l2,
          bb_pca_cos,      bb_pca_l2
    """
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    result = {}
    spaces = [
        ("action_full", X_action_full),
        ("action_pca",  X_pca_action),
        ("bb_full",     X_backbone_full),
        ("bb_pca",      X_pca_backbone),
    ]
    for name, X in spaces:
        print(f"  [dist] {name}  shape={X.shape} ...", end=" ", flush=True)
        cos = cosine_distances(X)
        l2  = euclidean_distances(X)
        result[f"{name}_cos"] = _upper_tri_b64(cos)
        result[f"{name}_l2"]  = _upper_tri_b64(l2)
        # Report estimated size
        nbytes = len(result[f"{name}_cos"]) + len(result[f"{name}_l2"])
        print(f"~{nbytes/1e6:.1f} MB")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Video clip extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_video_clips(dataset_root, frame_indices, cam_keys,
                        action_horizon, out_dir, fps=10, output_size=(320, 240)):
    import av
    import pandas as pd

    root = Path(dataset_root)
    W, H = output_size

    print("  [clips] Loading dataset metadata ...")
    data_df = pd.read_parquet(
        root / "data/chunk-000/file-000.parquet",
        columns=["index", "episode_index", "timestamp"],
    ).set_index("index")
    ep_df = pd.read_parquet(
        root / "meta/episodes/chunk-000/file-000.parquet"
    ).set_index("episode_index")

    result = {}
    for cam_key in cam_keys:
        cam_short = cam_key.split(".")[-1]
        cam_dir   = out_dir / cam_short
        cam_dir.mkdir(parents=True, exist_ok=True)
        result[cam_short] = cam_dir

        todo = [(i, gi) for i, gi in enumerate(frame_indices)
                if not (cam_dir / f"{i:04d}.mp4").exists()]
        if not todo:
            print(f"  [clips] {cam_short}: all {len(frame_indices)} clips already exist")
            continue
        print(f"  [clips] {cam_short}: extracting {len(todo)} clips ...")

        col_ci = f"videos/{cam_key}/chunk_index"
        col_fi = f"videos/{cam_key}/file_index"
        col_ft = f"videos/{cam_key}/from_timestamp"

        groups = defaultdict(list)
        for sample_idx, global_idx in todo:
            row = data_df.loc[global_idx]
            ep  = int(row["episode_index"]); ts = float(row["timestamp"])
            ep_r = ep_df.loc[ep]
            ci = int(ep_r[col_ci]); fi = int(ep_r[col_fi]); ft = float(ep_r[col_ft])
            groups[(ci, fi)].append((sample_idx, ft + ts))

        n_done = 0
        for (ci, fi), samples in sorted(groups.items()):
            vpath = root / f"videos/{cam_key}/chunk-{ci:03d}/file-{fi:03d}.mp4"
            samples_sorted = sorted(samples, key=lambda x: x[1])
            with av.open(str(vpath)) as vc:
                stream = vc.streams.video[0]
                tb = float(stream.time_base)
                for sample_idx, abs_ts in samples_sorted:
                    out_file = cam_dir / f"{sample_idx:04d}.mp4"
                    pts = int(abs_ts / tb)
                    vc.seek(pts, stream=stream, backward=True)
                    raw_frames = []
                    for pkt in vc.demux(stream):
                        for frm in pkt.decode():
                            fts = float(frm.pts * tb) if frm.pts is not None else 0.0
                            if fts < abs_ts - 1.0 / fps: continue
                            raw_frames.append(frm.to_ndarray(format="rgb24"))
                            if len(raw_frames) >= action_horizon: break
                        if len(raw_frames) >= action_horizon: break
                    if not raw_frames: continue
                    from PIL import Image as PILImage
                    frames_resized = []
                    for rf in raw_frames[:action_horizon]:
                        pil = PILImage.fromarray(rf)
                        if pil.size != (W, H): pil = pil.resize((W, H), PILImage.LANCZOS)
                        frames_resized.append(np.array(pil))
                    import io as _io
                    buf = _io.BytesIO()
                    with av.open(buf, mode="w", format="mp4") as oc:
                        ost = oc.add_stream("libx264", rate=fps)
                        ost.width = W; ost.height = H; ost.pix_fmt = "yuv420p"
                        ost.options = {"crf": "28", "preset": "ultrafast"}
                        for rf in frames_resized:
                            avf = av.VideoFrame.from_ndarray(rf, format="rgb24")
                            for pkt2 in ost.encode(avf): oc.mux(pkt2)
                        for pkt2 in ost.encode(None): oc.mux(pkt2)
                    with open(out_file, "wb") as fout:
                        fout.write(buf.getvalue())
                    n_done += 1
                    if n_done % 100 == 0:
                        print(f"  [clips] {cam_short}: {n_done}/{len(todo)} ...", end="\r")
        print(f"  [clips] {cam_short}: done ({len(todo)} clips)          ")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Static matplotlib plot
# ══════════════════════════════════════════════════════════════════════════════

def _task_cmap(task_ids):
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(task_ids), 1))
    return {t: cmap(i) for i, t in enumerate(task_ids)}


def _scatter_task(ax, X2d, task_labels, s=15, alpha=0.7):
    task_ids = sorted(set(task_labels)); cmap = _task_cmap(task_ids); arr = np.array(task_labels)
    for tid in task_ids:
        mask = arr == tid
        ax.scatter(X2d[mask,0], X2d[mask,1], c=[cmap[tid]], label=tid,
                   alpha=alpha, s=s, linewidths=0)
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.2)


def _scatter_category(ax, X2d, task_labels, s=15, alpha=0.7):
    cats = np.array([TASK_TO_CATEGORY[t] for t in task_labels])
    for cat, color in CATEGORY_COLORS.items():
        mask = cats == cat
        ax.scatter(X2d[mask,0], X2d[mask,1], c=color, label=cat,
                   alpha=alpha, s=s, linewidths=0)
    ax.legend(fontsize=10, markerscale=2); ax.grid(True, alpha=0.2)


def plot_combined_static(X2d_action, X2d_backbone, task_labels,
                         action_agg, backbone_agg, perplexity, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    pairs = [(X2d_action, f"action_encoder  ({action_agg})"),
             (X2d_backbone, f"backbone  ({backbone_agg})")]
    for col, (X2d, col_title) in enumerate(pairs):
        ax = axes[0][col]; _scatter_task(ax, X2d, task_labels)
        ax.set_title(f"{col_title}\nby task_id", fontsize=11)
        ax = axes[1][col]; _scatter_category(ax, X2d, task_labels)
        ax.set_title(f"{col_title}\nby category", fontsize=11)
    sup = (f"action_encoder ({action_agg})  vs  backbone ({backbone_agg})"
           f"  |  t-SNE perp={perplexity}  |  N={len(task_labels)}")
    fig.suptitle(sup, fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive HTML
# ══════════════════════════════════════════════════════════════════════════════

def make_joint_html(
    X2d_action:      np.ndarray,
    X2d_backbone:    np.ndarray,
    X_pca_action:    np.ndarray,    # (N, 50)  — kept for PCA-50 distance display
    X_pca_backbone:  np.ndarray,    # (N, 50)
    task_labels:     list,
    frame_indices:   list,
    episode_progress: list,         # (N,) float in [0, 1]
    action_agg:      str,
    backbone_agg:    str,
    perplexity:      int,
    out_path:        Path,
    clips_dirs:      dict = None,
    zscore:          bool = True,
    dist_b64:        dict = None,   # precomputed distance matrices (base64)
):
    N        = len(task_labels)
    task_ids = sorted(set(task_labels))
    cats     = [TASK_TO_CATEGORY.get(t, "?") for t in task_labels]

    cmap_mpl   = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(task_ids), 1))
    task_color = {t: f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
                  for t, (r, g, b, _) in
                  {t: cmap_mpl(i) for i, t in enumerate(task_ids)}.items()}

    def _make_traces(X2d):
        traces = []
        arr = np.array(task_labels)
        for tid in task_ids:
            idx = np.where(arr == tid)[0].tolist()
            traces.append({
                "type": "scatter", "x": X2d[idx,0].tolist(), "y": X2d[idx,1].tolist(),
                "mode": "markers", "name": tid,
                "marker": {"color": task_color[tid], "size": 7, "opacity": 0.85},
                "customdata": [[task_labels[i], cats[i], i, frame_indices[i]] for i in idx],
                "hovertemplate": (
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "frame: %{customdata[3]}<br>sample: %{customdata[2]}<extra></extra>"
                ),
            })
        # A highlight (blue ring)
        traces.append({
            "type": "scatter", "x": [], "y": [], "mode": "markers",
            "name": "A", "showlegend": False, "hoverinfo": "skip",
            "marker": {"symbol": "circle-open", "size": 22, "color": "#2196F3",
                       "line": {"color": "#2196F3", "width": 3}},
        })
        # B highlight (green ring)
        traces.append({
            "type": "scatter", "x": [], "y": [], "mode": "markers",
            "name": "B", "showlegend": False, "hoverinfo": "skip",
            "marker": {"symbol": "circle-open", "size": 22, "color": "#4CAF50",
                       "line": {"color": "#4CAF50", "width": 3}},
        })
        return traces

    action_traces   = _make_traces(X2d_action)
    backbone_traces = _make_traces(X2d_backbone)
    n_task_traces   = len(task_ids)

    action_xy   = X2d_action.tolist()
    backbone_xy = X2d_backbone.tolist()
    pca_act_js  = X_pca_action.tolist()
    pca_bb_js   = X_pca_backbone.tolist()

    shared_layout = {
        "height": 600, "hovermode": "closest",
        "xaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)", "title": "dim-1"},
        "yaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)", "title": "dim-2"},
        "legend": {"x": 1.01, "y": 1, "xanchor": "left", "font": {"size": 11}},
        "margin": {"l": 50, "r": 160, "t": 50, "b": 50},
    }
    action_layout   = dict(shared_layout, title=f"action_encoder  ({action_agg})")
    backbone_layout = dict(shared_layout, title=f"backbone  ({backbone_agg})")

    # ── Video panel ───────────────────────────────────────────────────────────
    has_clips  = bool(clips_dirs)
    cam_shorts = list(clips_dirs.keys()) if has_clips else []

    video_panel_html = ""
    if has_clips:
        for cs in cam_shorts:
            video_panel_html += f"""
<div class="cam-group">
  <p class="cam-label">{cs}</p>
  <div class="cam-row">
    <div class="vid-box">
      <p class="vid-ab-label" style="color:#2196F3;">A</p>
      <video id="vid_A_{cs}" autoplay loop muted playsinline width="100%"></video>
    </div>
    <div class="vid-box">
      <p class="vid-ab-label" style="color:#4CAF50;">B</p>
      <video id="vid_B_{cs}" autoplay loop muted playsinline width="100%"></video>
    </div>
  </div>
</div>"""

    cam_shorts_js = json.dumps(cam_shorts)
    title = (f"action_encoder ({action_agg})  vs  backbone ({backbone_agg})"
             f"  |  t-SNE perp={perplexity}  |  N={N}"
             + ("  |  no_zscore" if not zscore else ""))

    # ── Distance matrix JS embedding ─────────────────────────────────────────
    has_dist = bool(dist_b64)
    dist_b64_json = json.dumps(dist_b64) if has_dist else "null"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 8px; background: #f5f5f5; }}
  h3 {{ text-align: center; margin: 6px 0 10px; font-size: 14px; color: #333; }}
  #plots-row {{ display: flex; gap: 8px; }}
  #plots-row > div {{ flex: 1; min-width: 0; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
  #bottom-row {{ display: flex; gap: 12px; margin-top: 10px; align-items: flex-start; }}
  #video-panel {{
    display: flex; gap: 18px; padding: 10px 14px;
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    flex: 3; min-height: 80px; flex-wrap: wrap;
  }}
  .cam-group {{ display: flex; flex-direction: column; align-items: center; }}
  .cam-label {{ margin: 0 0 4px; font-size: 12px; font-weight: bold; color: #333; }}
  .cam-row {{ display: flex; gap: 8px; }}
  .vid-box {{ text-align: center; }}
  .vid-ab-label {{ margin: 0 0 2px; font-size: 13px; font-weight: bold; }}
  .vid-box video {{
    width: 160px; height: auto;
    border: 2px solid #ccc; border-radius: 4px; background: #111;
  }}
  #right-panel {{ flex: 1; display: flex; flex-direction: column; gap: 8px; min-width: 260px; }}
  #info-box {{
    padding: 10px 14px; background: #fff;
    border: 1px solid #ddd; border-radius: 6px;
    font-size: 13px; color: #444; white-space: pre-line;
  }}
  #dist-panel {{
    padding: 10px 14px; background: #fff;
    border: 1px solid #ddd; border-radius: 6px;
    font-size: 12px; color: #333;
  }}
  #dist-panel h4 {{ margin: 0 0 8px; font-size: 13px; color: #555; }}
  .dist-section {{ margin-bottom: 10px; }}
  .dist-section-title {{
    font-weight: bold; font-size: 11px; color: #888;
    text-transform: uppercase; margin-bottom: 4px;
    border-bottom: 1px solid #eee; padding-bottom: 2px;
  }}
  .dist-row {{ display: flex; justify-content: space-between; margin: 3px 0; }}
  .dist-label {{ color: #666; }}
  .dist-value {{ font-weight: bold; font-family: monospace; }}
  .dist-value.full {{ color: #1565C0; }}
  .dist-value.pca  {{ color: #888; }}
  #mode-hint {{ text-align: center; font-size: 12px; color: #888; margin: 4px 0; }}
</style>
</head>
<body>
<h3>{title}</h3>
<div id="mode-hint">클릭 1: A 선택 (파랑) &nbsp;|&nbsp; 클릭 2: B 선택 + 거리 계산 (초록) &nbsp;|&nbsp; 같은 점 클릭: 초기화</div>
<div id="plots-row">
  <div id="plot_action"></div>
  <div id="plot_backbone"></div>
</div>
<div id="bottom-row">
  <div id="video-panel">
{video_panel_html}
    <span id="vid-hint" style="color:#aaa;font-style:italic;font-size:13px;">
      &#8592; click a point to play the action clip
    </span>
  </div>
  <div id="right-panel">
    <div id="info-box"><span style="color:#aaa;font-style:italic;">click a point for details</span></div>
    <div id="dist-panel" style="display:none;">
      <h4>&#128207; Feature Distance</h4>
      <div class="dist-section">
        <div class="dist-section-title">Action Encoder ({action_agg})</div>
        <div class="dist-row">
          <span class="dist-label">Cosine (full space)</span>
          <span class="dist-value full" id="d_act_full_cos">—</span>
        </div>
        <div class="dist-row">
          <span class="dist-label">L2 &nbsp;&nbsp;&nbsp;&nbsp; (full space)</span>
          <span class="dist-value full" id="d_act_full_l2">—</span>
        </div>
        <div class="dist-row">
          <span class="dist-label">Cosine (PCA-50)</span>
          <span class="dist-value pca" id="d_act_pca_cos">—</span>
        </div>
        <div class="dist-row">
          <span class="dist-label">L2 &nbsp;&nbsp;&nbsp;&nbsp; (PCA-50)</span>
          <span class="dist-value pca" id="d_act_pca_l2">—</span>
        </div>
      </div>
      <div class="dist-section">
        <div class="dist-section-title">Backbone ({backbone_agg})</div>
        <div class="dist-row">
          <span class="dist-label">Cosine (full space)</span>
          <span class="dist-value full" id="d_bb_full_cos">—</span>
        </div>
        <div class="dist-row">
          <span class="dist-label">L2 &nbsp;&nbsp;&nbsp;&nbsp; (full space)</span>
          <span class="dist-value full" id="d_bb_full_l2">—</span>
        </div>
        <div class="dist-row">
          <span class="dist-label">Cosine (PCA-50)</span>
          <span class="dist-value pca" id="d_bb_pca_cos">—</span>
        </div>
        <div class="dist-row">
          <span class="dist-label">L2 &nbsp;&nbsp;&nbsp;&nbsp; (PCA-50)</span>
          <span class="dist-value pca" id="d_bb_pca_l2">—</span>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
var N            = {N};
var actionXY     = {json.dumps(action_xy)};
var backboneXY   = {json.dumps(backbone_xy)};
var pcaAction    = {json.dumps(pca_act_js)};
var pcaBackbone  = {json.dumps(pca_bb_js)};
var taskLabels   = {json.dumps(task_labels)};
var categories   = {json.dumps(cats)};
var frameIdxs    = {json.dumps(frame_indices)};
var epProgress   = {json.dumps(episode_progress)};
var camShorts    = {cam_shorts_js};
var N_TASK       = {n_task_traces};

// ── Decode precomputed distance matrices ────────────────────────────────────
var distB64Raw = {dist_b64_json};
var distMat = {{}};

function decodeDistMatrix(b64) {{
  var binary = atob(b64);
  var bytes  = new Uint8Array(binary.length);
  for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}

if (distB64Raw !== null) {{
  for (var key in distB64Raw) {{
    distMat[key] = decodeDistMatrix(distB64Raw[key]);
  }}
}}

// Upper-triangle index: row i, col j (j > i)
function triIdx(i, j) {{
  if (i > j) {{ var tmp = i; i = j; j = tmp; }}
  return i * (N - 1) - Math.floor(i * (i - 1) / 2) + (j - i - 1);
}}

function getDist(key, i, j) {{
  if (i === j) return 0.0;
  return distMat[key][triIdx(i, j)];
}}

// ── PCA-50 distance (fallback if no precomputed full-space) ──────────────────
function cosineDistPCA(a, b) {{
  var dot = 0, na = 0, nb = 0;
  for (var i = 0; i < a.length; i++) {{
    dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
  }}
  return 1.0 - Math.max(-1, Math.min(1, dot / (Math.sqrt(na)*Math.sqrt(nb) + 1e-10)));
}}

function l2DistPCA(a, b) {{
  var s = 0;
  for (var i = 0; i < a.length; i++) {{ var d = a[i]-b[i]; s += d*d; }}
  return Math.sqrt(s);
}}

// ── Helpers ──────────────────────────────────────────────────────────────────
function pad4(n) {{ return String(n).padStart(4, '0'); }}
function fmt(v) {{ return v.toFixed(4); }}

var selectedA = null;

function setHighlight(traceIdx, sampleIdx, plotId, xyArray) {{
  var xy = xyArray[sampleIdx];
  Plotly.restyle(plotId, {{x: [[xy[0]]], y: [[xy[1]]]}}, [traceIdx]);
}}
function clearHighlight(traceIdx, plotId) {{
  Plotly.restyle(plotId, {{x: [[]], y: [[]]}}, [traceIdx]);
}}

function playVideo(slot, sampleIdx) {{
  if (camShorts.length === 0) return;
  document.getElementById('vid-hint').style.display = 'none';
  camShorts.forEach(function(cs) {{
    var el = document.getElementById('vid_' + slot + '_' + cs);
    if (el) {{
      el.src = './clips/' + cs + '/' + pad4(sampleIdx) + '.mp4';
      el.style.border = '2px solid ' + (slot === 'A' ? '#2196F3' : '#4CAF50');
      el.load(); el.play().catch(function() {{}});
    }}
  }});
}}
function clearVideo(slot) {{
  camShorts.forEach(function(cs) {{
    var el = document.getElementById('vid_' + slot + '_' + cs);
    if (el) {{ el.src = ''; el.style.border = '2px solid #ccc'; }}
  }});
}}

function sampleInfo(idx) {{
  var prog = epProgress[idx];
  var progStr = (prog !== null && prog !== undefined)
    ? (prog * 100).toFixed(1) + '%'
    : 'N/A';
  return '<b>' + taskLabels[idx] + '</b>  (' + categories[idx] + ')\\n'
       + 'sample: ' + idx + '\\n'
       + 'frame : ' + frameIdxs[idx] + '\\n'
       + 'ep progress: ' + progStr;
}}

function showInfo(idxA, idxB) {{
  var html = '<b style="color:#2196F3;">A</b>  ' + sampleInfo(idxA);
  if (idxB !== null) {{
    html += '\\n\\n<b style="color:#4CAF50;">B</b>  ' + sampleInfo(idxB);
  }}
  document.getElementById('info-box').innerHTML = html;
}}

function showDist(idxA, idxB) {{
  // Full-space distances (precomputed)
  var hasFullDist = (Object.keys(distMat).length > 0);
  if (hasFullDist) {{
    document.getElementById('d_act_full_cos').textContent = fmt(getDist('action_full_cos', idxA, idxB));
    document.getElementById('d_act_full_l2' ).textContent = fmt(getDist('action_full_l2',  idxA, idxB));
    document.getElementById('d_bb_full_cos' ).textContent = fmt(getDist('bb_full_cos',     idxA, idxB));
    document.getElementById('d_bb_full_l2'  ).textContent = fmt(getDist('bb_full_l2',      idxA, idxB));
  }} else {{
    ['d_act_full_cos','d_act_full_l2','d_bb_full_cos','d_bb_full_l2'].forEach(function(id) {{
      document.getElementById(id).textContent = 'N/A';
    }});
  }}
  // PCA-50 distances (precomputed if available, else compute in browser)
  if (distMat['action_pca_cos']) {{
    document.getElementById('d_act_pca_cos').textContent = fmt(getDist('action_pca_cos', idxA, idxB));
    document.getElementById('d_act_pca_l2' ).textContent = fmt(getDist('action_pca_l2',  idxA, idxB));
    document.getElementById('d_bb_pca_cos' ).textContent = fmt(getDist('bb_pca_cos',     idxA, idxB));
    document.getElementById('d_bb_pca_l2'  ).textContent = fmt(getDist('bb_pca_l2',      idxA, idxB));
  }} else {{
    document.getElementById('d_act_pca_cos').textContent = fmt(cosineDistPCA(pcaAction[idxA],   pcaAction[idxB]));
    document.getElementById('d_act_pca_l2' ).textContent = fmt(l2DistPCA(    pcaAction[idxA],   pcaAction[idxB]));
    document.getElementById('d_bb_pca_cos' ).textContent = fmt(cosineDistPCA(pcaBackbone[idxA], pcaBackbone[idxB]));
    document.getElementById('d_bb_pca_l2'  ).textContent = fmt(l2DistPCA(    pcaBackbone[idxA], pcaBackbone[idxB]));
  }}
  document.getElementById('dist-panel').style.display = 'block';
}}

function hideDist() {{ document.getElementById('dist-panel').style.display = 'none'; }}

function handleClick(sampleIdx) {{
  if (selectedA === null) {{
    selectedA = sampleIdx;
    setHighlight(N_TASK,     sampleIdx, 'plot_action',   actionXY);
    setHighlight(N_TASK,     sampleIdx, 'plot_backbone', backboneXY);
    clearHighlight(N_TASK+1, 'plot_action');
    clearHighlight(N_TASK+1, 'plot_backbone');
    playVideo('A', sampleIdx); clearVideo('B');
    showInfo(sampleIdx, null); hideDist();
  }} else if (selectedA === sampleIdx) {{
    selectedA = null;
    clearHighlight(N_TASK,   'plot_action');   clearHighlight(N_TASK,   'plot_backbone');
    clearHighlight(N_TASK+1, 'plot_action');   clearHighlight(N_TASK+1, 'plot_backbone');
    clearVideo('A'); clearVideo('B');
    document.getElementById('info-box').innerHTML =
      '<span style="color:#aaa;font-style:italic;">click a point for details</span>';
    hideDist();
  }} else {{
    var idxA = selectedA;
    selectedA = null;
    setHighlight(N_TASK+1, sampleIdx, 'plot_action',   actionXY);
    setHighlight(N_TASK+1, sampleIdx, 'plot_backbone', backboneXY);
    playVideo('B', sampleIdx);
    showInfo(idxA, sampleIdx);
    showDist(idxA, sampleIdx);
  }}
}}

// ── Plotly init ──────────────────────────────────────────────────────────────
var actionData     = {json.dumps(action_traces)};
var backboneData   = {json.dumps(backbone_traces)};
var actionLayout   = {json.dumps(action_layout)};
var backboneLayout = {json.dumps(backbone_layout)};
var config = {{responsive: true, displayModeBar: true}};

Plotly.newPlot('plot_action', actionData, actionLayout, config)
.then(function(gd_a) {{
  return Plotly.newPlot('plot_backbone', backboneData, backboneLayout, config)
  .then(function(gd_b) {{
    gd_a.on('plotly_click', function(ev) {{ handleClick(ev.points[0].customdata[2]); }});
    gd_b.on('plotly_click', function(ev) {{ handleClick(ev.points[0].customdata[2]); }});
  }});
}});
</script>
</body>
</html>"""

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[html] saved → {out_path}  ({Path(out_path).stat().st_size/1e6:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    np.random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.output_dir) if args.output_dir else cache_dir / "joint_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_cache = cache_dir / "action_encoder_cache.npz"
    dit_cache = cache_dir / "dit_features_cache.npz"
    if not enc_cache.exists(): print(f"[error] Missing: {enc_cache}"); sys.exit(1)
    if not dit_cache.exists(): print(f"[error] Missing: {dit_cache}"); sys.exit(1)

    # ── Load caches ───────────────────────────────────────────────────────────
    print(f"[1/5] Loading caches from {cache_dir} ...")
    npz_enc          = np.load(enc_cache, allow_pickle=True)
    action_feats     = npz_enc["action_feats"]
    gt_actions       = npz_enc["gt_actions"] if "gt_actions" in npz_enc else None
    task_labels      = [str(x) for x in npz_enc["task_labels"]]
    frame_indices    = [int(x) for x in npz_enc["frame_indices"]]
    episode_progress = [float(x) for x in npz_enc["episode_progress"]] \
                       if "episode_progress" in npz_enc else [None] * len(task_labels)
    N, T, D_enc      = action_feats.shape
    print(f"  action_feats     : N={N}, T={T}, D_enc={D_enc}")
    print(f"  gt_actions       : {gt_actions.shape if gt_actions is not None else 'not in cache'}")
    print(f"  episode_progress : {'ok' if episode_progress[0] is not None else 'not in cache'}")

    npz_dit         = np.load(dit_cache, allow_pickle=True)
    bb_feats_np     = npz_dit["bb_feats"]
    bb_masks_np     = npz_dit["bb_masks"]
    task_labels_dit = [str(x) for x in npz_dit["task_labels"]]
    if task_labels == task_labels_dit:
        print(f"  backbone feats   : {len(bb_feats_np)} batches — task_labels MATCH ✓")
    else:
        n_mm = sum(a != b for a, b in zip(task_labels, task_labels_dit))
        print(f"  [warn] task_labels differ in {n_mm}/{N} positions!")

    bb_feats_list = [torch.from_numpy(x.astype(np.float32)) for x in bb_feats_np]
    bb_masks_list = [torch.from_numpy(x) for x in bb_masks_np]
    task_dist = {t: task_labels.count(t) for t in sorted(set(task_labels))}
    print(f"  task dist        : {task_dist}")

    # ── Video clip extraction ─────────────────────────────────────────────────
    clips_dirs = {}
    if args.dataset_root and not args.skip_clips:
        w, h = map(int, args.clip_size.split("x"))
        print(f"\n[2/5] Extracting video clips ({args.clip_size}, {args.action_horizon} frames) ...")
        clips_root = out_dir / "clips"
        joint_clips = cache_dir / "joint" / "clips"
        if joint_clips.exists() and not clips_root.exists():
            import os
            os.symlink(joint_clips.resolve(), clips_root)
            print(f"  [clips] symlinked from joint/clips/")
        clips_dirs = extract_video_clips(
            dataset_root=args.dataset_root,
            frame_indices=frame_indices,
            cam_keys=args.cam_keys,
            action_horizon=args.action_horizon,
            out_dir=clips_root,
            fps=10,
            output_size=(w, h),
        )
    else:
        print(f"\n[2/5] Skipping video clips")

    # ── Aggregate & t-SNE ─────────────────────────────────────────────────────
    print(f"\n[3/5] Aggregating & running t-SNE ...")
    print(f"  action_agg  : {args.action_agg}")
    print(f"  backbone_agg: {args.backbone_agg}")

    X_action   = agg_action(action_feats, args.action_agg, gt_actions=gt_actions)
    X_backbone = agg_backbone(bb_feats_list, bb_masks_list, args.backbone_agg)
    print(f"  X_action    : {X_action.shape}")
    print(f"  X_backbone  : {X_backbone.shape}")

    zscore = not args.no_zscore
    print(f"  zscore      : {zscore}")
    X_pca_action,   X2d_action   = run_tsne(X_action,   args.perplexity, args.tsne_n_iter, args.seed, "action_encoder", zscore=zscore)
    X_pca_backbone, X2d_backbone = run_tsne(X_backbone, args.perplexity, args.tsne_n_iter, args.seed, "backbone",        zscore=zscore)

    if X2d_action is None or X2d_backbone is None:
        print("[error] t-SNE failed."); sys.exit(1)

    # ── Pairwise distance matrices ────────────────────────────────────────────
    print(f"\n[4/5] Computing pairwise distance matrices ...")
    dist_b64 = compute_dist_matrices(X_action, X_backbone, X_pca_action, X_pca_backbone)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving outputs to {out_dir} ...")
    tag = f"tsne_p{args.perplexity}" + ("_noz" if args.no_zscore else "")

    plot_combined_static(
        X2d_action, X2d_backbone, task_labels,
        args.action_agg, args.backbone_agg, args.perplexity,
        out_dir / f"joint_{tag}_combined.png",
    )
    make_joint_html(
        X2d_action, X2d_backbone,
        X_pca_action, X_pca_backbone,
        task_labels, frame_indices, episode_progress,
        args.action_agg, args.backbone_agg, args.perplexity,
        out_dir / f"joint_{tag}.html",
        clips_dirs=clips_dirs,
        zscore=zscore,
        dist_b64=dist_b64,
    )

    print(f"\n✅ Done.  outputs → {out_dir}")


if __name__ == "__main__":
    main()
