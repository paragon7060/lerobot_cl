#!/usr/bin/env python3
"""
Joint visualization: action_encoder (delta_mean) vs VLM backbone (mask_mean)
for the SAME set of N samples, using t-SNE perp=30.

--joint_weights "idx0:w0,idx1:w1,..."  을 사용하면 raw_ aggregation method에서
특정 joint 차원에 가중치를 곱합니다. CW/CCW 같이 특정 joint로만 구분되는
패턴을 t-SNE에서 더 잘 분리할 수 있습니다.

예시:
  --action_agg raw_delta_mean --joint_weights "6:20.0,7:20.0"

Interactive HTML:
  - Two scatter plots side-by-side (action | backbone)
  - Click a point → same sample highlighted (red ring) in BOTH plots simultaneously
  - Camera videos (right_shoulder + wrist, action_horizon frames) autoplay below

Reads caches produced by visualize_action_embedding_tsne.py --run_dit:
  <cache_dir>/action_encoder_cache.npz  — action features  (N, T=16, D_enc)
  <cache_dir>/dit_features_cache.npz   — backbone features (per-batch arrays)

Usage:
  python src/lerobot/scripts/visualize_joint_embedding_weighted.py \
      --cache_dir /home/seonho/ws3/outputs/action_emb_vis_2048 \
      --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
      --action_agg raw_delta_mean \
      --joint_weights "6:20.0"
"""

import argparse
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
    p = argparse.ArgumentParser(description="Joint action vs backbone t-SNE (with joint weighting)")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Directory containing action_encoder_cache.npz and dit_features_cache.npz")
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Dataset root for video clip extraction. If None, skips clips.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory. Defaults to <cache_dir>/joint_weighted/")
    p.add_argument("--action_agg", type=str, default="raw_delta_mean",
                   choices=["mean", "max", "last", "first_last", "delta_total", "delta_mean",
                            "flatten", "first_mid_last", "raw_flatten", "raw_delta_mean",
                            "raw_delta_total"])
    p.add_argument("--backbone_agg", type=str, default="mask_mean",
                   choices=["mask_mean", "mean", "max", "first", "last"])
    p.add_argument("--perplexity",   type=int, default=30)
    p.add_argument("--tsne_n_iter",  type=int, default=1000)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--cam_keys", type=str, nargs="*", default=DEFAULT_CAM_KEYS,
                   help="Camera keys for video clip extraction")
    p.add_argument("--clip_size", type=str, default="320x240",
                   help="Width x Height of output clips (default: 320x240)")
    p.add_argument("--action_horizon", type=int, default=16,
                   help="Number of frames per clip (default: 16)")
    p.add_argument("--skip_clips", action="store_true",
                   help="Skip video clip extraction even if dataset_root is set")
    # ── Joint weighting ───────────────────────────────────────────────────────
    p.add_argument("--joint_weights", type=str, default=None,
                   help=(
                       "Joint-specific weights for raw_ aggregation methods. "
                       "Format: 'idx0:w0,idx1:w1,...'  e.g. '6:20.0,7:20.0'. "
                       "Only affects raw_delta_mean / raw_delta_total / raw_flatten."
                   ))
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Joint weight parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_joint_weights(spec: str, action_dim: int) -> np.ndarray:
    """'idx:weight,...' 문자열을 파싱해서 (action_dim,) 가중치 배열 반환.
    지정하지 않은 joint는 1.0 (변경 없음).
    """
    weights = np.ones(action_dim, dtype=np.float32)
    if not spec:
        return weights
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid joint_weights format: '{item}' (expected 'idx:weight')")
        idx, w = int(parts[0]), float(parts[1])
        if idx < 0 or idx >= action_dim:
            raise ValueError(f"Joint index {idx} out of range [0, {action_dim})")
        weights[idx] = w
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def agg_action(feats: np.ndarray, method: str,
               gt_actions: np.ndarray = None,
               joint_weights: np.ndarray = None) -> np.ndarray:
    """(N, T, D) → (N, D*)

    Encoder-space methods (use action_encoder output D=D_enc):
      mean, max, last, first_last, delta_total, delta_mean  — standard aggregations
      flatten        : (N, T*D)  — full temporal structure
      first_mid_last : (N, 3*D)  — concat t=0, t=T//2, t=-1
      ※ joint_weights는 encoder-space method에 적용되지 않습니다.

    Raw action-space methods (use gt_actions D=action_dim, joint_weights 적용됨):
      raw_flatten    : (N, T*D) — full GT trajectory
      raw_delta_mean : (N, D)   — mean per-step velocity in joint space
      raw_delta_total: (N, D)   — net displacement in joint space
    """
    # Raw action space methods — operate on gt_actions (N, T, action_dim)
    if method.startswith("raw_"):
        if gt_actions is None:
            raise ValueError(f"method='{method}' requires gt_actions (re-extract with updated script)")
        t = torch.from_numpy(gt_actions.astype(np.float32))
        N, T, D = t.shape

        # joint_weights 적용 — raw_ method는 action_dim 차원이므로 직접 가중치 적용 가능
        if joint_weights is not None:
            w = torch.from_numpy(joint_weights[:D])   # (D,)
            t = t * w                                  # (N, T, D) * (D,) 브로드캐스팅

        if method == "raw_flatten":
            out = t.reshape(N, T * D)
        elif method == "raw_delta_mean":
            out = (t[:, 1:, :] - t[:, :-1, :]).mean(dim=1)
        elif method == "raw_delta_total":
            out = t[:, -1, :] - t[:, 0, :]
        else:
            raise ValueError(method)
        return out.numpy()

    # Encoder-space methods — operate on action_encoder output (N, T, D_enc)
    if joint_weights is not None:
        print(f"  [warn] joint_weights는 encoder-space method '{method}'에 적용되지 않습니다. "
              f"raw_delta_mean / raw_delta_total / raw_flatten 중 하나를 사용하세요.")
    t = torch.from_numpy(feats)
    N, T, D = t.shape
    if method == "mean":           out = t.mean(dim=1)
    elif method == "max":          out = t.max(dim=1).values
    elif method == "last":         out = t[:, -1, :]
    elif method == "first_last":   out = (t[:, 0, :] + t[:, -1, :]) / 2.0
    elif method == "delta_total":  out = t[:, -1, :] - t[:, 0, :]
    elif method == "delta_mean":   out = (t[:, 1:, :] - t[:, :-1, :]).mean(dim=1)
    elif method == "flatten":
        out = t.reshape(N, T * D)
    elif method == "first_mid_last":
        mid = T // 2
        out = torch.cat([t[:, 0, :], t[:, mid, :], t[:, -1, :]], dim=-1)
    else:
        raise ValueError(method)
    return out.numpy()


def agg_backbone(feats_list: list, masks_list: list, method: str) -> np.ndarray:
    """List of (B, T_vl, D) batches → (N, D)"""
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

def pca_preprocess(X: np.ndarray, n_components: int = 50,
                    zscore: bool = True) -> np.ndarray | None:
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
             label: str = "", zscore: bool = True) -> np.ndarray | None:
    from sklearn.manifold import TSNE
    import sklearn
    X_pca = pca_preprocess(X, zscore=zscore)
    if X_pca is None:
        print(f"  [skip] {label}: all features constant"); return None
    if X_pca.std(axis=0)[0] < 1e-12:
        X_pca = X_pca + np.random.RandomState(seed).randn(*X_pca.shape) * 1e-6
    perplexity = min(perplexity, max(5.0, (X_pca.shape[0]-1)/3.0 - 1))
    kw = dict(n_components=2, perplexity=perplexity, random_state=seed, verbose=0)
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
        kw["max_iter"] = n_iter
    else:
        kw["n_iter"] = n_iter
    zs_tag = "zscore" if zscore else "no_zscore"
    print(f"  [t-SNE] {label} ({zs_tag})  shape={X_pca.shape}  perp={perplexity:.0f} ...")
    return TSNE(**kw).fit_transform(X_pca)


# ══════════════════════════════════════════════════════════════════════════════
# Video clip extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_video_clips(
    dataset_root: str,
    frame_indices: list,
    cam_keys: list,
    action_horizon: int,
    out_dir: Path,
    fps: int = 10,
    output_size: tuple = (320, 240),
) -> dict:
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
            row  = data_df.loc[global_idx]
            ep   = int(row["episode_index"])
            ts   = float(row["timestamp"])
            ep_r = ep_df.loc[ep]
            ci   = int(ep_r[col_ci])
            fi   = int(ep_r[col_fi])
            ft   = float(ep_r[col_ft])
            abs_ts = ft + ts
            groups[(ci, fi)].append((sample_idx, abs_ts))

        n_done = 0
        for (ci, fi), samples in sorted(groups.items()):
            vpath = root / f"videos/{cam_key}/chunk-{ci:03d}/file-{fi:03d}.mp4"
            samples_sorted = sorted(samples, key=lambda x: x[1])

            with av.open(str(vpath)) as vc:
                stream = vc.streams.video[0]
                tb     = float(stream.time_base)

                for sample_idx, abs_ts in samples_sorted:
                    out_file = cam_dir / f"{sample_idx:04d}.mp4"

                    pts = int(abs_ts / tb)
                    vc.seek(pts, stream=stream, backward=True)

                    raw_frames = []
                    for pkt in vc.demux(stream):
                        for frm in pkt.decode():
                            fts = float(frm.pts * tb) if frm.pts is not None else 0.0
                            if fts < abs_ts - 1.0 / fps:
                                continue
                            raw_frames.append(frm.to_ndarray(format="rgb24"))
                            if len(raw_frames) >= action_horizon:
                                break
                        if len(raw_frames) >= action_horizon:
                            break

                    if not raw_frames:
                        continue

                    from PIL import Image as PILImage
                    frames_resized = []
                    for rf in raw_frames[:action_horizon]:
                        pil = PILImage.fromarray(rf)
                        if pil.size != (W, H):
                            pil = pil.resize((W, H), PILImage.LANCZOS)
                        frames_resized.append(np.array(pil))

                    import io as _io
                    buf = _io.BytesIO()
                    with av.open(buf, mode="w", format="mp4") as oc:
                        ost = oc.add_stream("libx264", rate=fps)
                        ost.width   = W
                        ost.height  = H
                        ost.pix_fmt = "yuv420p"
                        ost.options = {"crf": "28", "preset": "ultrafast"}
                        for rf in frames_resized:
                            avf = av.VideoFrame.from_ndarray(rf, format="rgb24")
                            avf = avf.reformat(format="yuv420p")
                            for pkt in ost.encode(avf):
                                oc.mux(pkt)
                        for pkt in ost.encode():
                            oc.mux(pkt)

                    with open(str(out_file), "wb") as f:
                        f.write(buf.getvalue())

                    n_done += 1
                    if n_done % 100 == 0 or n_done == len(todo):
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
                         action_agg, backbone_agg, perplexity, joint_weights_spec, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    w_tag = f"  [joint_weights: {joint_weights_spec}]" if joint_weights_spec else ""
    pairs = [(X2d_action, f"action_encoder  ({action_agg}){w_tag}"),
             (X2d_backbone, f"backbone  ({backbone_agg})")]
    for col, (X2d, col_title) in enumerate(pairs):
        ax = axes[0][col]; _scatter_task(ax, X2d, task_labels)
        ax.set_title(f"{col_title}\nby task_id", fontsize=11)
        ax = axes[1][col]; _scatter_category(ax, X2d, task_labels)
        ax.set_title(f"{col_title}\nby category", fontsize=11)
    sup = (f"action_encoder ({action_agg}{w_tag})  vs  backbone ({backbone_agg})"
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
    X2d_action: np.ndarray,
    X2d_backbone: np.ndarray,
    task_labels: list,
    frame_indices: list,
    action_agg: str,
    backbone_agg: str,
    perplexity: int,
    out_path: Path,
    clips_dirs: dict = None,
    joint_weights_spec: str = None,
):
    N        = len(task_labels)
    task_ids = sorted(set(task_labels))
    cats     = [TASK_TO_CATEGORY.get(t, "?") for t in task_labels]

    cmap_mpl  = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(task_ids), 1))
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
        traces.append({
            "type": "scatter", "x": [], "y": [], "mode": "markers",
            "name": "selected", "showlegend": False, "hoverinfo": "skip",
            "marker": {"symbol": "circle-open", "size": 22, "color": "red",
                       "line": {"color": "red", "width": 3}},
        })
        return traces

    action_traces   = _make_traces(X2d_action)
    backbone_traces = _make_traces(X2d_backbone)
    n_task_traces   = len(task_ids)

    action_xy   = X2d_action.tolist()
    backbone_xy = X2d_backbone.tolist()

    shared_layout = {
        "height": 600, "hovermode": "closest",
        "xaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)", "title": "dim-1"},
        "yaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)", "title": "dim-2"},
        "legend": {"x": 1.01, "y": 1, "xanchor": "left", "font": {"size": 11}},
        "margin": {"l": 50, "r": 160, "t": 50, "b": 50},
    }
    w_tag = f"  [jw: {joint_weights_spec}]" if joint_weights_spec else ""
    action_layout   = dict(shared_layout, title=f"action_encoder  ({action_agg}){w_tag}")
    backbone_layout = dict(shared_layout, title=f"backbone  ({backbone_agg})")

    has_clips = bool(clips_dirs)
    cam_shorts = list(clips_dirs.keys()) if has_clips else []

    video_panel_html = ""
    if has_clips:
        for cs in cam_shorts:
            video_panel_html += (
                f'<div class="vid-box">'
                f'<p class="vid-label">{cs}</p>'
                f'<video id="vid_{cs}" autoplay loop muted playsinline width="100%">'
                f'</video>'
                f'</div>\n'
            )

    cam_shorts_js = json.dumps(cam_shorts)
    title = (f"action_encoder ({action_agg}{w_tag})  vs  backbone ({backbone_agg})"
             f"  |  t-SNE perp={perplexity}  |  N={N}")

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
    display: flex; gap: 14px; padding: 10px 14px;
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    flex: 3; min-height: 80px;
  }}
  .vid-box {{ text-align: center; flex: 1; min-width: 0; }}
  .vid-label {{ margin: 0 0 4px; font-size: 11px; font-weight: bold; color: #555; }}
  .vid-box video {{
    width: 100%; max-width: 340px; height: auto;
    border: 2px solid #ccc; border-radius: 4px; background: #111;
  }}
  #info-box {{
    flex: 1; padding: 10px 14px; background: #fff;
    border: 1px solid #ddd; border-radius: 6px;
    font-size: 13px; color: #444; white-space: pre-line; min-height: 80px;
  }}
</style>
</head>
<body>
<h3>{title}</h3>
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
  <div id="info-box"><span style="color:#aaa;font-style:italic;">click a point for details</span></div>
</div>

<script>
var actionXY    = {json.dumps(action_xy)};
var backboneXY  = {json.dumps(backbone_xy)};
var taskLabels  = {json.dumps(task_labels)};
var categories  = {json.dumps(cats)};
var frameIdxs   = {json.dumps(frame_indices)};
var camShorts   = {cam_shorts_js};
var N_TASK      = {n_task_traces};

var actionData   = {json.dumps(action_traces)};
var backboneData = {json.dumps(backbone_traces)};
var actionLayout   = {json.dumps(action_layout)};
var backboneLayout = {json.dumps(backbone_layout)};
var config = {{responsive: true, displayModeBar: true}};

function pad4(n) {{ return String(n).padStart(4, '0'); }}

function highlightBoth(sampleIdx) {{
  var ax = actionXY[sampleIdx], bx = backboneXY[sampleIdx];
  Plotly.restyle('plot_action',   {{x: [[ax[0]]], y: [[ax[1]]]}}, [N_TASK]);
  Plotly.restyle('plot_backbone', {{x: [[bx[0]]], y: [[bx[1]]]}}, [N_TASK]);

  if (camShorts.length > 0) {{
    document.getElementById('vid-hint').style.display = 'none';
    camShorts.forEach(function(cs) {{
      var el = document.getElementById('vid_' + cs);
      if (el) {{
        el.src = './clips/' + cs + '/' + pad4(sampleIdx) + '.mp4';
        el.style.border = '2px solid #4a90d9';
        el.load();
        el.play().catch(function() {{}});
      }}
    }});
  }}

  document.getElementById('info-box').innerHTML =
    '<b>' + taskLabels[sampleIdx] + '</b>  (' + categories[sampleIdx] + ')\\n' +
    'sample idx : ' + sampleIdx + '\\n' +
    'frame idx  : ' + frameIdxs[sampleIdx];
}}

Plotly.newPlot('plot_action', actionData, actionLayout, config)
.then(function(gd_a) {{
  return Plotly.newPlot('plot_backbone', backboneData, backboneLayout, config)
  .then(function(gd_b) {{
    gd_a.on('plotly_click', function(ev) {{
      highlightBoth(ev.points[0].customdata[2]);
    }});
    gd_b.on('plotly_click', function(ev) {{
      highlightBoth(ev.points[0].customdata[2]);
    }});
  }});
}});
</script>
</body>
</html>"""

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[html] saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Action-only interactive HTML (progress + task, click-to-show images)
# ══════════════════════════════════════════════════════════════════════════════

def make_action_html(
    X2d: np.ndarray,
    task_labels: list,
    frame_indices: list,
    title: str,
    out_path: Path,
    episode_progress: np.ndarray = None,
    clips_dirs: dict = None,
):
    """Action embedding 단독 interactive HTML.
    Left: colored by episode progress (있으면), Right: colored by task.
    Click → video clips 재생.
    """
    N        = len(task_labels)
    task_ids = sorted(set(task_labels))
    cats     = [TASK_TO_CATEGORY.get(t, "?") for t in task_labels]
    progress = episode_progress if episode_progress is not None else np.zeros(N)
    progress_pct = (progress * 100).round(1).tolist()

    cmap_mpl  = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(task_ids), 1))
    task_color = {t: f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
                  for t, (r, g, b, _) in
                  {t: cmap_mpl(i) for i, t in enumerate(task_ids)}.items()}

    # ── Left panel: progress ─────────────────────────────────────────────────
    progress_trace = {
        "type": "scatter",
        "x": X2d[:, 0].tolist(), "y": X2d[:, 1].tolist(),
        "mode": "markers",
        "marker": {
            "color": progress.tolist(),
            "colorscale": "Viridis", "size": 7, "opacity": 0.85,
            "showscale": True,
            "colorbar": {"title": "progress", "x": 0.44, "len": 0.9},
            "cmin": 0.0, "cmax": 1.0,
        },
        "customdata": [[task_labels[i], cats[i], progress_pct[i],
                         frame_indices[i], i] for i in range(N)],
        "hovertemplate": (
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "progress: %{customdata[2]:.1f}%<br>"
            "frame: %{customdata[3]}<br>sample: %{customdata[4]}<extra></extra>"
        ),
        "showlegend": False, "name": "progress",
    }
    # highlight ring for left
    progress_highlight = {
        "type": "scatter", "x": [], "y": [], "mode": "markers",
        "name": "selected", "showlegend": False, "hoverinfo": "skip",
        "marker": {"symbol": "circle-open", "size": 22, "color": "red",
                   "line": {"color": "red", "width": 3}},
    }

    # ── Right panel: task colour ─────────────────────────────────────────────
    task_traces = []
    arr = np.array(task_labels)
    for tid in task_ids:
        idx = np.where(arr == tid)[0].tolist()
        task_traces.append({
            "type": "scatter",
            "x": X2d[idx, 0].tolist(), "y": X2d[idx, 1].tolist(),
            "mode": "markers", "name": tid,
            "marker": {"color": task_color[tid], "size": 7, "opacity": 0.85},
            "customdata": [[task_labels[i], cats[i], progress_pct[i],
                             frame_indices[i], i] for i in idx],
            "hovertemplate": (
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "progress: %{customdata[2]:.1f}%<br>"
                "frame: %{customdata[3]}<br>sample: %{customdata[4]}<extra></extra>"
            ),
        })
    task_highlight = {
        "type": "scatter", "x": [], "y": [], "mode": "markers",
        "name": "selected", "showlegend": False, "hoverinfo": "skip",
        "marker": {"symbol": "circle-open", "size": 22, "color": "red",
                   "line": {"color": "red", "width": 3}},
    }

    left_traces  = [progress_trace, progress_highlight]
    right_traces = task_traces + [task_highlight]
    n_task_traces = len(task_ids)
    all_xy = X2d.tolist()

    has_clips = bool(clips_dirs)
    cam_shorts = list(clips_dirs.keys()) if has_clips else []
    video_panel_html = ""
    if has_clips:
        for cs in cam_shorts:
            video_panel_html += (
                f'<div class="vid-box">'
                f'<p class="vid-label">{cs}</p>'
                f'<video id="vid_{cs}" autoplay loop muted playsinline width="100%">'
                f'</video>'
                f'</div>\n'
            )

    shared_ax = {
        "showgrid": True, "gridcolor": "rgba(200,200,200,0.3)",
    }
    left_layout = {
        "title": "Colored by episode progress", "height": 600,
        "hovermode": "closest",
        "xaxis": shared_ax, "yaxis": shared_ax,
        "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
    }
    right_layout = {
        "title": "Colored by task", "height": 600,
        "hovermode": "closest",
        "xaxis": shared_ax, "yaxis": shared_ax,
        "legend": {"x": 1.01, "y": 1, "xanchor": "left", "font": {"size": 11}},
        "margin": {"l": 50, "r": 160, "t": 50, "b": 50},
    }

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
    display: flex; gap: 14px; padding: 10px 14px;
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    flex: 3; min-height: 80px;
  }}
  .vid-box {{ text-align: center; flex: 1; min-width: 0; }}
  .vid-label {{ margin: 0 0 4px; font-size: 11px; font-weight: bold; color: #555; }}
  .vid-box video {{
    width: 100%; max-width: 340px; height: auto;
    border: 2px solid #ccc; border-radius: 4px; background: #111;
  }}
  #info-box {{
    flex: 1; padding: 10px 14px; background: #fff;
    border: 1px solid #ddd; border-radius: 6px;
    font-size: 13px; color: #444; white-space: pre-line; min-height: 80px;
  }}
</style>
</head>
<body>
<h3>{title}</h3>
<div id="plots-row">
  <div id="plot_left"></div>
  <div id="plot_right"></div>
</div>
<div id="bottom-row">
  <div id="video-panel">
{video_panel_html}
    <span id="vid-hint" style="color:#aaa;font-style:italic;font-size:13px;">
      &#8592; click a point to play the action clip
    </span>
  </div>
  <div id="info-box"><span style="color:#aaa;font-style:italic;">click a point for details</span></div>
</div>

<script>
var allXY       = {json.dumps(all_xy)};
var taskLabels  = {json.dumps(task_labels)};
var categories  = {json.dumps(cats)};
var frameIdxs   = {json.dumps(frame_indices)};
var progressPct = {json.dumps(progress_pct)};
var camShorts   = {json.dumps(cam_shorts)};
var N_TASK      = {n_task_traces};

var leftData    = {json.dumps(left_traces)};
var rightData   = {json.dumps(right_traces)};
var leftLayout  = {json.dumps(left_layout)};
var rightLayout = {json.dumps(right_layout)};
var config = {{responsive: true, displayModeBar: true}};

function pad4(n) {{ return String(n).padStart(4, '0'); }}

function highlightBoth(sampleIdx) {{
  var xy = allXY[sampleIdx];
  // left: progress_highlight is trace index 1
  Plotly.restyle('plot_left',  {{x: [[xy[0]]], y: [[xy[1]]]}}, [1]);
  // right: task_highlight is last trace = N_TASK
  Plotly.restyle('plot_right', {{x: [[xy[0]]], y: [[xy[1]]]}}, [N_TASK]);

  if (camShorts.length > 0) {{
    document.getElementById('vid-hint').style.display = 'none';
    camShorts.forEach(function(cs) {{
      var el = document.getElementById('vid_' + cs);
      if (el) {{
        el.src = './clips/' + cs + '/' + pad4(sampleIdx) + '.mp4';
        el.style.border = '2px solid #4a90d9';
        el.load(); el.play().catch(function() {{}});
      }}
    }});
  }}

  document.getElementById('info-box').innerHTML =
    '<b>' + taskLabels[sampleIdx] + '</b>  (' + categories[sampleIdx] + ')\\n' +
    'progress : ' + progressPct[sampleIdx].toFixed(1) + '%\\n' +
    'sample   : ' + sampleIdx + '\\n' +
    'frame    : ' + frameIdxs[sampleIdx];
}}

Plotly.newPlot('plot_left', leftData, leftLayout, config)
.then(function(gd_l) {{
  return Plotly.newPlot('plot_right', rightData, rightLayout, config)
  .then(function(gd_r) {{
    gd_l.on('plotly_click', function(ev) {{ highlightBoth(ev.points[0].customdata[4]); }});
    gd_r.on('plotly_click', function(ev) {{ highlightBoth(ev.points[0].customdata[4]); }});
  }});
}});
</script>
</body>
</html>"""

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[html] saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    np.random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.output_dir) if args.output_dir else cache_dir / "joint_weighted"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_cache = cache_dir / "action_encoder_cache.npz"
    dit_cache = cache_dir / "dit_features_cache.npz"

    if not enc_cache.exists():
        print(f"[error] Missing: {enc_cache}"); sys.exit(1)
    if not dit_cache.exists():
        print(f"[error] Missing: {dit_cache}"); sys.exit(1)

    # ── Load caches ───────────────────────────────────────────────────────────
    print(f"[1/4] Loading caches from {cache_dir} ...")
    npz_enc       = np.load(enc_cache, allow_pickle=True)
    action_feats  = npz_enc["action_feats"]
    gt_actions    = npz_enc["gt_actions"] if "gt_actions" in npz_enc else None
    task_labels   = [str(x) for x in npz_enc["task_labels"]]
    frame_indices = [int(x) for x in npz_enc["frame_indices"]]
    N, T, D_enc   = action_feats.shape
    action_dim    = gt_actions.shape[-1] if gt_actions is not None else None
    print(f"  action_feats  : N={N}, T={T}, D_enc={D_enc}")
    print(f"  gt_actions    : shape={gt_actions.shape if gt_actions is not None else 'not in cache'}")

    npz_dit         = np.load(dit_cache, allow_pickle=True)
    bb_feats_np     = npz_dit["bb_feats"]
    bb_masks_np     = npz_dit["bb_masks"]
    task_labels_dit = [str(x) for x in npz_dit["task_labels"]]

    if task_labels == task_labels_dit:
        print(f"  backbone feats: {len(bb_feats_np)} batches — task_labels MATCH ✓")
    else:
        n_mismatch = sum(a != b for a, b in zip(task_labels, task_labels_dit))
        print(f"  [warn] task_labels differ in {n_mismatch}/{N} positions!")

    bb_feats_list = [torch.from_numpy(x.astype(np.float32)) for x in bb_feats_np]
    bb_masks_list = [torch.from_numpy(x) for x in bb_masks_np]
    D_vl = bb_feats_list[0].shape[-1]
    print(f"  backbone D_vl  : {D_vl}")

    task_dist = {t: task_labels.count(t) for t in sorted(set(task_labels))}
    print(f"  task dist      : {task_dist}")

    # ── Joint weights ─────────────────────────────────────────────────────────
    joint_weights = None
    if args.joint_weights:
        if args.action_agg.startswith("raw_"):
            if gt_actions is None:
                print("[error] gt_actions not in cache. Re-run visualize_action_embedding_tsne.py first.")
                sys.exit(1)
            joint_weights = parse_joint_weights(args.joint_weights, action_dim)
            print(f"\n  joint_weights applied: {args.joint_weights}")
            print(f"  weight vector (action_dim={action_dim}): {joint_weights}")
        else:
            print(f"  [warn] --joint_weights 무시됨: action_agg='{args.action_agg}'은 "
                  f"raw_ method가 아닙니다. raw_delta_mean / raw_delta_total / raw_flatten 사용 권장.")

    # ── Video clip extraction ─────────────────────────────────────────────────
    clips_dirs = {}
    if args.dataset_root and not args.skip_clips:
        w, h = map(int, args.clip_size.split("x"))
        print(f"\n[2/4] Extracting video clips ({args.clip_size}, {args.action_horizon} frames) ...")
        clips_root = out_dir / "clips"
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
        print(f"\n[2/4] Skipping video clips (no --dataset_root or --skip_clips set)")

    # ── Episode progress (for action-only HTML) ────────────────────────────────
    episode_progress = npz_enc["episode_progress"] if "episode_progress" in npz_enc else None

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Aggregating & running t-SNE (zscore + no_zscore) ...")
    print(f"  action   agg : {args.action_agg}")
    print(f"  backbone agg : {args.backbone_agg}")
    if joint_weights is not None:
        print(f"  joint_weights: {args.joint_weights}")

    X_action   = agg_action(action_feats, args.action_agg,
                             gt_actions=gt_actions, joint_weights=joint_weights)
    X_backbone = agg_backbone(bb_feats_list, bb_masks_list, args.backbone_agg)
    assert X_action.shape[0] == X_backbone.shape[0] == N
    print(f"  X_action   : {X_action.shape}")
    print(f"  X_backbone : {X_backbone.shape}")

    # ── Run t-SNE for both zscore variants ────────────────────────────────────
    results = {}  # {("zscore"|"no_zscore"): (X2d_action, X2d_backbone)}
    for zscore in [True, False]:
        zs_label = "zscore" if zscore else "no_zscore"
        print(f"\n  ── {zs_label} ──")
        X2d_a = run_tsne(X_action,   args.perplexity, args.tsne_n_iter, args.seed,
                         "action_encoder", zscore=zscore)
        X2d_b = run_tsne(X_backbone, args.perplexity, args.tsne_n_iter, args.seed,
                         "backbone", zscore=zscore)
        if X2d_a is None or X2d_b is None:
            print(f"  [warn] t-SNE failed for {zs_label}, skipping")
            continue
        results[zs_label] = (X2d_a, X2d_b)

    if not results:
        print("[error] All t-SNE runs failed."); sys.exit(1)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving outputs to {out_dir} ...")
    base_tag = f"tsne_p{args.perplexity}"
    if args.joint_weights:
        jw_tag = args.joint_weights.replace(":", "x").replace(",", "_")
        base_tag = f"{base_tag}_jw{jw_tag}"

    w_tag = f"  [jw: {args.joint_weights}]" if args.joint_weights else ""

    for zs_label, (X2d_action, X2d_backbone) in results.items():
        tag = f"{base_tag}_{zs_label}"
        zs_display = "z-score" if zs_label == "zscore" else "no z-score"

        # ① Joint HTML (action vs backbone)
        plot_combined_static(
            X2d_action, X2d_backbone, task_labels,
            args.action_agg, args.backbone_agg, args.perplexity,
            args.joint_weights,
            out_dir / f"joint_{tag}_combined.png",
        )
        make_joint_html(
            X2d_action, X2d_backbone, task_labels, frame_indices,
            args.action_agg, args.backbone_agg, args.perplexity,
            out_dir / f"joint_{tag}.html",
            clips_dirs=clips_dirs,
            joint_weights_spec=args.joint_weights,
        )

        # ② Action-only HTML (progress + task)
        action_title = (f"action_encoder ({args.action_agg}{w_tag})"
                        f"  |  {zs_display}  |  t-SNE perp={args.perplexity}  |  N={N}")
        make_action_html(
            X2d_action, task_labels, frame_indices,
            title=action_title,
            out_path=out_dir / f"action_{tag}.html",
            episode_progress=episode_progress,
            clips_dirs=clips_dirs,
        )

    print(f"\n✅ Done.  outputs → {out_dir}")
    for zs_label in results:
        tag = f"{base_tag}_{zs_label}"
        print(f"  ├── joint_{tag}.html          — action vs backbone (joint)")
        print(f"  ├── joint_{tag}_combined.png  — static 2×2")
        print(f"  ├── action_{tag}.html         — action embedding only")
    if args.joint_weights:
        print(f"  joint_weights: {args.joint_weights}")


if __name__ == "__main__":
    main()
