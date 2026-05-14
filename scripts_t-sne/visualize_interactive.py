#!/usr/bin/env python3
"""
Interactive t-SNE / UMAP visualisation of GROOT backbone_features.

두 가지 시각화를 생성한다:
  1. Plotly HTML  — 각 점에 마우스 오버 시 실제 입력 이미지가 팝업 (미팅용)
  2. Matplotlib 썸네일 — 각 scatter point 위치에 직접 이미지 렌더링 (논문용)

필요 사항:
  - features_cache.npz 에 frame_indices 가 포함되어 있어야 함.
    없으면 --skip_cache 플래그로 visualize_backbone_tsne.py 를 먼저 재실행할 것.

Usage:
    conda run -n groot python src/lerobot/scripts/visualize_interactive.py \
        --cache_path /home/seonho/ws3/outputs/tsne_vis/features_cache.npz \
        --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
        --output_dir /home/seonho/ws3/outputs/tsne_vis_interactive \
        --agg_method mask_mean \
        --reduction umap \
        --n_neighbors 15 \
        --image_keys wrist right_shoulder
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.offsetbox as moffsetbox
import numpy as np
import torch
from PIL import Image

SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ──────────────────────────────────────────────
# Task / category constants (same as main script)
# ──────────────────────────────────────────────
TASK_TO_CATEGORY = {
    "1ext": "cabinet",
    "3a": "door",  "3b": "door",  "3c": "door",  "3d": "door",
    "5a": "bottle", "5b": "bottle", "5c": "bottle", "5d": "bottle",
    "5e": "bottle", "5f": "bottle", "5g": "bottle", "5h": "bottle",
}

CATEGORY_COLORS_HEX = {
    "cabinet": "#1f77b4",
    "door":    "#ff7f0e",
    "bottle":  "#2ca02c",
}

# Dataset image keys available (without "observation.images." prefix)
AVAILABLE_IMAGE_KEYS = [
    "wrist", "right_shoulder", "left_shoulder", "guide",
]

# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive t-SNE/UMAP with per-point image hover"
    )
    p.add_argument("--cache_path",  type=str,
                   default="/home/seonho/ws3/outputs/tsne_vis/features_cache.npz")
    p.add_argument("--dataset_root", type=str,
                   default="/home/seonho/workspace/data/paragon7060/INSIGHTfixposV3")
    p.add_argument("--dataset_repo_id", type=str,
                   default="paragon7060/INSIGHTfixposV3")
    p.add_argument("--output_dir", type=str,
                   default="/home/seonho/ws3/outputs/tsne_vis_interactive")
    p.add_argument("--agg_method", type=str, default="mask_mean",
                   choices=["mask_mean", "mean", "max", "last"])
    p.add_argument("--reduction", type=str, default="umap",
                   choices=["tsne", "umap"])
    p.add_argument("--perplexity", type=int, default=30,
                   help="t-SNE perplexity (only used when --reduction=tsne)")
    p.add_argument("--n_neighbors", type=int, default=15,
                   help="UMAP n_neighbors (only used when --reduction=umap)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_keys", nargs="+",
                   default=["wrist", "right_shoulder"],
                   help="Which camera images to show. Options: " +
                        ", ".join(AVAILABLE_IMAGE_KEYS))
    p.add_argument("--thumbnail_px", type=int, default=96,
                   help="Thumbnail size in pixels for HTML hover and matplotlib")
    p.add_argument("--matplotlib_every", type=int, default=4,
                   help="Show thumbnail for every N-th point in matplotlib plot "
                        "(prevents overcrowding, 1 = all points)")
    p.add_argument("--max_html_images", type=int, default=512,
                   help="Max images to embed in HTML (large N → large file)")
    return p.parse_args()


# ──────────────────────────────────────────────
# Aggregation  (identical to main script)
# ──────────────────────────────────────────────
def _agg_single(feats, masks, method):
    if method == "mean":
        return feats.mean(dim=1)
    elif method == "mask_mean":
        m = masks.unsqueeze(-1).float()
        return (feats * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    elif method == "max":
        m = masks.unsqueeze(-1).float()
        masked = feats * m + (1.0 - m) * (-1e9)
        return masked.max(dim=1).values
    elif method == "last":
        lengths = masks.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, feats.shape[-1])
        return feats.gather(dim=1, index=idx).squeeze(1)
    else:
        raise ValueError(f"Unknown method: {method}")


def aggregate(features_batches, masks_batches, method):
    parts = [_agg_single(f, m, method) for f, m in zip(features_batches, masks_batches)]
    return torch.cat(parts, dim=0).numpy()


# ──────────────────────────────────────────────
# PCA + dimensionality reduction
# ──────────────────────────────────────────────
def pca_preprocess(X):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    var = X.var(axis=0)
    X_clean = X[:, var > 1e-10]
    if X_clean.shape[1] == 0:
        return None
    mean = X_clean.mean(axis=0); std = X_clean.std(axis=0).clip(min=1e-10)
    X_scaled = (X_clean - mean) / std
    n_comp = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    return np.nan_to_num(X_pca, nan=0., posinf=0., neginf=0.)


def reduce(X, method, perplexity=30, n_neighbors=15, seed=42):
    X_pca = pca_preprocess(X)
    if X_pca is None:
        raise ValueError("All features are constant — cannot reduce.")

    if method == "tsne":
        import sklearn
        from sklearn.manifold import TSNE
        kw = dict(n_components=2, perplexity=min(perplexity, (X_pca.shape[0]-1)/3.0 - 1),
                  random_state=seed, verbose=1)
        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
            kw["max_iter"] = 1000
        else:
            kw["n_iter"] = 1000
        return TSNE(**kw).fit_transform(X_pca)

    elif method == "umap":
        import umap as umap_lib
        return umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                             min_dist=0.1, random_state=seed).fit_transform(X_pca)


# ──────────────────────────────────────────────
# Image loading from dataset
# ──────────────────────────────────────────────
def load_images_from_dataset(dataset, frame_indices, image_keys, thumbnail_px):
    """
    Returns dict: key → list of PIL.Image (one per frame index, in order).
    Images are resized to thumbnail_px × thumbnail_px.
    """
    images = {k: [] for k in image_keys}
    print(f"  loading {len(frame_indices)} frames from dataset ...")
    for i, idx in enumerate(frame_indices):
        sample = dataset[int(idx)]
        for key in image_keys:
            full_key = f"observation.images.{key}"
            if full_key not in sample:
                images[key].append(None)
                continue
            img_tensor = sample[full_key]  # (C, H, W) or (H, W, C), uint8 or float
            # Convert to numpy uint8
            if isinstance(img_tensor, torch.Tensor):
                arr = img_tensor.numpy()
            else:
                arr = np.array(img_tensor)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            # Handle channel dimension
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # (C,H,W)
                arr = arr.transpose(1, 2, 0)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            pil = Image.fromarray(arr).resize((thumbnail_px, thumbnail_px), Image.LANCZOS)
            images[key].append(pil)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frame_indices)} frames loaded ...", end="\r")
    print()
    return images


def pil_to_base64(pil_img, fmt="JPEG", quality=75):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ──────────────────────────────────────────────
# Plotly interactive HTML
# ──────────────────────────────────────────────
def build_plotly_html(X_2d, task_labels, frame_indices, images_dict,
                      image_keys, title, out_path):
    """
    Single self-contained HTML file with:
      - Scatter plot colored by task_id AND category (toggle buttons)
      - Hover shows floating image popup via JavaScript (not hovertemplate)
      - Plotly hovertemplate only shows text; images rendered in a floating <div>
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [skip] plotly not installed. Run: pip install plotly")
        return

    N = len(task_labels)
    task_ids   = sorted(set(task_labels))
    categories = [TASK_TO_CATEGORY[t] for t in task_labels]

    # ── Color maps ──
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(task_ids), 1))
    task_color_map = {t: f"rgba({int(cmap(i)[0]*255)},{int(cmap(i)[1]*255)},{int(cmap(i)[2]*255)},0.85)"
                      for i, t in enumerate(task_ids)}

    # ── Pre-encode all images into base64 ──
    # customdata per point: [frame_idx, b64_key0, b64_key1, ...]
    print("  encoding images to base64 ...")
    all_customdata = []
    for i in range(N):
        row = [str(frame_indices[i])]
        for key in image_keys:
            pil_list = images_dict.get(key, [])
            pil = pil_list[i] if i < len(pil_list) else None
            row.append(pil_to_base64(pil) if pil is not None else "")
        all_customdata.append(row)
    all_customdata_arr = np.array(all_customdata, dtype=object)

    # ── Build two trace sets: by task_id and by category ──
    fig = go.Figure()

    # Traces by task_id (default visible)
    for tid in task_ids:
        mask = np.array(task_labels) == tid
        idx  = np.where(mask)[0]
        cat_labels = [categories[j] for j in idx]
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1],
            mode="markers",
            name=tid,
            marker=dict(size=7, color=task_color_map[tid], line=dict(width=0)),
            customdata=all_customdata_arr[mask].tolist(),
            hovertemplate=(
                f"<b>task: {tid}</b><br>"
                "frame: %{customdata[0]}<br>"
                "<extra></extra>"
            ),
            legendgroup="task",
            legendgrouptitle_text="task_id" if tid == task_ids[0] else None,
            visible=True,
        ))

    # Traces by category (initially hidden)
    for cat, hex_color in CATEGORY_COLORS_HEX.items():
        mask = np.array(categories) == cat
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1],
            mode="markers",
            name=cat,
            marker=dict(size=7, color=hex_color, opacity=0.75, line=dict(width=0)),
            customdata=all_customdata_arr[mask].tolist(),
            hovertemplate=(
                f"<b>category: {cat}</b><br>"
                "frame: %{customdata[0]}<br>"
                "<extra></extra>"
            ),
            legendgroup="category",
            legendgrouptitle_text="category" if cat == list(CATEGORY_COLORS_HEX)[0] else None,
            visible=False,
        ))

    n_task_traces = len(task_ids)
    n_cat_traces  = len(CATEGORY_COLORS_HEX)

    # Toggle buttons
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        width=1000, height=750,
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=12, namelength=-1),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.0, y=1.08, xanchor="left",
            buttons=[
                dict(label="by task_id",
                     method="update",
                     args=[{"visible": [True]*n_task_traces + [False]*n_cat_traces},
                           {"legend": {"title": "task_id"}}]),
                dict(label="by category",
                     method="update",
                     args=[{"visible": [False]*n_task_traces + [True]*n_cat_traces},
                           {"legend": {"title": "category"}}]),
            ],
            showactive=True,
        )],
        xaxis=dict(title="dim-1", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="dim-2", showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(itemsizing="constant"),
    )

    # ── JavaScript: floating image div on hover ──
    img_keys_js = str(image_keys).replace("'", '"')   # valid JSON array
    post_script = f"""
(function() {{
  var gd = document.querySelector('.plotly-graph-div');
  if (!gd) {{ return; }}

  // Create floating image container
  var imgDiv = document.createElement('div');
  imgDiv.id = 'hover-img-popup';
  Object.assign(imgDiv.style, {{
    position: 'fixed',
    zIndex: '99999',
    background: 'white',
    border: '1px solid #aaa',
    borderRadius: '6px',
    padding: '8px',
    pointerEvents: 'none',
    display: 'none',
    boxShadow: '3px 3px 10px rgba(0,0,0,0.25)',
    maxWidth: '400px',
  }});
  document.body.appendChild(imgDiv);

  var keys = {img_keys_js};

  gd.on('plotly_hover', function(data) {{
    var pt = data.points[0];
    var cd = pt.customdata;   // [frame_idx, b64_0, b64_1, ...]
    var html = '<div style="font-family:monospace;font-size:12px;margin-bottom:6px">frame: ' + cd[0] + '</div>';
    html += '<div style="display:flex;gap:6px">';
    for (var k = 0; k < keys.length; k++) {{
      var b64 = cd[k + 1];
      if (b64) {{
        html += '<div style="text-align:center">';
        html += '<div style="font-size:10px;margin-bottom:2px">' + keys[k] + '</div>';
        html += '<img src="data:image/jpeg;base64,' + b64 + '" width="120" height="120" style="display:block">';
        html += '</div>';
      }}
    }}
    html += '</div>';
    imgDiv.innerHTML = html;
    imgDiv.style.display = 'block';
  }});

  gd.on('plotly_unhover', function() {{
    imgDiv.style.display = 'none';
  }});

  document.addEventListener('mousemove', function(e) {{
    if (imgDiv.style.display !== 'none') {{
      var x = e.clientX + 18;
      var y = e.clientY + 18;
      // Keep popup within viewport
      if (x + imgDiv.offsetWidth > window.innerWidth - 10) {{
        x = e.clientX - imgDiv.offsetWidth - 10;
      }}
      if (y + imgDiv.offsetHeight > window.innerHeight - 10) {{
        y = e.clientY - imgDiv.offsetHeight - 10;
      }}
      imgDiv.style.left = x + 'px';
      imgDiv.style.top  = y + 'px';
    }}
  }});
}})();
"""

    fig.write_html(str(out_path), include_plotlyjs="cdn", post_script=post_script)
    print(f"  [plotly] saved → {out_path}")


# ──────────────────────────────────────────────
# Matplotlib thumbnail plot
# ──────────────────────────────────────────────
def build_matplotlib_thumbnail(X_2d, task_labels, images_dict,
                                primary_key, title, out_path,
                                every_n=4, thumb_px=48):
    """
    Scatter plot with miniature image thumbnails placed at each point.
    Shows every `every_n`-th point to prevent overcrowding.
    """
    categories = np.array([TASK_TO_CATEGORY[t] for t in task_labels])
    cat_colors = np.array([CATEGORY_COLORS_HEX[c] for c in categories])

    fig, ax = plt.subplots(figsize=(14, 10))

    # Background scatter (all points, category colored, small)
    for cat, color in CATEGORY_COLORS_HEX.items():
        mask = categories == cat
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, s=10, alpha=0.25, linewidths=0, label=cat)

    # Overlay image thumbnails for every N-th point
    pil_list = images_dict.get(primary_key, [])
    placed = 0
    for i in range(0, len(X_2d), every_n):
        pil = pil_list[i] if i < len(pil_list) else None
        if pil is None:
            continue
        arr = np.array(pil.resize((thumb_px, thumb_px), Image.LANCZOS))
        im = moffsetbox.OffsetImage(arr, zoom=1.0)
        ab = moffsetbox.AnnotationBbox(
            im, (X_2d[i, 0], X_2d[i, 1]),
            frameon=True,
            bboxprops=dict(edgecolor=CATEGORY_COLORS_HEX[TASK_TO_CATEGORY[task_labels[i]]],
                           linewidth=1.5, boxstyle="round,pad=0.05"),
            pad=0.05,
        )
        ax.add_artist(ab)
        placed += 1

    ax.legend(fontsize=10, markerscale=3, loc="upper right")
    ax.set_title(f"{title}\n(image: {primary_key}, 1 per {every_n} pts = {placed} shown)",
                 fontsize=11)
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [thumbnail] saved → {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cache ──────────────────────────────────────────────────────────
    print("[1/4] Loading cached features ...")
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        print(f"ERROR: cache not found at {cache_path}")
        print("Run visualize_backbone_tsne.py first (with --skip_cache if needed).")
        return

    npz = np.load(cache_path, allow_pickle=True)
    features_batches = [torch.from_numpy(a) for a in npz["features_batches"]]
    masks_batches    = [torch.from_numpy(a) for a in npz["masks_batches"]]
    task_labels      = list(npz["task_labels"].astype(str))

    if "frame_indices" not in npz:
        print("ERROR: cache does not contain frame_indices.")
        print("Re-run visualize_backbone_tsne.py with --skip_cache to regenerate the cache.")
        return

    frame_indices = list(npz["frame_indices"].astype(int))
    N = len(task_labels)
    print(f"  N={N}, agg_method={args.agg_method}, reduction={args.reduction}")

    # ── Aggregate features ──────────────────────────────────────────────────
    print("[2/4] Aggregating and reducing ...")
    X_raw = aggregate(features_batches, masks_batches, args.agg_method)
    X_2d  = reduce(X_raw, args.reduction,
                   perplexity=args.perplexity,
                   n_neighbors=args.n_neighbors,
                   seed=args.seed)
    print(f"  X_2d shape: {X_2d.shape}")

    # ── Load images ─────────────────────────────────────────────────────────
    print("[3/4] Loading images from dataset ...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )

    # Validate requested image keys
    valid_keys = []
    for key in args.image_keys:
        full = f"observation.images.{key}"
        if full in dataset.meta.features:
            valid_keys.append(key)
        else:
            print(f"  [warn] '{full}' not in dataset features, skipping")
    if not valid_keys:
        print("ERROR: no valid image keys found. Check --image_keys argument.")
        return
    print(f"  image keys to use: {valid_keys}")

    images_dict = load_images_from_dataset(
        dataset, frame_indices, valid_keys, args.thumbnail_px
    )

    # ── Generate visualisations ─────────────────────────────────────────────
    print("[4/4] Generating visualisations ...")
    tag = f"{args.agg_method}_{args.reduction}"
    if args.reduction == "tsne":
        tag += f"_p{args.perplexity}"
    else:
        tag += f"_nb{args.n_neighbors}"

    title = (f"GROOT backbone_features  |  agg={args.agg_method}  |  "
             f"{'t-SNE perp='+str(args.perplexity) if args.reduction=='tsne' else 'UMAP n_neighbors='+str(args.n_neighbors)}")

    # 1. Plotly interactive HTML (hover = images)
    html_path = out_dir / f"{tag}_interactive.html"
    build_plotly_html(
        X_2d, task_labels, frame_indices, images_dict,
        image_keys=valid_keys,
        title=title,
        out_path=html_path,
    )

    # 2. Matplotlib thumbnail plots (one per image key)
    for key in valid_keys:
        mpl_path = out_dir / f"{tag}_thumbnail_{key}.png"
        build_matplotlib_thumbnail(
            X_2d, task_labels, images_dict,
            primary_key=key,
            title=title,
            out_path=mpl_path,
            every_n=args.matplotlib_every,
            thumb_px=args.thumbnail_px // 2,
        )

    print(f"\n✅ Done. Outputs saved to: {out_dir}")
    print(f"   • HTML (hover images) : {html_path.name}")
    for key in valid_keys:
        print(f"   • Thumbnail plot ({key}): {tag}_thumbnail_{key}.png")


if __name__ == "__main__":
    main()
