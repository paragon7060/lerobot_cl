#!/usr/bin/env python3
"""
Visualize VLM backbone features (backbone_features from EagleBackbone)
of a trained GROOT model using t-SNE and UMAP.

Usage:
    conda run -n groot python src/lerobot/scripts/visualize_backbone_tsne.py \
        --checkpoint_dir ./outputs/groot_guide/checkpoints/050000/pretrained_model \
        --dataset_repo_id paragon7060/INSIGHTfixposV3 \
        --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
        --num_samples 512 \
        --batch_size 16 \
        --output_dir ./outputs/tsne_vis
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

# ──────────────────────────────────────────────
# sys.path: sampler / prompt are in scripts dir
# ──────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prompt import SCENE_TASK_PROMPT_GUIDE
from sampler import ProportionalTaskSampler

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
PROMPT_MAP = SCENE_TASK_PROMPT_GUIDE

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

# Keys present in dataset but NOT used by the policy – must be removed so the
# preprocessor (groot_pack_inputs_v3) does not try to normalise them.
POP_KEYS = [
    "observation.images.wrist_semantic",
    "observation.images.right_shoulder_semantic",
    "observation.images.left_shoulder_semantic",
    "observation.images.guide_semantic",
    "observation.images.left_shoulder",
    "observation.images.guide",
]

AGGREGATION_METHODS = ["mask_mean", "mean", "max", "first", "last"]


# ══════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="t-SNE / UMAP of GROOT backbone features")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./outputs/groot_guide/checkpoints/050000/pretrained_model",
        help="Path to pretrained_model dir (contains model.safetensors + config.json)",
    )
    p.add_argument(
        "--dataset_repo_id",
        type=str,
        default="paragon7060/INSIGHTfixposV3",
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default="/home/seonho/workspace/data/paragon7060/INSIGHTfixposV3",
    )
    p.add_argument("--num_samples", type=int, default=512,
                   help="Total frames to collect for visualisation")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  type=str, default="./outputs/tsne_vis")
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument(
        "--tsne_perplexities",
        type=int, nargs="+",
        default=[5, 15, 30, 50, 100],
        help="List of t-SNE perplexity values to try",
    )
    p.add_argument(
        "--tsne_n_iter", type=int, default=1000,
        help="Number of t-SNE iterations",
    )
    p.add_argument(
        "--umap_neighbors",
        type=int, nargs="+",
        default=[5, 15, 30],
        help="List of UMAP n_neighbors values to try (skipped if umap not installed)",
    )
    p.add_argument(
        "--agg_methods",
        type=str, nargs="+",
        default=AGGREGATION_METHODS,
        choices=AGGREGATION_METHODS,
        help="Aggregation methods to try",
    )
    p.add_argument(
        "--skip_cache", action="store_true",
        help="Re-extract features even if cache exists",
    )
    p.add_argument(
        "--pop_keys",
        type=str, nargs="*",
        default=None,
        help="Keys to pop from dataset features/stats. Defaults to POP_KEYS constant if not specified.",
    )
    # Token-level analysis
    p.add_argument(
        "--token_tsne", action="store_true",
        help="Also run t-SNE/UMAP on individual VL tokens (no temporal aggregation). "
             "Outputs go to <output_dir>/token_level/. Existing aggregation results are unaffected.",
    )
    p.add_argument(
        "--token_max_per_sample", type=int, default=20,
        help="Max valid tokens to sample per frame for token-level t-SNE (default: 20).",
    )
    return p.parse_args()


# ══════════════════════════════════════════════
# Dataset loading
# ══════════════════════════════════════════════
def load_dataset(repo_id: str, root: str, checkpoint_dir: str, pop_keys=None):
    """Load LeRobotDataset and pop keys not used by the policy."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        video_backend="pyav",
    )

    keys_to_pop = pop_keys if pop_keys is not None else POP_KEYS
    # Mirror training script: remove image modalities not in policy input_features
    for key in keys_to_pop:
        if key in dataset.meta.features:
            dataset.meta.features.pop(key)
        if key in dataset.meta.stats:
            dataset.meta.stats.pop(key)
        if dataset.delta_indices and key in dataset.delta_indices:
            dataset.delta_indices.pop(key)
        if dataset.delta_timestamps and key in dataset.delta_timestamps:
            dataset.delta_timestamps.pop(key)

    return dataset


# ══════════════════════════════════════════════
# Model + preprocessor loading
# ══════════════════════════════════════════════
_CL_ONLY_FIELDS = {
    "type", "use_contrastive", "contrastive_latent_dim", "contrastive_vlm_input_dim",
    "contrastive_cnn_hidden_dim", "contrastive_proj_hidden_dim", "contrastive_triplet_margin",
    "contrastive_loss_weight", "contrastive_phase", "contrastive_backprop_backbone",
    "contrastive_fallback_to_in_batch", "groot_pretrained_path",
}


def _load_policy_with_vision_lora(checkpoint_dir: Path, device: str):
    """
    Custom loader for checkpoints trained with vision LoRA (lora_target='vision').
    GR00TN15.from_pretrained only supports LLM LoRA, so we:
      1. Load config via PreTrainedConfig.from_pretrained (handles CL subclass dispatch)
      2. Create policy with lora_rank=0 (no LoRA in _create_groot_model)
      3. Manually apply wrap_backbone_lora to the vision model
      4. Load the safetensors checkpoint with strict=False
    """
    import json
    from safetensors.torch import load_model as safetensors_load_model
    from lerobot.configs.policies import PreTrainedConfig

    with open(checkpoint_dir / "config.json") as f:
        config_data = json.load(f)

    lora_rank    = config_data.get("lora_rank", 0)
    lora_alpha   = config_data.get("lora_alpha", 16)
    lora_dropout = config_data.get("lora_dropout", 0.05)
    policy_type  = config_data.get("type", config_data.get("policy_type", "groot"))

    # Load the correct config subclass (GrootCLConfig for groot_cl, GrootConfig for groot)
    config = PreTrainedConfig.from_pretrained(str(checkpoint_dir))

    # Zero out lora_rank so _create_groot_model doesn't apply LLM LoRA
    config.lora_rank = 0

    # Instantiate the right policy class
    if policy_type == "groot_cl":
        from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
        policy = GrootCLPolicy(config)
    else:
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        policy = GrootPolicy(config)

    # Apply LoRA to the correct target(s)
    eagle = policy._groot_model.backbone.eagle_model
    lora_target = config_data.get("lora_target", "vision")
    if lora_target in ("vision", "both"):
        eagle.wrap_backbone_lora(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        print(f"[load] Applied vision LoRA: rank={lora_rank}, alpha={lora_alpha}")
    if lora_target in ("llm", "both"):
        eagle.wrap_llm_lora(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        print(f"[load] Applied LLM LoRA: rank={lora_rank}, alpha={lora_alpha}")

    # Load trained weights (strict=False: CL-head keys are in the model but also in checkpoint,
    # plus a few mismatch keys from LoRA target change are acceptable)
    model_file = str(checkpoint_dir / "model.safetensors")
    missing, unexpected = safetensors_load_model(policy, model_file, strict=False)
    if missing:
        print(f"[load] Missing keys: {len(missing)} (e.g. {list(missing)[:2]})")
    if unexpected:
        print(f"[load] Unexpected keys: {len(unexpected)} (e.g. {list(unexpected)[:2]})")

    return policy


def load_model_and_preprocessor(checkpoint_dir: str, dataset_stats: dict, device: str):
    """
    Load GrootPolicy from a pretrained checkpoint and build the preprocessor
    pipeline using dataset statistics (for normalisation).
    Handles both standard GrootPolicy and CL variants (vision LoRA).
    """
    import json
    from lerobot.policies.factory import make_pre_post_processors

    checkpoint_dir = Path(checkpoint_dir)
    print(f"[load] Loading policy from {checkpoint_dir} ...")

    # Detect vision-LoRA checkpoints (GR00TN15.from_pretrained only handles LLM LoRA)
    with open(checkpoint_dir / "config.json") as f:
        config_data = json.load(f)
    lora_rank  = config_data.get("lora_rank", 0)
    lora_target = config_data.get("lora_target", "llm")

    if lora_rank > 0 and lora_target in ("vision", "both"):
        policy = _load_policy_with_vision_lora(checkpoint_dir, device)
    else:
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        policy = GrootPolicy.from_pretrained(str(checkpoint_dir))

    policy.eval()
    policy = policy.to(device)
    print("[load] Policy loaded.")

    print("[load] Building preprocessor ...")
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_dir),
        dataset_stats=dataset_stats,
    )
    print("[load] Preprocessor ready.")

    return policy, preprocessor


# ══════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════
def _build_backbone_input(processed: dict, device: str, use_bf16: bool):
    """
    From a preprocessed batch dict (already on GPU), extract eagle_* tensors,
    cast to bfloat16 when required, and wrap in a BatchFeature.
    """
    from transformers import BatchFeature

    eagle_data = {}
    for k, v in processed.items():
        if not k.startswith("eagle_"):
            continue
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v) and use_bf16:
                v = v.to(device=device, dtype=torch.bfloat16)
            else:
                v = v.to(device=device)
        eagle_data[k] = v
    return BatchFeature(data=eagle_data)


@torch.no_grad()
def extract_backbone_features(
    policy,
    preprocessor,
    dataloader,
    num_samples: int,
    device: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    """
    Run the backbone on batches from the dataloader and collect:
      - features_batches : list of (B_i, T_i, D) tensors  -- T_i varies per batch
      - masks_batches    : list of (B_i, T_i) tensors
      - task_labels      : list of str (task key, e.g. "3a"), length = sum(B_i)
      - frame_indices    : list of int (global dataset frame index, for image lookup)
    Tensors are float32 on CPU.
    """
    features_batches = []
    masks_batches    = []
    task_labels      = []
    frame_indices    = []

    use_bf16 = getattr(policy.config, "use_bf16", True)
    backbone = policy._groot_model.backbone

    total = 0
    for batch in dataloader:
        if total >= num_samples:
            break

        # ── Save task keys and frame indices BEFORE prompt conversion ──
        raw_task_keys = list(batch["task"])
        raw_indices   = batch["index"].tolist()  # global dataset frame index

        # ── Apply prompt text (same as training script) ──
        batch["task"] = [PROMPT_MAP[t] for t in raw_task_keys]

        # ── Preprocess (normalise, tokenise, collate, move to device) ──
        processed = preprocessor(batch)

        # ── Build backbone input ──
        backbone_input = _build_backbone_input(processed, device, use_bf16)

        # ── Forward through backbone only ──
        autocast_ctx = torch.autocast(
            device_type=device if device != "cpu" else "cpu",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        )
        with autocast_ctx:
            backbone_output = backbone(backbone_input)

        feats = backbone_output["backbone_features"].detach().float().cpu()  # (B, T, D)
        masks = backbone_output["backbone_attention_mask"].detach().cpu()     # (B, T)

        features_batches.append(feats)
        masks_batches.append(masks)
        task_labels.extend(raw_task_keys)
        frame_indices.extend(raw_indices)

        total += feats.shape[0]
        print(f"  collected {min(total, num_samples)}/{num_samples} samples ...", end="\r")

    print()

    # Trim to num_samples (trim the last batch if over-collected)
    if total > num_samples:
        excess = total - num_samples
        last_feats = features_batches[-1]
        last_masks = masks_batches[-1]
        keep = last_feats.shape[0] - excess
        features_batches[-1] = last_feats[:keep]
        masks_batches[-1]    = last_masks[:keep]
        task_labels          = task_labels[:num_samples]
        frame_indices        = frame_indices[:num_samples]

    B0, T0, D = features_batches[0].shape
    print(f"[extract] batches={len(features_batches)}, D={D}, T varies per batch")
    return features_batches, masks_batches, task_labels, frame_indices


# ══════════════════════════════════════════════
# Aggregation methods  list[(B_i,T_i,D)] → (N,D)
# ══════════════════════════════════════════════
def _agg_single(feats: torch.Tensor, masks: torch.Tensor, method: str) -> torch.Tensor:
    """Aggregate one batch: (B, T, D) → (B, D)."""
    if method == "mean":
        return feats.mean(dim=1)

    elif method == "mask_mean":
        m = masks.unsqueeze(-1).float()
        return (feats * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    elif method == "max":
        m = masks.unsqueeze(-1).float()
        masked = feats * m + (1.0 - m) * (-1e9)
        return masked.max(dim=1).values

    elif method == "first":
        return feats[:, 0, :]

    elif method == "last":
        lengths = masks.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, feats.shape[-1])
        return feats.gather(dim=1, index=idx).squeeze(1)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def aggregate(
    features_batches: list[torch.Tensor],
    masks_batches: list[torch.Tensor],
    method: str,
) -> np.ndarray:
    """
    Aggregate variable-T batches per-batch, then concatenate → (N, D) numpy array.
    """
    parts = [
        _agg_single(feats, masks, method)
        for feats, masks in zip(features_batches, masks_batches)
    ]
    return torch.cat(parts, dim=0).numpy()  # (N, D)


# ══════════════════════════════════════════════
# Dimensionality reduction helpers
# ══════════════════════════════════════════════
def sanitize(X: np.ndarray, method_name: str) -> np.ndarray | None:
    """
    Replace NaN/inf with 0 and warn.  If > 50 % of rows are invalid return None
    so the caller can skip this combination.
    """
    bad_rows = np.any(~np.isfinite(X), axis=1)
    n_bad = bad_rows.sum()
    if n_bad > 0:
        print(f"    [warn] {n_bad}/{len(X)} samples have NaN/inf in '{method_name}' – replacing with 0")
        if n_bad > len(X) // 2:
            print(f"    [skip] too many bad rows, skipping this aggregation method")
            return None
        X = X.copy()
        X[bad_rows] = 0.0
    return X


def pca_preprocess(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    Reduce to n_components via PCA before t-SNE / UMAP.
    - Removes zero-variance dimensions before PCA to avoid divide-by-zero.
    - Replaces any residual NaN/inf with 0 after PCA.
    """
    from sklearn.decomposition import PCA

    # 1. Remove constant (zero-variance) feature dimensions
    var = X.var(axis=0)
    nonzero_mask = var > 1e-10
    X_clean = X[:, nonzero_mask]
    if X_clean.shape[1] == 0:
        # All features are constant across all samples — nothing to visualise
        return None

    # 2. Z-score normalisation using only non-zero-variance dims
    mean = X_clean.mean(axis=0)
    std  = X_clean.std(axis=0).clip(min=1e-10)
    X_scaled = (X_clean - mean) / std

    # 3. PCA
    n_comp = min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    # 4. Guard against any residual NaN/inf from numerical issues
    X_pca = np.nan_to_num(X_pca, nan=0.0, posinf=0.0, neginf=0.0)
    return X_pca


def _check_pca_variance(X_pca: np.ndarray, label: str) -> np.ndarray | None:
    """
    Check PCA output for degenerate (zero-std) condition.
    If the first component has near-zero std, add tiny jitter so downstream
    algorithms don't encounter exactly-zero variance.
    Returns None if the whole matrix is essentially constant (all rows same).
    """
    stds = X_pca.std(axis=0)
    if stds[0] < 1e-12:
        print(f"    [warn] {label}: PCA output is near-constant (std={stds[0]:.2e}), adding jitter")
        rng = np.random.RandomState(42)
        X_pca = X_pca + rng.randn(*X_pca.shape) * 1e-6
        # If still degenerate, skip
        if X_pca.std(axis=0)[0] < 1e-12:
            print(f"    [skip] {label}: still degenerate after jitter, skipping")
            return None
    return X_pca


def run_tsne(X: np.ndarray, perplexity: int, n_iter: int, seed: int,
             label: str = "") -> np.ndarray | None:
    from sklearn.manifold import TSNE
    import sklearn

    # Reduce with PCA first (D=2048 is too large for t-SNE directly)
    X_pca = pca_preprocess(X, n_components=50)
    if X_pca is None:
        print(f"    [skip] {label}: all features are constant across samples")
        return None
    X_pca = _check_pca_variance(X_pca, label or "t-SNE")
    if X_pca is None:
        return None

    # Clamp perplexity to valid range
    max_perp = (X_pca.shape[0] - 1) / 3.0
    if perplexity >= max_perp:
        perplexity = max(5.0, max_perp - 1)
        print(f"    [t-SNE] perplexity clamped to {perplexity:.1f}")

    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        verbose=0,
    )
    # sklearn ≥ 1.5 renamed n_iter → max_iter
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
        tsne_kwargs["max_iter"] = n_iter
    else:
        tsne_kwargs["n_iter"] = n_iter
    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(X_pca)


def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float, seed: int,
             label: str = "") -> np.ndarray | None:
    import umap as umap_lib

    # PCA preprocessing also helps UMAP
    X_pca = pca_preprocess(X, n_components=50)
    if X_pca is None:
        print(f"    [skip] {label}: all features are constant across samples")
        return None
    X_pca = _check_pca_variance(X_pca, label or "UMAP")
    if X_pca is None:
        return None

    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        verbose=False,
    )
    return reducer.fit_transform(X_pca)


# ══════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════
def _task_colors(task_ids_sorted):
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(task_ids_sorted), 1))
    return {t: cmap(i) for i, t in enumerate(task_ids_sorted)}


def plot_by_task(X_2d, task_labels, title, out_path: Path):
    task_ids = sorted(set(task_labels))
    color_map = _task_colors(task_ids)
    labels_arr = np.array(task_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    for tid in task_ids:
        mask = labels_arr == tid
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color_map[tid]],
            label=tid,
            alpha=0.65,
            s=18,
            linewidths=0,
        )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, markerscale=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


def plot_by_category(X_2d, task_labels, title, out_path: Path):
    categories = np.array([TASK_TO_CATEGORY[t] for t in task_labels])

    fig, ax = plt.subplots(figsize=(9, 7))
    for cat, color in CATEGORY_COLORS.items():
        mask = categories == cat
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color,
            label=cat,
            alpha=0.65,
            s=18,
            linewidths=0,
        )
    ax.legend(fontsize=10, markerscale=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


def plot_combined(X_2d, task_labels, title, out_path: Path):
    """Side-by-side: left = by task_id, right = by category."""
    task_ids = sorted(set(task_labels))
    color_map = _task_colors(task_ids)
    labels_arr = np.array(task_labels)
    categories = np.array([TASK_TO_CATEGORY[t] for t in task_labels])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: by task
    ax = axes[0]
    for tid in task_ids:
        mask = labels_arr == tid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[color_map[tid]], label=tid, alpha=0.65, s=15, linewidths=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    ax.set_title("by task_id")
    ax.grid(True, alpha=0.2)

    # Right: by category
    ax = axes[1]
    for cat, color in CATEGORY_COLORS.items():
        mask = categories == cat
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=cat, alpha=0.65, s=15, linewidths=0)
    ax.legend(fontsize=10, markerscale=2)
    ax.set_title("by category")
    ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


# ══════════════════════════════════════════════
# Token-level analysis helpers
# ══════════════════════════════════════════════
def build_token_level_data(
    features_batches: list,
    masks_batches: list,
    task_labels: list,
    max_per_sample: int = 20,
    seed: int = 42,
) -> tuple:
    """
    Expand batches of shape (B, T_vl, D) to per-token rows.

    For each sample, valid tokens (where attention mask == 1) are filtered,
    then up to `max_per_sample` tokens are drawn (uniformly, preserving order).

    Returns:
      X_tokens       : (M, D) float32 numpy array  — one row per token
      token_tasks    : list[str] of length M        — task label of parent sample
      token_pos_norm : (M,) float32 in [0, 1]       — token position / (T_valid - 1)
    """
    rng = np.random.default_rng(seed)

    X_list    = []
    task_list = []
    pos_list  = []

    sample_idx = 0
    for feats, masks in zip(features_batches, masks_batches):
        # feats: (B, T, D), masks: (B, T)  — both on CPU
        B = feats.shape[0]
        for b in range(B):
            task = task_labels[sample_idx]
            sample_idx += 1

            valid_mask  = masks[b].bool()          # (T,)
            valid_feats = feats[b][valid_mask]      # (T_valid, D)
            T_valid     = valid_feats.shape[0]

            if T_valid == 0:
                continue

            # Subsample ≤ max_per_sample positions, keeping them in order
            if T_valid <= max_per_sample:
                chosen = np.arange(T_valid)
            else:
                chosen = np.sort(rng.choice(T_valid, max_per_sample, replace=False))

            chosen_feats = valid_feats[chosen].numpy().astype(np.float32)  # (k, D)
            pos_norm     = chosen / max(T_valid - 1, 1)                    # (k,) in [0,1]

            X_list.append(chosen_feats)
            task_list.extend([task] * len(chosen))
            pos_list.extend(pos_norm.tolist())

    X_tokens = np.vstack(X_list)          # (M, D)
    return X_tokens, task_list, np.array(pos_list, dtype=np.float32)


def plot_by_token_position(X_2d: np.ndarray, pos_norm: np.ndarray, title: str, out_path: Path):
    """Scatter plot where each point is coloured by its normalised token position (viridis)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=pos_norm,
        cmap="viridis",
        alpha=0.45,
        s=8,
        linewidths=0,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("token position (0 = first, 1 = last)", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


def plot_token_combined(X_2d: np.ndarray, task_labels: list, pos_norm: np.ndarray,
                        title: str, out_path: Path):
    """Side-by-side: left = coloured by task, right = coloured by token position."""
    task_ids  = sorted(set(task_labels))
    color_map = _task_colors(task_ids)
    labels_arr = np.array(task_labels)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    for tid in task_ids:
        mask = labels_arr == tid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[color_map[tid]], label=tid, alpha=0.45, s=8, linewidths=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    ax.set_title("by task_id")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=pos_norm, cmap="viridis",
                    alpha=0.45, s=8, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("token position", fontsize=9)
    ax.set_title("by token position")
    ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "features_cache.npz"

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("[1/5] Loading dataset ...")
    dataset = load_dataset(args.dataset_repo_id, args.dataset_root, args.checkpoint_dir, pop_keys=args.pop_keys)

    # ── Load model & preprocessor ─────────────────────────────────────────────
    print("[2/5] Loading model and preprocessor ...")
    policy, preprocessor = load_model_and_preprocessor(
        args.checkpoint_dir,
        dataset_stats=dataset.meta.stats,
        device=args.device,
    )

    # ── Build DataLoader ──────────────────────────────────────────────────────
    print("[3/5] Building dataloader ...")
    corrupted_path = Path(args.dataset_root) / "all_excluded_indices_seed2.json"
    if corrupted_path.exists():
        with open(corrupted_path) as f:
            corrupted = set(json.load(f))
    else:
        corrupted = set()

    valid_episodes = [i for i in range(dataset.num_episodes) if i not in corrupted]

    sampler = ProportionalTaskSampler(
        dataset=dataset,
        valid_episode_indices=valid_episodes,
        target_proportions={"cabinet": 1.0, "door": 1.0, "bottle": 1.0},
        epoch_size=args.num_samples * 4,  # generate enough frames to draw from
        shuffle=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ── Feature extraction ────────────────────────────────────────────────────
    # Cache stores per-batch arrays (variable T) using numpy object arrays.
    if cache_path.exists() and not args.skip_cache:
        print(f"[4/5] Loading cached features from {cache_path}")
        npz = np.load(cache_path, allow_pickle=True)
        features_batches = [torch.from_numpy(a) for a in npz["features_batches"]]
        masks_batches    = [torch.from_numpy(a) for a in npz["masks_batches"]]
        task_labels      = list(npz["task_labels"])
        frame_indices    = list(npz["frame_indices"].astype(int)) if "frame_indices" in npz else []
    else:
        print("[4/5] Extracting backbone features ...")
        features_batches, masks_batches, task_labels, frame_indices = extract_backbone_features(
            policy=policy,
            preprocessor=preprocessor,
            dataloader=dataloader,
            num_samples=args.num_samples,
            device=args.device,
        )
        # Save as object arrays to handle variable T dimension
        fb_np = np.empty(len(features_batches), dtype=object)
        mb_np = np.empty(len(masks_batches),    dtype=object)
        for i, (f, m) in enumerate(zip(features_batches, masks_batches)):
            fb_np[i] = f.numpy()
            mb_np[i] = m.numpy()
        np.savez_compressed(
            cache_path,
            features_batches=fb_np,
            masks_batches=mb_np,
            task_labels=np.array(task_labels),
            frame_indices=np.array(frame_indices, dtype=np.int64),
        )
        print(f"  [cache] features saved → {cache_path}")

    N = sum(f.shape[0] for f in features_batches)
    D = features_batches[0].shape[-1]
    print(f"  N={N}, D={D}, batches={len(features_batches)}")
    task_dist = {t: task_labels.count(t) for t in sorted(set(task_labels))}
    print(f"  task distribution: {task_dist}")

    # ── Reduce & visualise ─────────────────────────────────────────────────────
    print("[5/5] Running dimensionality reduction and plotting ...")

    # Check UMAP availability
    umap_available = True
    try:
        import umap  # noqa: F401
    except ImportError:
        umap_available = False
        print("  [umap] package not found – skipping UMAP plots")

    for agg_method in args.agg_methods:
        print(f"\n── Aggregation: {agg_method} ──")
        X_raw = aggregate(features_batches, masks_batches, method=agg_method)  # (N, D)
        print(f"  aggregated shape: {X_raw.shape}")

        X = sanitize(X_raw, agg_method)
        if X is None:
            print(f"  [skip] {agg_method} skipped due to too many invalid values")
            continue

        agg_dir = out_dir / agg_method
        agg_dir.mkdir(exist_ok=True)

        # ── t-SNE ──
        for perp in args.tsne_perplexities:
            print(f"  [t-SNE] perplexity={perp} ...")
            try:
                X_2d = run_tsne(X, perplexity=perp, n_iter=args.tsne_n_iter,
                                seed=args.seed, label=f"{agg_method}/tsne_p{perp}")
            except Exception as e:
                print(f"    [error] t-SNE perp={perp} failed: {e}")
                continue
            if X_2d is None:
                continue
            tag = f"tsne_p{perp}"
            base_title = f"backbone_features  |  agg={agg_method}  |  t-SNE perp={perp}"
            plot_combined(
                X_2d, task_labels, base_title,
                agg_dir / f"{tag}_combined.png",
            )
            plot_by_task(
                X_2d, task_labels,
                base_title + " (by task_id)",
                agg_dir / f"{tag}_task.png",
            )
            plot_by_category(
                X_2d, task_labels,
                base_title + " (by category)",
                agg_dir / f"{tag}_category.png",
            )

        # ── UMAP ──
        if umap_available:
            for n_nb in args.umap_neighbors:
                print(f"  [UMAP] n_neighbors={n_nb} ...")
                try:
                    X_2d = run_umap(X, n_neighbors=n_nb, min_dist=0.1,
                                    seed=args.seed, label=f"{agg_method}/umap_nb{n_nb}")
                except Exception as e:
                    print(f"    [error] UMAP n_neighbors={n_nb} failed: {e}")
                    continue
                if X_2d is None:
                    continue
                tag = f"umap_nb{n_nb}"
                base_title = f"backbone_features  |  agg={agg_method}  |  UMAP n_neighbors={n_nb}"
                plot_combined(
                    X_2d, task_labels, base_title,
                    agg_dir / f"{tag}_combined.png",
                )
                plot_by_task(
                    X_2d, task_labels,
                    base_title + " (by task_id)",
                    agg_dir / f"{tag}_task.png",
                )
                plot_by_category(
                    X_2d, task_labels,
                    base_title + " (by category)",
                    agg_dir / f"{tag}_category.png",
                )

    # ── Token-level t-SNE / UMAP ─────────────────────────────────────────────
    if args.token_tsne:
        print("\n── Token-level t-SNE / UMAP ──")
        token_dir = out_dir / "token_level"
        token_dir.mkdir(exist_ok=True)

        print(f"  Building per-token data (max {args.token_max_per_sample} tokens/sample) ...")
        X_tokens, token_tasks, token_pos = build_token_level_data(
            features_batches, masks_batches, task_labels,
            max_per_sample=args.token_max_per_sample,
            seed=args.seed,
        )
        M = X_tokens.shape[0]
        print(f"  token-level: M={M} tokens, D={X_tokens.shape[1]}")
        task_dist_tok = {t: token_tasks.count(t) for t in sorted(set(token_tasks))}
        print(f"  task distribution: {task_dist_tok}")

        X_tokens = sanitize(X_tokens, "token_level")
        if X_tokens is None:
            print("  [skip] token_level skipped due to too many invalid values")
        else:
            # ── t-SNE ──
            for perp in args.tsne_perplexities:
                print(f"  [t-SNE] perplexity={perp} ...")
                try:
                    X_2d = run_tsne(X_tokens, perplexity=perp, n_iter=args.tsne_n_iter,
                                    seed=args.seed, label=f"token_level/tsne_p{perp}")
                except Exception as e:
                    print(f"    [error] t-SNE perp={perp} failed: {e}")
                    continue
                if X_2d is None:
                    continue
                tag        = f"tsne_p{perp}"
                base_title = f"backbone_tokens  |  t-SNE perp={perp}  |  M={M}"
                plot_token_combined(
                    X_2d, token_tasks, token_pos,
                    base_title, token_dir / f"{tag}_combined.png",
                )
                plot_by_task(
                    X_2d, token_tasks,
                    base_title + " (by task)",
                    token_dir / f"{tag}_task.png",
                )
                plot_by_token_position(
                    X_2d, token_pos,
                    base_title + " (by token position)",
                    token_dir / f"{tag}_position.png",
                )

            # ── UMAP ──
            if umap_available:
                for n_nb in args.umap_neighbors:
                    print(f"  [UMAP] n_neighbors={n_nb} ...")
                    try:
                        X_2d = run_umap(X_tokens, n_neighbors=n_nb, min_dist=0.1,
                                        seed=args.seed, label=f"token_level/umap_nb{n_nb}")
                    except Exception as e:
                        print(f"    [error] UMAP n_neighbors={n_nb} failed: {e}")
                        continue
                    if X_2d is None:
                        continue
                    tag        = f"umap_nb{n_nb}"
                    base_title = f"backbone_tokens  |  UMAP n_neighbors={n_nb}  |  M={M}"
                    plot_token_combined(
                        X_2d, token_tasks, token_pos,
                        base_title, token_dir / f"{tag}_combined.png",
                    )
                    plot_by_task(
                        X_2d, token_tasks,
                        base_title + " (by task)",
                        token_dir / f"{tag}_task.png",
                    )
                    plot_by_token_position(
                        X_2d, token_pos,
                        base_title + " (by token position)",
                        token_dir / f"{tag}_position.png",
                    )

    print(f"\n✅ Done. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
