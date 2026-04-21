#!/usr/bin/env python3
"""
Joint action_encoder + VLM backbone t-SNE visualization for GR00T policies
trained on robocasa datasets — loads LeRobotDataset directly from local
per-task directories under --dataset_root (no DATASET_SOUP_REGISTRY).

Expected layout:
  {dataset_root}/{split}/task_XXXX/{data,meta,videos}   # LeRobot v3.0 format

Interactive Plotly HTML output with:
  - Task-level coloring; hover shows {split}_{description} / sample idx / frame
  - Legend click toggles individual tasks on/off
  - Group macro buttons: Pretrain-Atomic / Pretrain-Composite /
                         Target-Atomic / Target-Composite  + All ON/OFF
  - A/B two-click distance (cosine/L2 in full-feature space and PCA-50)
  - Per-sample 16-frame video clips side-by-side (A=blue, B=green) per camera
  - --no_zscore flag to skip z-score before PCA
  - --stats_source {local,checkpoint} to switch normalizer stats source

Example:
  ssh seonho@166.104.35.48 "cd ~/ws3/lerobot && \
      /home/seonho/miniconda3/envs/lerobot050_groot/bin/python \
          -m lerobot.scripts.visualize_joint_embedding_robocasa \
          --checkpoint_dir /home/seonho/groot_robocasa/outputs/pretrain/checkpoints/080000/pretrained_model \
          --dataset_root /home/seonho/slicing_robocasa_human_v3 \
          --splits robocasa_target_human_atomic robocasa_target_human_composite \
          --samples_per_task 8 \
          --cache_dir ~/ws3/outputs/robocasa_emb"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

MODALITY_VIDEO_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]
MODALITY_STATE_KEYS = [
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
]
MODALITY_ACTION_KEYS = [
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
]

# video.X → observation.images.X so the policy preprocessor sees expected keys
VIDEO_TO_IMAGE_KEY = {vk: f"observation.images.{vk[len('video.'):]}" for vk in MODALITY_VIDEO_KEYS}

DEFAULT_CAM_KEYS = ["robot0_eye_in_hand", "robot0_agentview_right"]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str,
                   default="/home/seonho/groot_robocasa/outputs/pretrain/checkpoints/080000/pretrained_model")
    p.add_argument("--dataset_root", type=str,
                   default="/home/seonho/slicing_robocasa_human_v3")
    p.add_argument("--splits", type=str, nargs="+",
                   default=["robocasa_pretrain_human_atomic",
                            "robocasa_pretrain_human_composite",
                            "robocasa_target_human_atomic",
                            "robocasa_target_human_composite"])
    p.add_argument("--samples_per_task", type=int, default=8)
    p.add_argument("--stats_source", type=str, default="local",
                   choices=["local", "checkpoint"],
                   help="local: meta/stats.json weighted avg; checkpoint: make_pre_post_processors stats=None")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cache_dir", type=str, default="./outputs/robocasa_joint_emb")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--action_agg", type=str, default="flatten",
                   choices=["mean", "max", "last", "first_last", "delta_total", "delta_mean",
                            "flatten", "first_mid_last", "raw_flatten", "raw_delta_mean",
                            "raw_delta_total"])
    p.add_argument("--backbone_agg", type=str, default="mask_mean",
                   choices=["mask_mean", "mean", "max", "first", "last"])
    p.add_argument("--perplexity", type=int, default=30)
    p.add_argument("--tsne_n_iter", type=int, default=1000)
    p.add_argument("--cam_keys", type=str, nargs="*", default=DEFAULT_CAM_KEYS,
                   help="Short cam names (without 'video.' prefix); default eye_in_hand + agentview_right")
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--clip_size", type=str, default="320x240")
    p.add_argument("--skip_clips", action="store_true")
    p.add_argument("--skip_feature_cache", action="store_true",
                   help="Re-extract features even if cache exists")
    p.add_argument("--no_zscore", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Task description loader
# ══════════════════════════════════════════════════════════════════════════════

def load_task_description(task_dir: Path) -> str:
    """meta/tasks.parquet → task description string (첫 번째 row)."""
    import pandas as pd
    p = task_dir / "meta" / "tasks.parquet"
    if not p.exists():
        return task_dir.name  # fallback: dir name
    df = pd.read_parquet(p)
    if "task" in df.columns and len(df) > 0:
        return str(df["task"].iloc[0])
    return task_dir.name


# ══════════════════════════════════════════════════════════════════════════════
# Split → per-split datasets
# ══════════════════════════════════════════════════════════════════════════════

def load_split_datasets(dataset_root: str, splits: list[str]) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_root)
    result = {}

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"[skip] split not found: {split_dir}")
            continue

        task_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        print(f"[split={split}] {len(task_dirs)} task dirs")

        loaded = []
        for task_dir in task_dirs:
            if not (task_dir / "meta" / "info.json").exists():
                continue
            description = load_task_description(task_dir)
            task_label = f"{split}_{description}"

            try:
                ds = LeRobotDataset(
                    repo_id="local",
                    root=str(task_dir),
                    video_backend="pyav",
                )
            except Exception:
                try:
                    import json as _json
                    with open(task_dir / "meta" / "info.json") as f:
                        info = _json.load(f)
                    repo_id = info.get("repo_id", task_dir.name)
                    ds = LeRobotDataset(
                        repo_id=repo_id,
                        root=str(task_dir),
                        video_backend="pyav",
                    )
                except Exception as e:
                    print(f"  [skip] {task_dir.name}: {e}")
                    continue

            loaded.append((ds, task_label, task_dir))

        result[split] = loaded
        print(f"  loaded {len(loaded)}/{len(task_dirs)} tasks for {split}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Sample manifest (per-split, samples_per_task)
# ══════════════════════════════════════════════════════════════════════════════

def build_manifest_local(split_to_datasets: dict, samples_per_task: int, seed: int) -> list[dict]:
    import random
    rng = random.Random(seed)
    manifest = []

    for split, entries in split_to_datasets.items():
        for ds, task_label, task_dir in entries:
            n = len(ds)
            if n == 0:
                continue
            idxs = rng.sample(range(n), min(samples_per_task, n))
            for ds_idx in idxs:
                sample = ds[ds_idx]
                ep_idx = int(sample["episode_index"].item()) if hasattr(sample["episode_index"], "item") else int(sample["episode_index"])
                frame_idx = int(sample["frame_index"].item()) if hasattr(sample["frame_index"], "item") else int(sample["frame_index"])
                manifest.append({
                    "split": split,
                    "task": task_label,
                    "dataset": ds,
                    "task_dir": task_dir,
                    "ds_index": ds_idx,
                    "episode_index": ep_idx,
                    "frame_index": frame_idx,
                })

    rng.shuffle(manifest)
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# Batch assembly (manifest → policy input batch)
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_one_sample(raw: dict) -> dict:
    out = {}

    # Images
    for k, v in raw.items():
        if k.startswith("observation.images."):
            if isinstance(v, torch.Tensor):
                t = v.float()
                if t.ndim == 4:   # (T, C, H, W) → take t=0
                    t = t[0]
                if t.max() > 1.5:  # uint8 range → normalize
                    t = t / 255.0
            else:
                arr = np.asarray(v)
                if arr.ndim == 4:
                    arr = arr[0]
                t = torch.from_numpy(arr)
                if arr.ndim == 3 and arr.shape[-1] == 3:  # (H,W,C) → (C,H,W)
                    t = t.permute(2, 0, 1)
                t = t.float()
                if t.max() > 1.5:
                    t = t / 255.0
            out[k] = t

    # State
    state = raw.get("observation.state")
    if state is not None:
        t = torch.as_tensor(state, dtype=torch.float32)
        if t.ndim == 2:
            t = t[0]
        out["observation.state"] = t

    # Action
    action = raw.get("action")
    if action is not None:
        out["action"] = torch.as_tensor(action, dtype=torch.float32)

    # Task prompt
    task = raw.get("task", raw.get("annotation.human.task_description", ""))
    if isinstance(task, (list, np.ndarray)):
        task = task[0] if len(task) > 0 else ""
    out["task"] = str(task)

    return out


def _collate(samples: list[dict]) -> dict:
    """Batch-stack the per-sample dicts. Tensors are stacked; strings → list."""
    keys = samples[0].keys()
    batch = {}
    for k in keys:
        vals = [s[k] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            batch[k] = torch.stack(vals, dim=0)
        else:
            batch[k] = vals
    return batch


# ══════════════════════════════════════════════════════════════════════════════
# Stats construction (needed by policy preprocessor normalizer)
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset_stats_local(split_to_datasets: dict) -> dict:
    import json as _json
    state_means, state_stds, state_ws = [], [], []
    action_means, action_stds, action_ws = [], [], []

    for split, entries in split_to_datasets.items():
        for ds, task_label, task_dir in entries:
            stats_path = task_dir / "meta" / "stats.json"
            if not stats_path.exists():
                continue
            with open(stats_path) as f:
                stats = _json.load(f)
            w = len(ds)

            if "observation.state" in stats:
                s = stats["observation.state"]
                state_means.append(np.array(s["mean"], dtype=np.float32))
                state_stds.append(np.array(s["std"],  dtype=np.float32))
                state_ws.append(w)

            if "action" in stats:
                s = stats["action"]
                action_means.append(np.array(s["mean"], dtype=np.float32))
                action_stds.append(np.array(s["std"],  dtype=np.float32))
                action_ws.append(w)

    def _wavg(means, ws):
        arr = np.stack(means)
        w = np.array(ws, dtype=np.float64)
        w /= w.sum()
        return (arr * w[:, None]).sum(axis=0).astype(np.float32)

    def _pooled_std(means, stds, ws):
        means = np.stack(means); stds = np.stack(stds)
        w = np.array(ws, dtype=np.float64); w /= w.sum()
        mu = (means * w[:, None]).sum(axis=0)
        var = (w[:, None] * (stds**2 + (means - mu)**2)).sum(axis=0)
        return np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32)

    dataset_stats = {}

    if state_means:
        dataset_stats["observation.state"] = {
            "mean": _wavg(state_means, state_ws),
            "std":  _pooled_std(state_means, state_stds, state_ws),
        }
    if action_means:
        dataset_stats["action"] = {
            "mean": _wavg(action_means, action_ws),
            "std":  _pooled_std(action_means, action_stds, action_ws),
        }

    # image: identity normalization
    for ik in VIDEO_TO_IMAGE_KEY.values():
        dataset_stats[ik] = {
            "mean": np.zeros(3, dtype=np.float32),
            "std":  np.ones(3,  dtype=np.float32),
        }

    return dataset_stats


# ══════════════════════════════════════════════════════════════════════════════
# Policy loading
# ══════════════════════════════════════════════════════════════════════════════

def load_policy_and_preprocessor(checkpoint_dir: str, dataset_stats: dict | None, device: str):
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.factory import make_pre_post_processors

    ckpt = Path(checkpoint_dir)
    print(f"[load] Loading policy from {ckpt} ...")
    policy = GrootPolicy.from_pretrained(str(ckpt))
    policy.eval().to(device)
    print(f"[load] Building preprocessor ...")
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        dataset_stats=dataset_stats,
    )
    print(f"[load] Done.")
    return policy, preprocessor


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _build_backbone_input(processed: dict, device: str, use_bf16: bool):
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
def extract_features(policy, preprocessor, manifest: list[dict],
                     batch_size: int, device: str):
    """
    Iterate manifest in batches. For each batch:
      - fetch samples from their per-task datasets
      - flatten/rename to policy input format
      - preprocess, then run BOTH:
         (a) backbone forward on eagle_* tensors → (B, T_vl, D_bb) + mask
         (b) action_encoder on prepared action inputs at t=999 → (B, T_action, D_enc)
    Returns:
      action_feats   : np.ndarray (N, T_action, D_enc)
      gt_actions     : np.ndarray (N, T_action, action_dim)
      bb_feats_list  : list[torch.Tensor (B_i, T_vl_i, D_bb)]
      bb_masks_list  : list[torch.Tensor (B_i, T_vl_i)]
      meta: dict with task_labels, split_labels, frame_indices, traj_ids, base_idxs, dataset_paths
    """
    use_bf16 = getattr(policy.config, "use_bf16", True)
    dev_type = "cuda" if device != "cpu" else "cpu"
    action_head = policy._groot_model.action_head
    action_encoder = action_head.action_encoder
    backbone = policy._groot_model.backbone

    ac_list, gt_list = [], []
    bb_f_list, bb_m_list = [], []

    N_total = len(manifest)
    for start in range(0, N_total, batch_size):
        chunk = manifest[start:start + batch_size]
        # ── Gather raw samples ──
        samples = []
        for m in chunk:
            raw = m["dataset"][m["ds_index"]]
            samples.append(_flatten_one_sample(raw))
        # ── Attach task prompts per sample (preprocessor expects string list in "task") ──
        batch = _collate(samples)
        # move tensors to device via preprocessor (it handles device_processor)
        processed = preprocessor(batch)

        # ── Backbone forward ──
        bin = _build_backbone_input(processed, device, use_bf16)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
            bb_out = backbone(bin)
        bb_f_list.append(bb_out["backbone_features"].detach().float().cpu())
        bb_m_list.append(bb_out["backbone_attention_mask"].detach().cpu())

        # ── Action encoder forward (clean t=999) ──
        _, action_inputs = policy._groot_model.prepare_input(processed)
        gt_actions = action_inputs.action                   # (B, T_action, action_dim)
        embodiment_id = action_inputs.embodiment_id
        B = gt_actions.shape[0]
        t_clean = torch.full((B,), 999, dtype=torch.long, device=gt_actions.device)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
            enc_out = action_encoder(gt_actions, t_clean, embodiment_id)
        ac_list.append(enc_out.float().cpu())
        gt_list.append(gt_actions.float().cpu())

        print(f"  [extract] {min(start + B, N_total)}/{N_total} ...", end="\r")
    print()

    action_feats = np.concatenate([x.numpy() for x in ac_list], axis=0)
    gt_actions = np.concatenate([x.numpy() for x in gt_list], axis=0)
    meta = {
        "task_labels":   [m["task"]          for m in manifest],
        "split_labels":  [m["split"]         for m in manifest],
        "ds_indices":    [m["ds_index"]      for m in manifest],
        "traj_ids":      [m["episode_index"] for m in manifest],
        "base_idxs":     [m["frame_index"]   for m in manifest],
        "dataset_paths": [str(m["task_dir"]) for m in manifest],
    }
    return action_feats, gt_actions, bb_f_list, bb_m_list, meta


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation + t-SNE  (reused from visualize_joint_embedding_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

def agg_action(feats: np.ndarray, method: str, gt_actions: np.ndarray | None = None) -> np.ndarray:
    if method.startswith("raw_"):
        if gt_actions is None:
            raise ValueError(f"method='{method}' requires gt_actions")
        t = torch.from_numpy(gt_actions)
        N, T, D = t.shape
        if method == "raw_flatten":        out = t.reshape(N, T * D)
        elif method == "raw_delta_mean":   out = (t[:, 1:, :] - t[:, :-1, :]).mean(dim=1)
        elif method == "raw_delta_total":  out = t[:, -1, :] - t[:, 0, :]
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


def agg_backbone(feats_list, masks_list, method):
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
            idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, f.shape[-1])
            agg = f.gather(dim=1, index=idx).squeeze(1)
        else: raise ValueError(method)
        parts.append(agg)
    return torch.cat(parts, dim=0).numpy()


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
    from sklearn.manifold import TSNE
    import sklearn
    X_pca = pca_preprocess(X, zscore=zscore)
    if X_pca is None:
        print(f"  [skip] {label}: all constant")
        return None, None
    if X_pca.std(axis=0)[0] < 1e-12:
        X_pca = X_pca + np.random.RandomState(seed).randn(*X_pca.shape) * 1e-6
    perplexity = min(perplexity, max(5.0, (X_pca.shape[0] - 1) / 3.0 - 1))
    kw = dict(n_components=2, perplexity=perplexity, random_state=seed, verbose=0)
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
        kw["max_iter"] = n_iter
    else:
        kw["n_iter"] = n_iter
    print(f"  [t-SNE] {label} shape={X_pca.shape} perp={perplexity:.0f} zscore={zscore} ...")
    X_2d = TSNE(**kw).fit_transform(X_pca)
    return X_pca, X_2d


# ══════════════════════════════════════════════════════════════════════════════
# Pairwise distance (base64)
# ══════════════════════════════════════════════════════════════════════════════

def _upper_tri_b64(mat: np.ndarray) -> str:
    idx = np.triu_indices(mat.shape[0], k=1)
    return base64.b64encode(mat[idx].astype(np.float32).tobytes()).decode("ascii")


def compute_dist_matrices(X_action_full, X_backbone_full, X_pca_action, X_pca_backbone):
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
    out = {}
    for name, X in [("action_full", X_action_full), ("action_pca", X_pca_action),
                    ("bb_full", X_backbone_full),   ("bb_pca",    X_pca_backbone)]:
        print(f"  [dist] {name} shape={X.shape} ...", end=" ", flush=True)
        c = cosine_distances(X); l = euclidean_distances(X)
        out[f"{name}_cos"] = _upper_tri_b64(c)
        out[f"{name}_l2"]  = _upper_tri_b64(l)
        nbytes = len(out[f"{name}_cos"]) + len(out[f"{name}_l2"])
        print(f"~{nbytes/1e6:.1f} MB")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Video clip extraction (lerobot local format)
# ══════════════════════════════════════════════════════════════════════════════

def _get_video_path(task_dir: Path, obs_key: str, episode_index: int) -> Path | None:
    chunk_size = 1000
    chunk = episode_index // chunk_size
    candidates = [
        task_dir / "videos" / f"chunk-{chunk:03d}" / obs_key / f"episode_{episode_index:06d}.mp4",
        task_dir / "videos" / obs_key / f"episode_{episode_index:06d}.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    # glob fallback
    matches = list(task_dir.glob(f"videos/**/*{episode_index:06d}.mp4"))
    if matches:
        return matches[0]
    return None


def extract_video_clips_local(manifest: list[dict], cam_short_keys: list[str],
                               horizon: int, out_dir: Path,
                               fps: int = 20, output_size=(320, 240)) -> dict:
    import av
    import io as _io
    from PIL import Image as PILImage

    W, H = output_size
    cam_to_dir = {}
    for cam_short in cam_short_keys:
        cam_dir = out_dir / cam_short
        cam_dir.mkdir(parents=True, exist_ok=True)
        cam_to_dir[cam_short] = cam_dir

    for sample_idx, m in enumerate(manifest):
        task_dir = m["task_dir"]
        ep_idx = m["episode_index"]
        frame_idx = m["frame_index"]

        for cam_short in cam_short_keys:
            out_file = cam_to_dir[cam_short] / f"{sample_idx:04d}.mp4"
            if out_file.exists():
                continue

            obs_key = f"observation.images.{cam_short}"
            vpath = _get_video_path(task_dir, obs_key, ep_idx)
            if vpath is None:
                continue

            start_ts = frame_idx / fps
            try:
                with av.open(str(vpath)) as vc:
                    stream = vc.streams.video[0]
                    tb = float(stream.time_base)
                    pts = int(start_ts / tb)
                    vc.seek(pts, stream=stream, backward=True)

                    raw_frames = []
                    for pkt in vc.demux(stream):
                        for frm in pkt.decode():
                            fts = float(frm.pts * tb) if frm.pts is not None else 0.0
                            if fts < start_ts - 1.0 / fps:
                                continue
                            raw_frames.append(frm.to_ndarray(format="rgb24"))
                            if len(raw_frames) >= horizon:
                                break
                        if len(raw_frames) >= horizon:
                            break
            except Exception as e:
                print(f"  [clips] {cam_short}/{sample_idx}: {e}")
                continue

            if not raw_frames:
                continue

            frames_resized = []
            for rf in raw_frames[:horizon]:
                pil = PILImage.fromarray(rf)
                if pil.size != (W, H):
                    pil = pil.resize((W, H), PILImage.LANCZOS)
                frames_resized.append(np.array(pil))

            buf = _io.BytesIO()
            with av.open(buf, mode="w", format="mp4") as oc:
                ost = oc.add_stream("libx264", rate=fps)
                ost.width = W; ost.height = H; ost.pix_fmt = "yuv420p"
                ost.options = {"crf": "28", "preset": "ultrafast"}
                for rf in frames_resized:
                    avf = av.VideoFrame.from_ndarray(rf, format="rgb24")
                    for pkt in ost.encode(avf):
                        oc.mux(pkt)
                for pkt in ost.encode(None):
                    oc.mux(pkt)
            with open(out_file, "wb") as f:
                f.write(buf.getvalue())

        if (sample_idx + 1) % 50 == 0:
            print(f"  [clips] {sample_idx+1}/{len(manifest)}", end="\r")
    print()
    return cam_to_dir


# ══════════════════════════════════════════════════════════════════════════════
# Static matplotlib overview plot (by split only — task-level only in HTML)
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview_by_split(X2d_a, X2d_b, split_labels, perplexity, out_path):
    splits = sorted(set(split_labels))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(max(len(splits), 1))
    color = {s: cmap(i) for i, s in enumerate(splits)}
    arr = np.array(split_labels)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, X2d, title in [(axes[0], X2d_a, "action_encoder"), (axes[1], X2d_b, "backbone")]:
        for s in splits:
            m = arr == s
            ax.scatter(X2d[m, 0], X2d[m, 1], c=[color[s]], label=s, s=12, alpha=0.7, linewidths=0)
        ax.legend(fontsize=9, markerscale=2)
        ax.set_title(title); ax.grid(True, alpha=0.2)
    fig.suptitle(f"by split  |  t-SNE perp={perplexity}  |  N={len(split_labels)}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Group classification for updatemenus
# ══════════════════════════════════════════════════════════════════════════════

def classify_split_group(split: str) -> str:
    s = split.lower()
    if "pretrain" in s and "atomic" in s:    return "Pretrain-Atomic"
    if "pretrain" in s and "composite" in s: return "Pretrain-Composite"
    if "target" in s and "atomic" in s:      return "Target-Atomic"
    if "target" in s and "composite" in s:   return "Target-Composite"
    return "Other"


# ══════════════════════════════════════════════════════════════════════════════
# Interactive HTML  (task-name colors + group ON/OFF + A/B distance + videos)
# ══════════════════════════════════════════════════════════════════════════════

def _task_color_map(task_ids: list[str]) -> dict:
    n = max(len(task_ids), 1)
    if n <= 20:
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n)
    else:
        cmap = matplotlib.colormaps.get_cmap("gist_rainbow").resampled(n)
    return {t: f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
            for t, (r, g, b, _) in
            {tid: cmap(i) for i, tid in enumerate(task_ids)}.items()}


def make_joint_html(X2d_action, X2d_backbone,
                    X_pca_action, X_pca_backbone,
                    task_labels, split_labels,
                    frame_indices, episode_progress,
                    action_agg, backbone_agg, perplexity,
                    out_path: Path,
                    clips_dirs: dict | None = None,
                    zscore: bool = True,
                    dist_b64: dict | None = None):
    N = len(task_labels)
    # Group traces by (split, task) so user can toggle individual tasks AND whole groups
    pair_to_idx = defaultdict(list)
    for i, (s, t) in enumerate(zip(split_labels, task_labels)):
        pair_to_idx[(s, t)].append(i)
    pairs = sorted(pair_to_idx.keys())

    # Task color (stable across plots)
    task_ids = sorted(set(task_labels))
    task_color = _task_color_map(task_ids)

    # Classify each pair's split into a macro group for updatemenus
    group_of_trace = []
    for (s, t) in pairs:
        group_of_trace.append(classify_split_group(s))
    groups_present = sorted(set(group_of_trace))

    def _make_traces(X2d):
        traces = []
        for (s, t) in pairs:
            idxs = pair_to_idx[(s, t)]
            traces.append({
                "type": "scatter",
                "x": X2d[idxs, 0].tolist(),
                "y": X2d[idxs, 1].tolist(),
                "mode": "markers",
                "name": f"{t} ({s})",
                "legendgroup": t,
                "marker": {"color": task_color[t], "size": 6, "opacity": 0.85},
                "customdata": [[t, s, i, frame_indices[i]] for i in idxs],
                "hovertemplate": (
                    "<b>%{customdata[0]}</b><br>"
                    "split: %{customdata[1]}<br>"
                    "sample: %{customdata[2]}<br>"
                    "frame: %{customdata[3]}<extra></extra>"
                ),
            })
        # A highlight ring
        traces.append({
            "type": "scatter", "x": [], "y": [], "mode": "markers",
            "name": "A", "showlegend": False, "hoverinfo": "skip",
            "marker": {"symbol": "circle-open", "size": 22, "color": "#2196F3",
                       "line": {"color": "#2196F3", "width": 3}},
        })
        # B highlight ring
        traces.append({
            "type": "scatter", "x": [], "y": [], "mode": "markers",
            "name": "B", "showlegend": False, "hoverinfo": "skip",
            "marker": {"symbol": "circle-open", "size": 22, "color": "#4CAF50",
                       "line": {"color": "#4CAF50", "width": 3}},
        })
        return traces

    n_data_traces = len(pairs)
    action_traces = _make_traces(X2d_action)
    backbone_traces = _make_traces(X2d_backbone)

    # ── updatemenus buttons: All ON/OFF + per-group ON/OFF ─────────────────────
    def _visibility_pattern(on_groups: set) -> list:
        """Length = n_data_traces + 2 (A/B rings). True/legendonly for data, True for rings."""
        out = []
        for g in group_of_trace:
            out.append(True if g in on_groups else "legendonly")
        out += [True, True]
        return out

    buttons = []
    # All ON / All OFF
    buttons.append({
        "label": "All ON", "method": "restyle",
        "args": [{"visible": _visibility_pattern(set(groups_present))}],
    })
    buttons.append({
        "label": "All OFF", "method": "restyle",
        "args": [{"visible": _visibility_pattern(set())}],
    })
    # Per group ON/OFF
    for g in groups_present:
        buttons.append({
            "label": f"{g} ON", "method": "restyle",
            "args": [{"visible": _visibility_pattern({g} | set(groups_present))}],
        })
        buttons.append({
            "label": f"{g} OFF", "method": "restyle",
            "args": [{"visible": _visibility_pattern(set(groups_present) - {g})}],
        })

    updatemenus = [{
        "type": "buttons",
        "direction": "right",
        "buttons": buttons,
        "x": 0.0, "xanchor": "left",
        "y": 1.15, "yanchor": "top",
        "pad": {"r": 4, "t": 4}, "showactive": False,
        "font": {"size": 10},
    }]

    shared_layout = {
        "height": 650,
        "hovermode": "closest",
        "xaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
        "yaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
        "legend": {
            "x": 1.01, "y": 1, "xanchor": "left",
            "font": {"size": 9},
            "itemclick": "toggle", "itemdoubleclick": "toggleothers",
            "groupclick": "togglegroup",
            "tracegroupgap": 1,
        },
        "margin": {"l": 50, "r": 260, "t": 80, "b": 40},
        "updatemenus": updatemenus,
    }
    action_layout = dict(shared_layout, title=f"action_encoder ({action_agg})")
    backbone_layout = dict(shared_layout, title=f"backbone ({backbone_agg})")

    # ── Video panel ──
    has_clips = bool(clips_dirs)
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

    action_xy = X2d_action.tolist()
    backbone_xy = X2d_backbone.tolist()
    pca_act_js = X_pca_action.tolist()
    pca_bb_js = X_pca_backbone.tolist()
    has_dist = bool(dist_b64)
    dist_b64_json = json.dumps(dist_b64) if has_dist else "null"

    title = (f"action_encoder ({action_agg}) vs backbone ({backbone_agg})"
             f"  |  t-SNE perp={perplexity}  |  N={N}  |  tasks={len(task_ids)}"
             + ("  |  no_zscore" if not zscore else ""))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
 body {{ font-family: sans-serif; margin: 0; padding: 6px; background: #f5f5f5; }}
 h3 {{ text-align: center; margin: 4px 0 6px; font-size: 13px; color: #333; }}
 #plots-row {{ display: flex; gap: 6px; }}
 #plots-row > div {{ flex: 1; min-width: 0; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
 #bottom-row {{ display: flex; gap: 10px; margin-top: 8px; align-items: flex-start; }}
 #video-panel {{ display: flex; gap: 12px; padding: 8px 10px;
                 background: #fff; border: 1px solid #ddd; border-radius: 6px;
                 flex: 3; min-height: 80px; flex-wrap: wrap; }}
 .cam-group {{ display: flex; flex-direction: column; align-items: center; }}
 .cam-label {{ margin: 0 0 3px; font-size: 11px; font-weight: bold; color: #333; }}
 .cam-row {{ display: flex; gap: 6px; }}
 .vid-box {{ text-align: center; }}
 .vid-ab-label {{ margin: 0 0 2px; font-size: 12px; font-weight: bold; }}
 .vid-box video {{ width: 160px; height: auto; border: 2px solid #ccc; border-radius: 4px; background: #111; }}
 #right-panel {{ flex: 1; display: flex; flex-direction: column; gap: 6px; min-width: 260px; }}
 #info-box {{ padding: 8px 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px;
              font-size: 12px; color: #444; white-space: pre-line; }}
 #dist-panel {{ padding: 8px 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; font-size: 11px; color: #333; }}
 #dist-panel h4 {{ margin: 0 0 6px; font-size: 12px; color: #555; }}
 .dist-section {{ margin-bottom: 8px; }}
 .dist-section-title {{ font-weight: bold; font-size: 10px; color: #888;
                        text-transform: uppercase; margin-bottom: 3px;
                        border-bottom: 1px solid #eee; padding-bottom: 1px; }}
 .dist-row {{ display: flex; justify-content: space-between; margin: 2px 0; }}
 .dist-label {{ color: #666; }}
 .dist-value {{ font-weight: bold; font-family: monospace; }}
 .dist-value.full {{ color: #1565C0; }}
 .dist-value.pca {{ color: #888; }}
 #mode-hint {{ text-align: center; font-size: 11px; color: #888; margin: 3px 0; }}
</style></head><body>
<h3>{title}</h3>
<div id="mode-hint">click 1: A (blue) &nbsp;|&nbsp; click 2: B (green) + distance &nbsp;|&nbsp; same point: reset</div>
<div id="plots-row">
 <div id="plot_action"></div>
 <div id="plot_backbone"></div>
</div>
<div id="bottom-row">
 <div id="video-panel">
{video_panel_html}
  <span id="vid-hint" style="color:#aaa;font-style:italic;font-size:12px;">&#8592; click a point to play clip</span>
 </div>
 <div id="right-panel">
  <div id="info-box"><span style="color:#aaa;font-style:italic;">click a point for details</span></div>
  <div id="dist-panel" style="display:none;">
   <h4>&#128207; Feature Distance</h4>
   <div class="dist-section"><div class="dist-section-title">Action Encoder ({action_agg})</div>
    <div class="dist-row"><span class="dist-label">Cos full</span><span class="dist-value full" id="d_act_full_cos">—</span></div>
    <div class="dist-row"><span class="dist-label">L2 full</span><span class="dist-value full" id="d_act_full_l2">—</span></div>
    <div class="dist-row"><span class="dist-label">Cos PCA-50</span><span class="dist-value pca" id="d_act_pca_cos">—</span></div>
    <div class="dist-row"><span class="dist-label">L2 PCA-50</span><span class="dist-value pca" id="d_act_pca_l2">—</span></div>
   </div>
   <div class="dist-section"><div class="dist-section-title">Backbone ({backbone_agg})</div>
    <div class="dist-row"><span class="dist-label">Cos full</span><span class="dist-value full" id="d_bb_full_cos">—</span></div>
    <div class="dist-row"><span class="dist-label">L2 full</span><span class="dist-value full" id="d_bb_full_l2">—</span></div>
    <div class="dist-row"><span class="dist-label">Cos PCA-50</span><span class="dist-value pca" id="d_bb_pca_cos">—</span></div>
    <div class="dist-row"><span class="dist-label">L2 PCA-50</span><span class="dist-value pca" id="d_bb_pca_l2">—</span></div>
   </div>
  </div>
 </div>
</div>
<script>
var N            = {N};
var N_DATA       = {n_data_traces};
var actionXY     = {json.dumps(action_xy)};
var backboneXY   = {json.dumps(backbone_xy)};
var pcaAction    = {json.dumps(pca_act_js)};
var pcaBackbone  = {json.dumps(pca_bb_js)};
var taskLabels   = {json.dumps(task_labels)};
var splitLabels  = {json.dumps(split_labels)};
var frameIdxs    = {json.dumps(frame_indices)};
var camShorts    = {json.dumps(cam_shorts)};

var distB64Raw = {dist_b64_json};
var distMat = {{}};
function decodeDistMatrix(b64) {{
  var bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}
if (distB64Raw !== null) {{
  for (var key in distB64Raw) distMat[key] = decodeDistMatrix(distB64Raw[key]);
}}
function triIdx(i, j) {{ if (i > j) {{ var t = i; i = j; j = t; }} return i * (N - 1) - Math.floor(i * (i - 1) / 2) + (j - i - 1); }}
function getDist(key, i, j) {{ if (i === j) return 0.0; return distMat[key][triIdx(i, j)]; }}
function cosineDistPCA(a, b) {{ var d=0, na=0, nb=0; for (var i=0;i<a.length;i++){{ d+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }} return 1.0 - Math.max(-1, Math.min(1, d/(Math.sqrt(na)*Math.sqrt(nb)+1e-10))); }}
function l2DistPCA(a, b) {{ var s=0; for (var i=0;i<a.length;i++){{ var d=a[i]-b[i]; s+=d*d; }} return Math.sqrt(s); }}

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
    var el = document.getElementById('vid_'+slot+'_'+cs);
    if (el) {{ el.src = './clips/'+cs+'/'+pad4(sampleIdx)+'.mp4';
              el.style.border = '2px solid '+(slot==='A' ? '#2196F3' : '#4CAF50');
              el.load(); el.play().catch(function() {{}}); }}
  }});
}}
function clearVideo(slot) {{
  camShorts.forEach(function(cs) {{
    var el = document.getElementById('vid_'+slot+'_'+cs);
    if (el) {{ el.src=''; el.style.border = '2px solid #ccc'; }}
  }});
}}
function sampleInfo(idx) {{
  return '<b>'+taskLabels[idx]+'</b>\\nsplit: '+splitLabels[idx]+'\\nsample: '+idx+'\\nframe: '+frameIdxs[idx];
}}
function showInfo(ia, ib) {{
  var h = '<b style="color:#2196F3;">A</b>  '+sampleInfo(ia);
  if (ib !== null) h += '\\n\\n<b style="color:#4CAF50;">B</b>  '+sampleInfo(ib);
  document.getElementById('info-box').innerHTML = h;
}}
function showDist(ia, ib) {{
  var has = (Object.keys(distMat).length > 0);
  if (has) {{
    document.getElementById('d_act_full_cos').textContent = fmt(getDist('action_full_cos', ia, ib));
    document.getElementById('d_act_full_l2' ).textContent = fmt(getDist('action_full_l2',  ia, ib));
    document.getElementById('d_bb_full_cos' ).textContent = fmt(getDist('bb_full_cos',     ia, ib));
    document.getElementById('d_bb_full_l2'  ).textContent = fmt(getDist('bb_full_l2',      ia, ib));
  }}
  if (distMat['action_pca_cos']) {{
    document.getElementById('d_act_pca_cos').textContent = fmt(getDist('action_pca_cos', ia, ib));
    document.getElementById('d_act_pca_l2' ).textContent = fmt(getDist('action_pca_l2',  ia, ib));
    document.getElementById('d_bb_pca_cos' ).textContent = fmt(getDist('bb_pca_cos',     ia, ib));
    document.getElementById('d_bb_pca_l2'  ).textContent = fmt(getDist('bb_pca_l2',      ia, ib));
  }} else {{
    document.getElementById('d_act_pca_cos').textContent = fmt(cosineDistPCA(pcaAction[ia],   pcaAction[ib]));
    document.getElementById('d_act_pca_l2' ).textContent = fmt(l2DistPCA(    pcaAction[ia],   pcaAction[ib]));
    document.getElementById('d_bb_pca_cos' ).textContent = fmt(cosineDistPCA(pcaBackbone[ia], pcaBackbone[ib]));
    document.getElementById('d_bb_pca_l2'  ).textContent = fmt(l2DistPCA(    pcaBackbone[ia], pcaBackbone[ib]));
  }}
  document.getElementById('dist-panel').style.display = 'block';
}}
function hideDist() {{ document.getElementById('dist-panel').style.display = 'none'; }}
function handleClick(sampleIdx) {{
  if (selectedA === null) {{
    selectedA = sampleIdx;
    setHighlight(N_DATA,     sampleIdx, 'plot_action',   actionXY);
    setHighlight(N_DATA,     sampleIdx, 'plot_backbone', backboneXY);
    clearHighlight(N_DATA+1, 'plot_action');
    clearHighlight(N_DATA+1, 'plot_backbone');
    playVideo('A', sampleIdx); clearVideo('B');
    showInfo(sampleIdx, null); hideDist();
  }} else if (selectedA === sampleIdx) {{
    selectedA = null;
    clearHighlight(N_DATA,   'plot_action');   clearHighlight(N_DATA,   'plot_backbone');
    clearHighlight(N_DATA+1, 'plot_action');   clearHighlight(N_DATA+1, 'plot_backbone');
    clearVideo('A'); clearVideo('B');
    document.getElementById('info-box').innerHTML = '<span style="color:#aaa;font-style:italic;">click a point for details</span>';
    hideDist();
  }} else {{
    var idxA = selectedA; selectedA = null;
    setHighlight(N_DATA+1, sampleIdx, 'plot_action',   actionXY);
    setHighlight(N_DATA+1, sampleIdx, 'plot_backbone', backboneXY);
    playVideo('B', sampleIdx);
    showInfo(idxA, sampleIdx);
    showDist(idxA, sampleIdx);
  }}
}}

var actionData = {json.dumps(action_traces)};
var backboneData = {json.dumps(backbone_traces)};
var actionLayout = {json.dumps(action_layout)};
var backboneLayout = {json.dumps(backbone_layout)};
var config = {{responsive: true, displayModeBar: true}};

Plotly.newPlot('plot_action', actionData, actionLayout, config).then(function(ga) {{
  return Plotly.newPlot('plot_backbone', backboneData, backboneLayout, config).then(function(gb) {{
    ga.on('plotly_click', function(ev) {{ handleClick(ev.points[0].customdata[2]); }});
    gb.on('plotly_click', function(ev) {{ handleClick(ev.points[0].customdata[2]); }});
  }});
}});
</script></body></html>"""
    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[html] {out_path}  ({Path(out_path).stat().st_size/1e6:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# Cache I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_cache(path: Path, action_feats, gt_actions, bb_f_list, bb_m_list, meta):
    fb = np.empty(len(bb_f_list), dtype=object)
    mb = np.empty(len(bb_m_list), dtype=object)
    for i, (f, m) in enumerate(zip(bb_f_list, bb_m_list)):
        fb[i] = f.numpy(); mb[i] = m.numpy()
    np.savez_compressed(
        path,
        action_feats=action_feats,
        gt_actions=gt_actions,
        bb_feats=fb,
        bb_masks=mb,
        task_labels=np.array(meta["task_labels"]),
        split_labels=np.array(meta["split_labels"]),
        ds_indices=np.array(meta["ds_indices"], dtype=np.int64),
        traj_ids=np.array(meta["traj_ids"], dtype=np.int64),
        base_idxs=np.array(meta["base_idxs"], dtype=np.int64),
        dataset_paths=np.array(meta["dataset_paths"]),
    )


def load_cache(path: Path):
    npz = np.load(path, allow_pickle=True)
    bb_f = [torch.from_numpy(x.astype(np.float32)) for x in npz["bb_feats"]]
    bb_m = [torch.from_numpy(x) for x in npz["bb_masks"]]
    meta = {
        "task_labels":   [str(x) for x in npz["task_labels"]],
        "split_labels":  [str(x) for x in npz["split_labels"]],
        "ds_indices":    [int(x) for x in npz["ds_indices"]],
        "traj_ids":      [int(x) for x in npz["traj_ids"]],
        "base_idxs":     [int(x) for x in npz["base_idxs"]],
        "dataset_paths": [str(x) for x in npz["dataset_paths"]],
    }
    return npz["action_feats"], npz["gt_actions"], bb_f, bb_m, meta


def rebuild_manifest_from_meta(meta: dict, split_to_datasets: dict) -> list[dict]:
    """Re-attach live dataset handles so video clip extraction can run after a cache hit."""
    path_to_ds = {}
    for split, entries in split_to_datasets.items():
        for ds, task, task_dir in entries:
            path_to_ds[str(task_dir)] = (ds, task_dir)
    manifest = []
    for i in range(len(meta["task_labels"])):
        ds_path = meta["dataset_paths"][i]
        ds_entry = path_to_ds.get(ds_path)
        ds = ds_entry[0] if ds_entry else None
        task_dir = ds_entry[1] if ds_entry else Path(ds_path)
        manifest.append({
            "split": meta["split_labels"][i],
            "task": meta["task_labels"][i],
            "dataset": ds,
            "task_dir": task_dir,
            "ds_index": meta["ds_indices"][i],
            "episode_index": meta["traj_ids"][i],
            "frame_index": meta["base_idxs"][i],
        })
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    split_tag = "_".join(args.splits)
    cache_root = Path(args.cache_dir) / split_tag
    cache_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) if args.output_dir else cache_root / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cache_root] {cache_root}")
    print(f"[output_dir] {out_dir}")

    # ── 1. Load per-split datasets ──
    print(f"\n[1/6] Loading splits: {args.splits}")
    split_to_datasets = load_split_datasets(args.dataset_root, args.splits)
    total_tasks = sum(len(v) for v in split_to_datasets.values())
    print(f"  total datasets (=tasks): {total_tasks}")
    if total_tasks == 0:
        print("[error] No datasets loaded."); sys.exit(1)

    # ── 2. Feature cache ──
    cache_path = cache_root / "features.npz"
    if cache_path.exists() and not args.skip_feature_cache:
        print(f"\n[2/6] Loading cached features from {cache_path}")
        action_feats, gt_actions, bb_f_list, bb_m_list, meta = load_cache(cache_path)
        manifest = rebuild_manifest_from_meta(meta, split_to_datasets)
    else:
        # Build manifest → build stats → load policy → extract
        print(f"\n[2/6] Building sample manifest ...")
        manifest = build_manifest_local(split_to_datasets, args.samples_per_task, args.seed)
        print(f"  manifest size: {len(manifest)}")

        print(f"\n[3/6] Building dataset stats ...")
        if args.stats_source == "local":
            dataset_stats = build_dataset_stats_local(split_to_datasets)
        else:
            dataset_stats = None

        print(f"\n[4/6] Loading policy + preprocessor ...")
        policy, preprocessor = load_policy_and_preprocessor(
            args.checkpoint_dir, dataset_stats, args.device)

        print(f"\n[5/6] Extracting features (batch_size={args.batch_size}) ...")
        action_feats, gt_actions, bb_f_list, bb_m_list, meta = extract_features(
            policy, preprocessor, manifest, args.batch_size, args.device)
        save_cache(cache_path, action_feats, gt_actions, bb_f_list, bb_m_list, meta)
        print(f"  [cache] saved → {cache_path}")
        del policy, preprocessor

    N = action_feats.shape[0]
    print(f"\n  N={N}  action_feats={action_feats.shape}  backbone batches={len(bb_f_list)}")
    print(f"  splits: {dict((s, meta['split_labels'].count(s)) for s in sorted(set(meta['split_labels'])))}")
    print(f"  tasks: {len(set(meta['task_labels']))}")

    # ── 3. Aggregate + t-SNE ──
    print(f"\n[6/6] Aggregating + t-SNE ...")
    X_action = agg_action(action_feats, args.action_agg, gt_actions=gt_actions)
    X_backbone = agg_backbone(bb_f_list, bb_m_list, args.backbone_agg)
    print(f"  X_action={X_action.shape}  X_backbone={X_backbone.shape}")
    zscore = not args.no_zscore
    X_pca_act, X2d_act = run_tsne(X_action, args.perplexity, args.tsne_n_iter, args.seed,
                                  "action_encoder", zscore=zscore)
    X_pca_bb,  X2d_bb  = run_tsne(X_backbone, args.perplexity, args.tsne_n_iter, args.seed,
                                  "backbone", zscore=zscore)
    if X2d_act is None or X2d_bb is None:
        print("[error] t-SNE failed."); sys.exit(1)

    print(f"  computing distance matrices ...")
    dist_b64 = compute_dist_matrices(X_action, X_backbone, X_pca_act, X_pca_bb)

    # ── 4. Video clips ──
    clips_dirs = {}
    if not args.skip_clips:
        print(f"\n  extracting video clips ...")
        w, h = map(int, args.clip_size.split("x"))
        clips_dirs = extract_video_clips_local(
            manifest, args.cam_keys, args.action_horizon,
            out_dir / "clips", fps=20, output_size=(w, h))

    # ── 5. Output ──
    tag = f"tsne_p{args.perplexity}" + ("_noz" if not zscore else "")
    plot_overview_by_split(X2d_act, X2d_bb, meta["split_labels"], args.perplexity,
                           out_dir / f"overview_by_split_{tag}.png")

    # episode progress: try ds.meta.episodes, fallback to 0.0
    ep_progress = []
    path_to_ds = {}
    for split, entries in split_to_datasets.items():
        for ds, task, task_dir in entries:
            path_to_ds[str(task_dir)] = ds
    for i in range(N):
        ds = path_to_ds.get(meta["dataset_paths"][i])
        if ds is None:
            ep_progress.append(0.0); continue
        try:
            ep_idx = meta["traj_ids"][i]
            ep_info = ds.meta.episodes[ep_idx]
            length = int(ep_info.get("length", ep_info.get("num_frames", 1)))
            ep_progress.append(meta["base_idxs"][i] / max(length - 1, 1))
        except Exception:
            ep_progress.append(0.0)

    make_joint_html(
        X2d_act, X2d_bb, X_pca_act, X_pca_bb,
        task_labels=meta["task_labels"],
        split_labels=meta["split_labels"],
        frame_indices=meta["base_idxs"],
        episode_progress=ep_progress,
        action_agg=args.action_agg, backbone_agg=args.backbone_agg,
        perplexity=args.perplexity,
        out_path=out_dir / f"joint_{tag}.html",
        clips_dirs=clips_dirs,
        zscore=zscore,
        dist_b64=dist_b64,
    )
    print(f"\nDone. Output -> {out_dir}")


if __name__ == "__main__":
    main()
