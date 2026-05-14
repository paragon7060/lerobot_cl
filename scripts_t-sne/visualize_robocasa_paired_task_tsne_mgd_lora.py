#!/usr/bin/env python3
"""
t-SNE visualization for RoboCasa atomic task PAIRS using a Groot-MGD LoRA
checkpoint.

Groups visualized:
  - AdjustToasterOvenTemp : Increase vs Decrease
  - AdjustWaterTemp       : hot→cold vs cold→hot
  - OpenDrawer            : left vs right
  - CloseDrawer           : left vs right
  - TurnSinkSpout         : left vs right
  - TurnMicrowave         : start (on) vs stop (off)
  - TurnOnStove           : front-left / front-right / ... (7 burners)
  - TurnOffStove          : front-left / front-right / ... (7 burners)

Points are colored by task description text.
Legend group buttons toggle each task group on/off.

This variant differs from visualize_robocasa_paired_task_tsne.py in two ways:
  - it loads lerobot.policies.groot_mgd.GrootMGDPolicy, not baseline GrootPolicy
  - it extracts processed backbone features from GR00TN15.forward(...,
    return_intermediate=True), matching the VLM latent used by the MGD loss

Example:
  ssh seonho@166.104.35.48 "cd ~/ws3/lerobot && \\
      /home/seonho/miniconda3/envs/lerobot050_groot/bin/python \\
          src/lerobot/scripts/visualize_robocasa_paired_task_tsne_mgd_lora.py \\
          --lerobot_src /home/seonho/clvla/lerobot_cl/src \\
          --checkpoint_dir /home/seonho/clvla/MGD/outputs/.../checkpoints/last/pretrained_model \\
          --dataset_root /home/seonho/slicing_robocasa_human_v3 \\
          --samples_per_desc 16 \\
          --action_agg_list flatten rp_flatten \\
          --rp_dim 512 \\
          --skip_feature_cache \\
          --cache_dir ~/ws3/outputs/tsne_vis_paired_mgd_lora"
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
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Task pair definitions
# ══════════════════════════════════════════════════════════════════════════════

TASK_PAIR_GROUPS: dict[str, list[str]] = {
    "AdjustToasterOvenTemp": [
        "Increase the toaster oven temperature.",
        "Decrease the toaster oven temperature.",
    ],
    "AdjustWaterTemp": [
        "The water running in the sink is cold. Adjust the faucet handle to run the water hot. Make sure to keep the water on.",
        "The water running in the sink is hot. Adjust the faucet handle to run the water cold. Make sure to keep the water on.",
    ],
    "OpenDrawer": [
        "Open the left drawer.",
        "Open the right drawer.",
    ],
    "CloseDrawer": [
        "Close the left drawer.",
        "Close the right drawer.",
    ],
    "TurnSinkSpout": [
        "Turn the sink spout to the left.",
        "Turn the sink spout to the right.",
    ],
    "TurnMicrowave": [
        "Press the start button on the microwave.",
        "Press the stop button on the microwave.",
    ],
    "TurnOnStove": [
        "Turn on the front left burner of the stove.",
        "Turn on the front right burner of the stove.",
        "Turn on the front center burner of the stove.",
        "Turn on the rear left burner of the stove.",
        "Turn on the rear right burner of the stove.",
        "Turn on the rear center burner of the stove.",
        "Turn on the center burner of the stove.",
    ],
    "TurnOffStove": [
        "Turn off the front left burner of the stove.",
        "Turn off the front right burner of the stove.",
        "Turn off the front center burner of the stove.",
        "Turn off the rear left burner of the stove.",
        "Turn off the rear right burner of the stove.",
        "Turn off the rear center burner of the stove.",
        "Turn off the center burner of the stove.",
    ],
}

MODALITY_VIDEO_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]
VIDEO_TO_IMAGE_KEY = {vk: f"observation.images.{vk[len('video.'):]}" for vk in MODALITY_VIDEO_KEYS}
DEFAULT_CAM_KEYS = ["robot0_eye_in_hand", "robot0_agentview_right"]
ATOMIC_SPLIT = "robocasa_pretrain_human_atomic"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot_src", type=str, default="/home/seonho/clvla/lerobot_cl/src",
                   help="LeRobot source tree that contains policies.groot_mgd")
    p.add_argument("--checkpoint_dir", type=str,
                   default="/home/seonho/groot_robocasa/outputs/pretrain/checkpoints/080000/pretrained_model")
    p.add_argument("--dataset_root", type=str,
                   default="/home/seonho/slicing_robocasa_human_v3")
    p.add_argument("--task_groups", type=str, nargs="*", default=None,
                   help="Which groups to include (default: all). e.g. --task_groups OpenDrawer TurnSinkSpout")
    p.add_argument("--samples_per_desc", type=int, default=16,
                   help="Random frame samples per task description")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--cache_dir", type=str, default="./outputs/tsne_vis_paired_mgd_lora")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--action_agg_list", type=str, nargs="+",
                   default=["flatten", "rp_flatten"],
                   choices=["mean", "max", "last", "first_last", "delta_total", "delta_mean",
                            "flatten", "rp_flatten", "first_mid_last", "raw_flatten",
                            "raw_delta_mean", "raw_delta_total"],
                   help="One or more action aggregation methods; separate HTML per method")
    p.add_argument("--rp_dim", type=int, default=512,
                   help="Output dimension for --action_agg_list rp_flatten")
    p.add_argument("--backbone_agg", type=str, default="mask_mean",
                   choices=["mask_mean", "mean", "max", "first", "last"])
    p.add_argument("--perplexity", type=int, default=30)
    p.add_argument("--tsne_n_iter", type=int, default=1000)
    p.add_argument("--cam_keys", type=str, nargs="*", default=DEFAULT_CAM_KEYS)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--clip_size", type=str, default="320x240")
    p.add_argument("--skip_clips", action="store_true")
    p.add_argument("--reuse_clips", action="store_true",
                   help="Reuse existing clips instead of regenerating them. Off by default to keep clips aligned with the sampled manifest.")
    p.add_argument("--skip_feature_cache", action="store_true")
    p.add_argument("--no_proc", action="store_true",
                   help="Skip process_backbone_output (vlln+vl_self_attention) features and toggle button")
    p.add_argument("--no_zscore", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ensure_lerobot_src(lerobot_src: str):
    src = Path(lerobot_src).expanduser()
    if src.exists():
        src_s = str(src)
        if src_s not in sys.path:
            sys.path.insert(0, src_s)
        print(f"[lerobot_src] {src_s}")
    else:
        print(f"[warn] --lerobot_src does not exist: {src}")


# ══════════════════════════════════════════════════════════════════════════════
# Scan task dirs for matching descriptions
# ══════════════════════════════════════════════════════════════════════════════

def scan_paired_task_dirs(dataset_root: str, groups: dict[str, list[str]],
                          action_horizon: int) -> dict[str, list[tuple]]:
    """
    Scan ATOMIC_SPLIT for task_dirs containing at least one target description.

    Returns:
        desc_to_entries: {description: [(ds, task_dir), ...]}
    """
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    all_target_descs: set[str] = set()
    for descs in groups.values():
        all_target_descs.update(descs)

    split_dir = Path(dataset_root) / ATOMIC_SPLIT
    if not split_dir.exists():
        print(f"[error] split dir not found: {split_dir}")
        sys.exit(1)

    desc_to_entries: dict[str, list] = defaultdict(list)

    for task_dir in sorted(split_dir.iterdir()):
        tasks_p = task_dir / "meta" / "tasks.parquet"
        info_p = task_dir / "meta" / "info.json"
        if not tasks_p.exists() or not info_p.exists():
            continue

        tasks_df = pd.read_parquet(tasks_p)
        dir_descs = set(tasks_df.index.tolist())
        matched = dir_descs & all_target_descs
        if not matched:
            continue

        try:
            with open(info_p) as f:
                info = json.load(f)
            repo_id = info.get("repo_id", task_dir.name)
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(task_dir))
            delta_ts = {"action": [i / meta.fps for i in range(action_horizon)]}
            ds = LeRobotDataset(repo_id=repo_id, root=str(task_dir),
                                video_backend="pyav", delta_timestamps=delta_ts)
        except Exception as e:
            print(f"  [skip] {task_dir.name}: {e}")
            continue

        for desc in matched:
            desc_to_entries[desc].append((ds, task_dir))
        print(f"  [scan] {task_dir.name}: {sorted(matched)}")

    return desc_to_entries


# ══════════════════════════════════════════════════════════════════════════════
# Find dataset indices per description inside a task_dir
# ══════════════════════════════════════════════════════════════════════════════

def find_frames_by_description(task_dir: Path,
                                target_descs: list[str]) -> dict[str, list[int]]:
    """
    Returns {description: [global_dataset_indices]} for target_descs found in task_dir.
    Reads the raw data parquet files to map task_index → frame indices.
    """
    import pandas as pd

    tasks_p = task_dir / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_p)
    desc_to_task_idx: dict[str, int] = {}
    for desc in target_descs:
        if desc in tasks_df.index:
            desc_to_task_idx[desc] = int(tasks_df.loc[desc, "task_index"])

    if not desc_to_task_idx:
        return {}

    data_files = sorted((task_dir / "data").glob("**/*.parquet"))
    if not data_files:
        return {}

    frames_df = pd.concat(
        [pd.read_parquet(fp, columns=["task_index", "index"]) for fp in data_files],
        ignore_index=True,
    )

    result: dict[str, list[int]] = {}
    for desc, task_idx in desc_to_task_idx.items():
        idxs = frames_df.loc[frames_df["task_index"] == task_idx, "index"].tolist()
        result[desc] = [int(x) for x in idxs]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Build manifest
# ══════════════════════════════════════════════════════════════════════════════

def build_manifest_paired(desc_to_entries: dict[str, list[tuple]],
                          groups: dict[str, list[str]],
                          samples_per_desc: int,
                          seed: int) -> list[dict]:
    rng = random.Random(seed)

    # desc → group name
    desc_to_group: dict[str, str] = {}
    for gname, descs in groups.items():
        for d in descs:
            desc_to_group[d] = gname

    manifest: list[dict] = []

    for desc, entries in desc_to_entries.items():
        group = desc_to_group.get(desc, "Other")

        # Collect (ds, task_dir, global_idx) across all task_dirs with this desc
        all_candidates: list[tuple] = []
        for ds, task_dir in entries:
            frames_map = find_frames_by_description(task_dir, [desc])
            idxs = frames_map.get(desc, [])
            for idx in idxs:
                all_candidates.append((ds, task_dir, idx))

        if not all_candidates:
            print(f"  [warn] no frames found for: {desc!r}")
            continue

        chosen = rng.sample(all_candidates, min(samples_per_desc, len(all_candidates)))
        for ds, task_dir, ds_idx in chosen:
            raw = ds[ds_idx]
            ep_idx = int(raw["episode_index"].item()) if hasattr(raw["episode_index"], "item") else int(raw["episode_index"])
            fr_idx = int(raw["frame_index"].item()) if hasattr(raw["frame_index"], "item") else int(raw["frame_index"])
            manifest.append({
                "group": group,
                "description": desc,
                "dataset": ds,
                "task_dir": task_dir,
                "ds_index": ds_idx,
                "episode_index": ep_idx,
                "frame_index": fr_idx,
            })

    rng.shuffle(manifest)
    print(f"[manifest] total {len(manifest)} samples across {len(desc_to_entries)} descriptions")
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# Batch assembly helpers (same as robocasa script)
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_one_sample(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if k.startswith("observation.images."):
            if isinstance(v, torch.Tensor):
                t = v.float()
                if t.ndim == 4:
                    t = t[0]
                if t.max() > 1.5:
                    t = t / 255.0
            else:
                arr = np.asarray(v)
                if arr.ndim == 4:
                    arr = arr[0]
                t = torch.from_numpy(arr)
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    t = t.permute(2, 0, 1)
                t = t.float()
                if t.max() > 1.5:
                    t = t / 255.0
            out[k] = t
    state = raw.get("observation.state")
    if state is not None:
        t = torch.as_tensor(state, dtype=torch.float32)
        if t.ndim == 2:
            t = t[0]
        out["observation.state"] = t
    action = raw.get("action")
    if action is not None:
        out["action"] = torch.as_tensor(action, dtype=torch.float32)
    task = raw.get("task", raw.get("annotation.human.task_description", ""))
    if isinstance(task, (list, np.ndarray)):
        task = task[0] if len(task) > 0 else ""
    out["task"] = str(task)
    return out


def _collate(samples: list[dict]) -> dict:
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
# Dataset stats
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset_stats(desc_to_entries: dict[str, list[tuple]]) -> dict:
    import json as _json
    state_means, state_stds, state_ws = [], [], []
    action_means, action_stds, action_ws = [], [], []

    seen_dirs: set[str] = set()
    for entries in desc_to_entries.values():
        for ds, task_dir in entries:
            key = str(task_dir)
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
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
        w = np.array(ws, dtype=np.float64); w /= w.sum()
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
    for ik in VIDEO_TO_IMAGE_KEY.values():
        dataset_stats[ik] = {
            "mean": np.zeros(3, dtype=np.float32),
            "std":  np.ones(3,  dtype=np.float32),
        }
    return dataset_stats


# ══════════════════════════════════════════════════════════════════════════════
# Policy loading
# ══════════════════════════════════════════════════════════════════════════════

def load_policy_and_preprocessor(checkpoint_dir: str, dataset_stats: dict, device: str):
    from lerobot.policies.groot_mgd.modeling_groot import GrootMGDPolicy
    from lerobot.policies.factory import make_pre_post_processors
    ckpt = Path(checkpoint_dir)
    print(f"[load:groot_mgd] {ckpt}")
    policy = GrootMGDPolicy.from_pretrained(str(ckpt))
    policy.eval().to(device)
    print(
        "[policy] "
        f"type={getattr(policy.config, 'type', None)} "
        f"mgd_trainable_mode={getattr(policy.config, 'mgd_trainable_mode', None)} "
        f"lora_rank={getattr(policy.config, 'lora_rank', None)} "
        f"target={getattr(policy.config, 'mgd_target_pooling', None)}+"
        f"{getattr(policy.config, 'mgd_target_projection', None)}"
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset_stats,
    )
    return policy, preprocessor


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _build_groot_inputs(processed: dict) -> dict:
    allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
    return {
        k: v
        for k, v in processed.items()
        if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
    }


def _masked_mean_pool_torch(features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return features.mean(dim=1)
    mask_f = mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    return (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def extract_features(policy, preprocessor, manifest: list[dict],
                     batch_size: int, device: str, with_proc: bool = True):
    use_bf16 = getattr(policy.config, "use_bf16", True)
    dev_type = "cuda" if device != "cpu" else "cpu"
    action_encoder = policy._groot_model.action_head.action_encoder

    ac_list, gt_list = [], []
    bb_f_list, bb_m_list = [], []           # lora_on raw
    bb_proc_f_list, bb_proc_m_list = [], [] # lora_on processed (process_backbone_output)
    vlm_orig_f_list, vlm_orig_m_list = [], []           # lora_off raw
    vlm_orig_proc_f_list, vlm_orig_proc_m_list = [], [] # lora_off processed
    drift_cos_list, drift_l2_list = [], []
    N_total = len(manifest)

    action_head = policy._groot_model.action_head

    for start in range(0, N_total, batch_size):
        chunk = manifest[start:start + batch_size]
        samples = [_flatten_one_sample(m["dataset"][m["ds_index"]]) for m in chunk]
        batch = _collate(samples)
        processed = preprocessor(batch)
        groot_inputs = _build_groot_inputs(processed)

        # lora_on: raw backbone
        backbone_inputs, action_inputs = policy._groot_model.prepare_input(groot_inputs)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
            lora_on_out = policy._groot_model.backbone(backbone_inputs)
        # save raw before process_backbone_output mutates lora_on_out in-place
        lora_on_raw_f = lora_on_out["backbone_features"].detach().float()
        lora_on_raw_m = lora_on_out.get("backbone_attention_mask")
        bb_f_list.append(lora_on_raw_f.cpu())
        bb_m_list.append(lora_on_raw_m.detach().cpu() if lora_on_raw_m is not None else None)
        # lora_on: processed (process_backbone_output mutates lora_on_out in-place)
        if with_proc:
            action_head.set_frozen_modules_to_eval_mode()
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
                lora_on_proc = action_head.process_backbone_output(lora_on_out)
            bb_proc_f_list.append(lora_on_proc["backbone_features"].detach().float().cpu())
            bb_proc_m_list.append(lora_on_proc["backbone_attention_mask"].detach().cpu())

        # lora_off: raw backbone
        adapter_ctx = (policy._disable_lora_adapters()
                       if hasattr(policy, "_disable_lora_adapters")
                       else __import__("contextlib").nullcontext())
        with adapter_ctx:
            backbone_inputs_off, _ = policy._groot_model.prepare_input(groot_inputs)
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
                orig_out = policy._groot_model.backbone(backbone_inputs_off)
        # save raw before process_backbone_output mutates orig_out in-place
        orig_raw_f = orig_out["backbone_features"].detach().float()
        orig_raw_m = orig_out.get("backbone_attention_mask")
        vlm_orig_f_list.append(orig_raw_f.cpu())
        vlm_orig_m_list.append(orig_raw_m.detach().cpu() if orig_raw_m is not None else None)
        # lora_off: processed (process_backbone_output mutates orig_out in-place)
        if with_proc:
            action_head.set_frozen_modules_to_eval_mode()
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
                orig_proc = action_head.process_backbone_output(orig_out)
            vlm_orig_proc_f_list.append(orig_proc["backbone_features"].detach().float().cpu())
            vlm_orig_proc_m_list.append(orig_proc["backbone_attention_mask"].detach().cpu())

        # Drift: raw lora_on vs raw lora_off (same feature level)
        z_v_on = _masked_mean_pool_torch(lora_on_raw_f, lora_on_raw_m)
        z_v_off = _masked_mean_pool_torch(orig_raw_f, orig_raw_m)
        drift_cos = F.cosine_similarity(z_v_on, z_v_off, dim=-1)
        drift_l2 = torch.norm(z_v_on - z_v_off, dim=-1)
        drift_cos_list.append(drift_cos.cpu())
        drift_l2_list.append(drift_l2.cpu())

        gt_actions = action_inputs.action
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
    gt_actions_np = np.concatenate([x.numpy() for x in gt_list], axis=0)
    meta = {
        "groups":       [m["group"]        for m in manifest],
        "descriptions": [m["description"]  for m in manifest],
        "ds_indices":   [m["ds_index"]     for m in manifest],
        "traj_ids":     [m["episode_index"] for m in manifest],
        "base_idxs":    [m["frame_index"]  for m in manifest],
        "dataset_paths":[str(m["task_dir"]) for m in manifest],
        "vlm_drift_cos": np.concatenate([x.numpy() for x in drift_cos_list], axis=0).astype(np.float32),
        "vlm_drift_l2":  np.concatenate([x.numpy() for x in drift_l2_list], axis=0).astype(np.float32),
    }
    return (action_feats, gt_actions_np,
            bb_f_list, bb_m_list, bb_proc_f_list, bb_proc_m_list,
            vlm_orig_f_list, vlm_orig_m_list, vlm_orig_proc_f_list, vlm_orig_proc_m_list,
            meta)


# ══════════════════════════════════════════════════════════════════════════════
# Cache I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_cache(path: Path, action_feats, gt_actions,
               bb_f_list, bb_m_list, bb_proc_f_list, bb_proc_m_list,
               vlm_orig_f_list, vlm_orig_m_list, vlm_orig_proc_f_list, vlm_orig_proc_m_list,
               meta):
    def _pack(lst):
        arr = np.empty(len(lst), dtype=object)
        for i, x in enumerate(lst):
            arr[i] = x.numpy() if isinstance(x, torch.Tensor) else x
        return arr

    np.savez_compressed(
        path,
        action_feats=action_feats, gt_actions=gt_actions,
        bb_feats=_pack(bb_f_list), bb_masks=_pack(bb_m_list),
        bb_proc_feats=_pack(bb_proc_f_list), bb_proc_masks=_pack(bb_proc_m_list),
        vlm_orig_feats=_pack(vlm_orig_f_list), vlm_orig_masks=_pack(vlm_orig_m_list),
        vlm_orig_proc_feats=_pack(vlm_orig_proc_f_list), vlm_orig_proc_masks=_pack(vlm_orig_proc_m_list),
        groups=np.array(meta["groups"]),
        descriptions=np.array(meta["descriptions"]),
        ds_indices=np.array(meta["ds_indices"], dtype=np.int64),
        traj_ids=np.array(meta["traj_ids"], dtype=np.int64),
        base_idxs=np.array(meta["base_idxs"], dtype=np.int64),
        dataset_paths=np.array(meta["dataset_paths"]),
        vlm_drift_cos=np.asarray(
            meta.get("vlm_drift_cos", np.full(len(meta["groups"]), np.nan, dtype=np.float32)),
            dtype=np.float32,
        ),
        vlm_drift_l2=np.asarray(
            meta.get("vlm_drift_l2", np.full(len(meta["groups"]), np.nan, dtype=np.float32)),
            dtype=np.float32,
        ),
    )


def load_cache(path: Path):
    npz = np.load(path, allow_pickle=True)
    bb_f = [torch.from_numpy(x.astype(np.float32)) for x in npz["bb_feats"]]
    bb_m = [torch.from_numpy(x) for x in npz["bb_masks"]]
    def _unpack(key):
        if key in npz:
            return [torch.from_numpy(x.astype(np.float32)) for x in npz[key]]
        return []
    def _unpack_mask(key):
        if key in npz:
            return [torch.from_numpy(x) for x in npz[key]]
        return []

    vlm_orig_f      = _unpack("vlm_orig_feats");      vlm_orig_m      = _unpack_mask("vlm_orig_masks")
    bb_proc_f       = _unpack("bb_proc_feats");        bb_proc_m       = _unpack_mask("bb_proc_masks")
    vlm_orig_proc_f = _unpack("vlm_orig_proc_feats");  vlm_orig_proc_m = _unpack_mask("vlm_orig_proc_masks")
    meta = {
        "groups":       [str(x) for x in npz["groups"]],
        "descriptions": [str(x) for x in npz["descriptions"]],
        "ds_indices":   [int(x) for x in npz["ds_indices"]],
        "traj_ids":     [int(x) for x in npz["traj_ids"]],
        "base_idxs":    [int(x) for x in npz["base_idxs"]],
        "dataset_paths":[str(x) for x in npz["dataset_paths"]],
    }
    n = len(meta["groups"])
    meta["vlm_drift_cos"] = (
        npz["vlm_drift_cos"].astype(np.float32)
        if "vlm_drift_cos" in npz
        else np.full(n, np.nan, dtype=np.float32)
    )
    meta["vlm_drift_l2"] = (
        npz["vlm_drift_l2"].astype(np.float32)
        if "vlm_drift_l2" in npz
        else np.full(n, np.nan, dtype=np.float32)
    )
    return (npz["action_feats"], npz["gt_actions"],
            bb_f, bb_m, bb_proc_f, bb_proc_m,
            vlm_orig_f, vlm_orig_m, vlm_orig_proc_f, vlm_orig_proc_m,
            meta)


def rebuild_manifest_from_meta(meta: dict, desc_to_entries: dict) -> list[dict]:
    path_to_ds: dict[str, tuple] = {}
    for entries in desc_to_entries.values():
        for ds, task_dir in entries:
            path_to_ds[str(task_dir)] = (ds, task_dir)
    manifest = []
    for i in range(len(meta["groups"])):
        ds_path = meta["dataset_paths"][i]
        entry = path_to_ds.get(ds_path)
        ds = entry[0] if entry else None
        task_dir = entry[1] if entry else Path(ds_path)
        manifest.append({
            "group": meta["groups"][i],
            "description": meta["descriptions"][i],
            "dataset": ds,
            "task_dir": task_dir,
            "ds_index": meta["ds_indices"][i],
            "episode_index": meta["traj_ids"][i],
            "frame_index": meta["base_idxs"][i],
        })
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation + t-SNE
# ══════════════════════════════════════════════════════════════════════════════

def agg_action(feats: np.ndarray, method: str, gt_actions: np.ndarray | None = None) -> np.ndarray:
    if method.startswith("raw_"):
        if gt_actions is None:
            raise ValueError(f"method='{method}' requires gt_actions")
        t = torch.from_numpy(gt_actions)
        N, T, D = t.shape
        if method == "raw_flatten":       out = t.reshape(N, T * D)
        elif method == "raw_delta_mean":  out = (t[:, 1:, :] - t[:, :-1, :]).mean(dim=1)
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


def random_project_flatten(feats: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    from sklearn.random_projection import GaussianRandomProjection

    t = torch.from_numpy(feats)
    N, T, D = t.shape
    X = t.reshape(N, T * D).numpy()
    n_components = min(n_components, X.shape[1])
    projector = GaussianRandomProjection(
        n_components=n_components,
        random_state=seed,
    )
    return projector.fit_transform(X).astype(np.float32)


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
# Pairwise distance
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
        print(f"~{(len(out[f'{name}_cos'])+len(out[f'{name}_l2']))/1e6:.1f} MB")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Video clip extraction
# ══════════════════════════════════════════════════════════════════════════════

def _get_video_path(task_dir: Path, obs_key: str, episode_index: int) -> Path | None:
    chunk = episode_index // 1000
    # Actual format: videos/{obs_key}/chunk-{chunk:03d}/file-*.mp4
    chunk_dir = task_dir / "videos" / obs_key / f"chunk-{chunk:03d}"
    if chunk_dir.exists():
        files = sorted(chunk_dir.glob("file-*.mp4"))
        if files:
            return files[0]
    # Legacy fallbacks
    candidates = [
        task_dir / "videos" / f"chunk-{chunk:03d}" / obs_key / f"episode_{episode_index:06d}.mp4",
        task_dir / "videos" / obs_key / f"episode_{episode_index:06d}.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(task_dir.glob(f"videos/**/*{episode_index:06d}.mp4"))
    return matches[0] if matches else None


def _get_chunk_start_index(task_dir: Path, episode_index: int,
                           _cache: dict = {}) -> int:
    """Return the global ds_index of the first frame in the chunk containing episode_index."""
    import pandas as pd
    chunk = episode_index // 1000
    key = (str(task_dir), chunk)
    if key in _cache:
        return _cache[key]
    data_dir = task_dir / "data" / f"chunk-{chunk:03d}"
    start = 0
    if data_dir.exists():
        files = sorted(data_dir.glob("*.parquet"))
        if files:
            df = pd.read_parquet(files[0], columns=["index"])
            start = int(df["index"].min())
    _cache[key] = start
    return start


def extract_video_clips(manifest: list[dict], cam_short_keys: list[str],
                        horizon: int, out_dir: Path,
                        fps: int = 20, output_size=(320, 240),
                        reuse_existing: bool = False) -> dict:
    import av
    import io as _io
    from PIL import Image as PILImage

    W, H = output_size
    cam_to_dir = {}
    for cs in cam_short_keys:
        d = out_dir / cs
        d.mkdir(parents=True, exist_ok=True)
        cam_to_dir[cs] = d

    for si, m in enumerate(manifest):
        for cs in cam_short_keys:
            out_file = cam_to_dir[cs] / f"{si:04d}.mp4"
            if reuse_existing and out_file.exists():
                continue
            obs_key = f"observation.images.{cs}"
            vpath = _get_video_path(m["task_dir"], obs_key, m["episode_index"])
            if vpath is None:
                continue
            chunk_start = _get_chunk_start_index(m["task_dir"], m["episode_index"])
            start_ts = (m["ds_index"] - chunk_start) / fps
            try:
                with av.open(str(vpath)) as vc:
                    stream = vc.streams.video[0]
                    tb = float(stream.time_base)
                    vc.seek(int(start_ts / tb), stream=stream, backward=True)
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
                print(f"  [clips] {cs}/{si}: {e}")
                continue
            if not raw_frames:
                continue
            frames_r = []
            for rf in raw_frames[:horizon]:
                pil = PILImage.fromarray(rf)
                if pil.size != (W, H):
                    pil = pil.resize((W, H), PILImage.LANCZOS)
                frames_r.append(np.array(pil))
            buf = _io.BytesIO()
            with av.open(buf, mode="w", format="mp4") as oc:
                ost = oc.add_stream("libx264", rate=fps)
                ost.width = W; ost.height = H; ost.pix_fmt = "yuv420p"
                ost.options = {"crf": "28", "preset": "ultrafast"}
                for rf in frames_r:
                    avf = av.VideoFrame.from_ndarray(rf, format="rgb24")
                    for pkt in ost.encode(avf):
                        oc.mux(pkt)
                for pkt in ost.encode(None):
                    oc.mux(pkt)
            with open(out_file, "wb") as fh:
                fh.write(buf.getvalue())
        if (si + 1) % 50 == 0:
            print(f"  [clips] {si+1}/{len(manifest)}", end="\r")
    print()
    return cam_to_dir


# ══════════════════════════════════════════════════════════════════════════════
# Static matplotlib overview
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview_by_group(X2d_a, X2d_b, groups, descriptions, perplexity, out_path,
                           right_label: str = "backbone"):
    all_groups = sorted(set(groups))
    n = max(len(all_groups), 1)
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(n)
    gcolor = {g: cmap(i) for i, g in enumerate(all_groups)}
    arr_g = np.array(groups)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, X2d, title in [(axes[0], X2d_a, "action_encoder"), (axes[1], X2d_b, right_label)]:
        for g in all_groups:
            m = arr_g == g
            ax.scatter(X2d[m, 0], X2d[m, 1], c=[gcolor[g]], label=g, s=12, alpha=0.7, linewidths=0)
        ax.legend(fontsize=9, markerscale=2)
        ax.set_title(title); ax.grid(True, alpha=0.2)
    fig.suptitle(f"by group  |  t-SNE perp={perplexity}  |  N={len(groups)}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Description color map
# ══════════════════════════════════════════════════════════════════════════════

def _desc_color_map(desc_ids: list[str]) -> dict:
    import colorsys
    golden_ratio = 0.618033988749895
    sat_cycle   = [0.85, 0.65, 0.75]
    light_cycle = [0.42, 0.52, 0.34]
    colors = {}
    for i, did in enumerate(sorted(desc_ids)):
        h = (i * golden_ratio) % 1.0
        s = sat_cycle[i % len(sat_cycle)]
        l = light_cycle[i % len(light_cycle)]
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        colors[did] = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
    return colors


# ══════════════════════════════════════════════════════════════════════════════
# Interactive HTML
# ══════════════════════════════════════════════════════════════════════════════

def make_joint_html(X2d_action, X2d_backbone,
                    X_pca_action, X_pca_backbone,
                    groups, descriptions, frame_indices,
                    action_agg, backbone_agg, perplexity,
                    out_path: Path,
                    X2d_backbone_alt=None, backbone_agg_alt: str = "processed",
                    clips_dirs: dict | None = None,
                    zscore: bool = True,
                    dist_b64: dict | None = None,
                    vlm_drift_cos: np.ndarray | None = None,
                    vlm_drift_l2: np.ndarray | None = None,
                    backbone_label: str = "backbone"):
    N = len(descriptions)
    all_groups = sorted(set(groups))
    all_descs = sorted(set(descriptions))
    desc_color = _desc_color_map(all_descs)

    # Traces: one per (group, description)
    pair_to_idx: dict[tuple, list[int]] = defaultdict(list)
    for i, (g, d) in enumerate(zip(groups, descriptions)):
        pair_to_idx[(g, d)].append(i)
    pairs = sorted(pair_to_idx.keys())

    group_of_trace = [g for (g, _) in pairs]

    def _make_traces(X2d):
        traces = []
        for (g, d) in pairs:
            idxs = pair_to_idx[(g, d)]
            short_d = d[:60] + ("…" if len(d) > 60 else "")
            traces.append({
                "type": "scatter",
                "x": X2d[idxs, 0].tolist(),
                "y": X2d[idxs, 1].tolist(),
                "mode": "markers",
                "name": f"[{g}] {short_d}",
                "legendgroup": g,
                "legendgrouptitle": {"text": g},
                "marker": {"color": desc_color[d], "size": 7, "opacity": 0.85},
                "customdata": [[d, g, i, frame_indices[i]] for i in idxs],
                "hovertemplate": (
                    "<b>%{customdata[1]}</b><br>"
                    "%{customdata[0]}<br>"
                    "sample: %{customdata[2]}<br>"
                    "frame: %{customdata[3]}<extra></extra>"
                ),
            })
        traces.append({
            "type": "scatter", "x": [], "y": [], "mode": "markers",
            "name": "A", "showlegend": False, "hoverinfo": "skip",
            "marker": {"symbol": "circle-open", "size": 22, "color": "#2196F3",
                       "line": {"color": "#2196F3", "width": 3}},
        })
        traces.append({
            "type": "scatter", "x": [], "y": [], "mode": "markers",
            "name": "B", "showlegend": False, "hoverinfo": "skip",
            "marker": {"symbol": "circle-open", "size": 22, "color": "#4CAF50",
                       "line": {"color": "#4CAF50", "width": 3}},
        })
        return traces

    n_data_traces = len(pairs)
    action_traces  = _make_traces(X2d_action)
    backbone_traces = _make_traces(X2d_backbone)

    # Group toggle buttons (custom HTML, not Plotly updatemenus)
    group_btns_html = ""
    for g in all_groups:
        group_btns_html += f"<button class='ctrl-btn' onclick='toggleGroup({json.dumps(g)})'>{g}</button>"

    shared_layout = {
        "height": 650, "hovermode": "closest",
        "xaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
        "yaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
        "legend": {"x": 1.01, "y": 1, "xanchor": "left", "font": {"size": 9},
                   "itemclick": "toggle", "itemdoubleclick": False,
                   "groupclick": "toggleitem", "tracegroupgap": 4},
        "margin": {"l": 50, "r": 300, "t": 80, "b": 40},
    }
    action_layout   = dict(shared_layout, title=f"action_encoder ({action_agg})")
    backbone_layout = dict(shared_layout, title=f"{backbone_label} ({backbone_agg})")

    has_clips = bool(clips_dirs)
    cam_shorts = list(clips_dirs.keys()) if has_clips else []
    video_panel_html = ""
    if has_clips:
        for cs in cam_shorts:
            video_panel_html += f"""
<div class="cam-group">
  <p class="cam-label">{cs}</p>
  <div class="cam-row">
    <div class="vid-box"><p class="vid-ab-label" style="color:#2196F3;">A</p>
      <video id="vid_A_{cs}" autoplay loop muted playsinline width="100%"></video></div>
    <div class="vid-box"><p class="vid-ab-label" style="color:#4CAF50;">B</p>
      <video id="vid_B_{cs}" autoplay loop muted playsinline width="100%"></video></div>
  </div>
</div>"""

    action_xy        = X2d_action.tolist()
    backbone_xy      = X2d_backbone.tolist()
    backbone_xy_alt  = X2d_backbone_alt.tolist() if X2d_backbone_alt is not None else "null"
    has_bb_alt       = X2d_backbone_alt is not None
    pca_act_js  = X_pca_action.tolist()
    pca_bb_js   = X_pca_backbone.tolist()
    has_dist    = bool(dist_b64)
    dist_b64_json = json.dumps(dist_b64) if has_dist else "null"
    if vlm_drift_cos is None:
        vlm_drift_cos = np.full(N, np.nan, dtype=np.float32)
    if vlm_drift_l2 is None:
        vlm_drift_l2 = np.full(N, np.nan, dtype=np.float32)
    drift_cos_arr = np.asarray(vlm_drift_cos, dtype=np.float32)
    drift_l2_arr = np.asarray(vlm_drift_l2, dtype=np.float32)
    drift_cos_mean = float(np.nanmean(drift_cos_arr)) if np.isfinite(drift_cos_arr).any() else float("nan")
    drift_l2_mean = float(np.nanmean(drift_l2_arr)) if np.isfinite(drift_l2_arr).any() else float("nan")
    drift_text = (
        f"  |  drift cos={drift_cos_mean:.4f} l2={drift_l2_mean:.4f}"
        if np.isfinite(drift_cos_mean) and np.isfinite(drift_l2_mean)
        else ""
    )

    title = (f"Paired Task t-SNE  |  action_encoder ({action_agg}) vs {backbone_label} ({backbone_agg})"
             f"  |  perp={perplexity}  |  N={N}  |  descs={len(all_descs)}"
             + drift_text
             + ("  |  no_zscore" if not zscore else ""))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
 body {{ font-family: sans-serif; margin: 0; padding: 6px; background: #f5f5f5; }}
 h3 {{ text-align: center; margin: 4px 0 4px; font-size: 13px; color: #333; }}
 #btn-row {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 5px 8px;
             background: #fff; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 5px; }}
 .ctrl-btn {{ font-size: 11px; padding: 3px 8px; border: 1px solid #bbb; border-radius: 3px;
              background: #f0f0f0; cursor: pointer; white-space: nowrap; }}
 .ctrl-btn:hover {{ background: #dde; border-color: #99a; }}
 .ctrl-btn.sep {{ margin-left: 8px; }}
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
 .dist-value.pca  {{ color: #888; }}
 #mode-hint {{ text-align: center; font-size: 11px; color: #888; margin: 3px 0; }}
</style></head><body>
<h3>{title}</h3>
<div id="btn-row">
  <button class="ctrl-btn" onclick="setGroupVisible('all',true)">All ON</button>
  <button class="ctrl-btn" onclick="setGroupVisible('all','legendonly')">All OFF</button>
  <span style="color:#bbb;padding:0 4px;">|</span>
  {group_btns_html}
  {'<span style="color:#bbb;padding:0 4px;">|</span><button class="ctrl-btn sep" id="bb-toggle-btn" onclick="toggleBackbone()" style="background:#e8f0fe;border-color:#7baaf7;font-weight:bold;">backbone: ' + backbone_agg + ' ⇄ ' + backbone_agg_alt + '</button>' if has_bb_alt else ''}
</div>
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
var backboneXY_alt = {json.dumps(backbone_xy_alt) if has_bb_alt else 'null'};
var bbAggMain = {json.dumps(backbone_agg)};
var bbAggAlt  = {json.dumps(backbone_agg_alt)};
var bbShowingMain = true;
var pcaAction    = {json.dumps(pca_act_js)};
var pcaBackbone  = {json.dumps(pca_bb_js)};
var descLabels   = {json.dumps(descriptions)};
var groupLabels  = {json.dumps(groups)};
var frameIdxs    = {json.dumps(frame_indices)};
var vlmDriftCos  = {json.dumps(drift_cos_arr.tolist())};
var vlmDriftL2   = {json.dumps(drift_l2_arr.tolist())};
var camShorts    = {json.dumps(cam_shorts)};
var groupOfTrace = {json.dumps(group_of_trace)};
var traceVisible = new Array(N_DATA).fill(true);

function hslToRgb(h, s, l) {{
  var c=(1-Math.abs(2*l-1))*s, x=c*(1-Math.abs((h*6)%2-1)), m=l-c/2, r=0,g=0,b=0;
  if(h<1/6){{r=c;g=x;}}else if(h<2/6){{r=x;g=c;}}else if(h<3/6){{g=c;b=x;}}
  else if(h<4/6){{g=x;b=c;}}else if(h<5/6){{r=x;b=c;}}else{{r=c;b=x;}}
  return [Math.round((r+m)*255),Math.round((g+m)*255),Math.round((b+m)*255)];
}}
function toggleBackbone() {{
  if (!backboneXY_alt) return;
  bbShowingMain = !bbShowingMain;
  var xy = bbShowingMain ? backboneXY : backboneXY_alt;
  var newX = [], newY = [], idxs = [];
  for (var ti = 0; ti < N_DATA; ti++) {{
    var cd = backboneData[ti].customdata;
    if (!cd) continue;
    newX.push(cd.map(function(c){{ return xy[c[2]][0]; }}));
    newY.push(cd.map(function(c){{ return xy[c[2]][1]; }}));
    idxs.push(ti);
  }}
  Plotly.restyle('plot_backbone', {{x: newX, y: newY}}, idxs);
  var curAgg  = bbShowingMain ? bbAggMain : bbAggAlt;
  var nextAgg = bbShowingMain ? bbAggAlt  : bbAggMain;
  var btn = document.getElementById('bb-toggle-btn');
  if (btn) btn.textContent = 'backbone: ' + curAgg + ' ⇄ ' + nextAgg;
  Plotly.relayout('plot_backbone', {{'title.text': 'backbone (' + curAgg + ')'}});
}}
function recolorVisible() {{
  var vis=[];
  for(var i=0;i<N_DATA;i++) if(traceVisible[i]===true) vis.push(i);
  var n=vis.length, phi=0.618033988749895, sc=[0.85,0.65,0.75], lc=[0.42,0.52,0.34];
  var colors=[];
  for(var j=0;j<n;j++){{
    var rgb=hslToRgb((j*phi)%1.0, sc[j%sc.length], lc[j%lc.length]);
    colors.push('rgba('+rgb[0]+','+rgb[1]+','+rgb[2]+',0.85)');
  }}
  if(n) {{
    Plotly.restyle('plot_action',   {{'marker.color': colors}}, vis);
    Plotly.restyle('plot_backbone', {{'marker.color': colors}}, vis);
  }}
}}

function setGroupVisible(group, vis) {{
  var idxs = [], visArr = [];
  for (var i = 0; i < N_DATA; i++) {{
    if (group === 'all' || groupOfTrace[i] === group) {{
      idxs.push(i); visArr.push(vis); traceVisible[i] = vis;
    }}
  }}
  if (!idxs.length) return;
  Plotly.restyle('plot_action',   {{visible: visArr}}, idxs);
  Plotly.restyle('plot_backbone', {{visible: visArr}}, idxs);
  recolorVisible();
}}
function toggleGroup(group) {{
  var allOn = true;
  for (var i = 0; i < N_DATA; i++) {{
    if (group === 'all' || groupOfTrace[i] === group) {{
      if (traceVisible[i] !== true) {{ allOn = false; break; }}
    }}
  }}
  setGroupVisible(group, allOn ? 'legendonly' : true);
}}

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
function triIdx(i, j) {{ if (i > j) {{ var t=i; i=j; j=t; }} return i*(N-1) - Math.floor(i*(i-1)/2) + (j-i-1); }}
function getDist(key, i, j) {{ if (i===j) return 0.0; return distMat[key][triIdx(i,j)]; }}
function cosineDistPCA(a,b) {{ var d=0,na=0,nb=0; for(var i=0;i<a.length;i++){{d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}} return 1.0-Math.max(-1,Math.min(1,d/(Math.sqrt(na)*Math.sqrt(nb)+1e-10))); }}
function l2DistPCA(a,b) {{ var s=0; for(var i=0;i<a.length;i++){{var d=a[i]-b[i];s+=d*d;}} return Math.sqrt(s); }}
function pad4(n) {{ return String(n).padStart(4,'0'); }}
function fmt(v) {{ return v.toFixed(4); }}
function fmtMaybe(v) {{ return Number.isFinite(v) ? v.toFixed(4) : 'n/a'; }}

var selectedA = null;
function setHighlight(ti, si, plotId, xyArr) {{ var xy=xyArr[si]; Plotly.restyle(plotId,{{x:[[xy[0]]],y:[[xy[1]]]}}, [ti]); }}
function clearHighlight(ti, plotId) {{ Plotly.restyle(plotId,{{x:[[]],y:[[]]}}, [ti]); }}
function playVideo(slot, si) {{
  if (!camShorts.length) return;
  document.getElementById('vid-hint').style.display='none';
  camShorts.forEach(function(cs) {{
    var el=document.getElementById('vid_'+slot+'_'+cs);
    if(el){{
      el.pause();
      el.src='./clips/'+cs+'/'+pad4(si)+'.mp4';
      el.style.border='2px solid '+(slot==='A'?'#2196F3':'#4CAF50');
      el.oncanplay=function(){{ el.oncanplay=null; el.play().catch(function(){{}}); }};
      el.load();
    }}
  }});
}}
function clearVideo(slot) {{ camShorts.forEach(function(cs) {{ var el=document.getElementById('vid_'+slot+'_'+cs); if(el){{el.src='';el.style.border='2px solid #ccc';}} }}); }}
function sampleInfo(idx) {{
  return '<b>'+groupLabels[idx]+'</b>\\n'+descLabels[idx]+
         '\\nsample: '+idx+'\\nframe: '+frameIdxs[idx]+
         '\\nvlm_drift_cos: '+fmtMaybe(vlmDriftCos[idx])+
         '\\nvlm_drift_l2: '+fmtMaybe(vlmDriftL2[idx]);
}}
function showInfo(ia, ib) {{
  var h='<b style="color:#2196F3;">A</b>  '+sampleInfo(ia);
  if(ib!==null) h+='\\n\\n<b style="color:#4CAF50;">B</b>  '+sampleInfo(ib);
  document.getElementById('info-box').innerHTML=h;
}}
function showDist(ia, ib) {{
  var has=(Object.keys(distMat).length>0);
  if(has) {{
    document.getElementById('d_act_full_cos').textContent=fmt(getDist('action_full_cos',ia,ib));
    document.getElementById('d_act_full_l2' ).textContent=fmt(getDist('action_full_l2', ia,ib));
    document.getElementById('d_bb_full_cos' ).textContent=fmt(getDist('bb_full_cos',    ia,ib));
    document.getElementById('d_bb_full_l2'  ).textContent=fmt(getDist('bb_full_l2',     ia,ib));
  }}
  if(distMat['action_pca_cos']) {{
    document.getElementById('d_act_pca_cos').textContent=fmt(getDist('action_pca_cos',ia,ib));
    document.getElementById('d_act_pca_l2' ).textContent=fmt(getDist('action_pca_l2', ia,ib));
    document.getElementById('d_bb_pca_cos' ).textContent=fmt(getDist('bb_pca_cos',    ia,ib));
    document.getElementById('d_bb_pca_l2'  ).textContent=fmt(getDist('bb_pca_l2',     ia,ib));
  }} else {{
    document.getElementById('d_act_pca_cos').textContent=fmt(cosineDistPCA(pcaAction[ia],  pcaAction[ib]));
    document.getElementById('d_act_pca_l2' ).textContent=fmt(l2DistPCA(    pcaAction[ia],  pcaAction[ib]));
    document.getElementById('d_bb_pca_cos' ).textContent=fmt(cosineDistPCA(pcaBackbone[ia],pcaBackbone[ib]));
    document.getElementById('d_bb_pca_l2'  ).textContent=fmt(l2DistPCA(    pcaBackbone[ia],pcaBackbone[ib]));
  }}
  document.getElementById('dist-panel').style.display='block';
}}
function hideDist() {{ document.getElementById('dist-panel').style.display='none'; }}
function handleClick(si) {{
  var bbXY = (bbShowingMain || !backboneXY_alt) ? backboneXY : backboneXY_alt;
  if(selectedA===null) {{
    selectedA=si;
    setHighlight(N_DATA,   si,'plot_action',  actionXY);
    setHighlight(N_DATA,   si,'plot_backbone',bbXY);
    clearHighlight(N_DATA+1,'plot_action');  clearHighlight(N_DATA+1,'plot_backbone');
    playVideo('A',si); clearVideo('B');
    showInfo(si,null); hideDist();
  }} else if(selectedA===si) {{
    selectedA=null;
    clearHighlight(N_DATA,  'plot_action');  clearHighlight(N_DATA,  'plot_backbone');
    clearHighlight(N_DATA+1,'plot_action');  clearHighlight(N_DATA+1,'plot_backbone');
    clearVideo('A'); clearVideo('B');
    document.getElementById('info-box').innerHTML='<span style="color:#aaa;font-style:italic;">click a point for details</span>';
    hideDist();
  }} else {{
    var idxA=selectedA; selectedA=null;
    setHighlight(N_DATA+1,si,'plot_action',  actionXY);
    setHighlight(N_DATA+1,si,'plot_backbone',bbXY);
    playVideo('B',si);
    showInfo(idxA,si);
    showDist(idxA,si);
  }}
}}

var actionData   = {json.dumps(action_traces)};
var backboneData = {json.dumps(backbone_traces)};
var actionLayout   = {json.dumps(action_layout)};
var backboneLayout = {json.dumps(backbone_layout)};
var config = {{responsive: true, displayModeBar: true}};

function syncLegendClick(ev) {{
  var ti=ev.curveNumber; if(ti>=N_DATA) return false;
  var nv=(traceVisible[ti]===true)?'legendonly':true; traceVisible[ti]=nv;
  Plotly.restyle('plot_action',  {{visible:[nv]}},[ti]);
  Plotly.restyle('plot_backbone',{{visible:[nv]}},[ti]);
  recolorVisible();
  return false;
}}
Plotly.newPlot('plot_action', actionData, actionLayout, config).then(function(ga) {{
  return Plotly.newPlot('plot_backbone', backboneData, backboneLayout, config).then(function(gb) {{
    ga.on('plotly_click', function(ev) {{ handleClick(ev.points[0].customdata[2]); }});
    gb.on('plotly_click', function(ev) {{ handleClick(ev.points[0].customdata[2]); }});
    ga.on('plotly_legendclick', syncLegendClick);
    gb.on('plotly_legendclick', syncLegendClick);
    recolorVisible();
  }});
}});
</script></body></html>"""

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[html] {out_path}  ({Path(out_path).stat().st_size/1e6:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    ensure_lerobot_src(args.lerobot_src)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    groups = {k: v for k, v in TASK_PAIR_GROUPS.items()
              if args.task_groups is None or k in args.task_groups}
    if not groups:
        print(f"[error] No groups selected. Available: {list(TASK_PAIR_GROUPS.keys())}")
        sys.exit(1)
    print(f"[groups] {list(groups.keys())}")

    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir) if args.output_dir else cache_root / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cache_root] {cache_root}")
    print(f"[output_dir] {out_dir}")

    # ── 1. Scan task dirs ──
    print(f"\n[1/6] Scanning {ATOMIC_SPLIT} for paired tasks ...")
    desc_to_entries = scan_paired_task_dirs(args.dataset_root, groups, args.action_horizon)
    print(f"  found {len(desc_to_entries)} descriptions")
    if not desc_to_entries:
        print("[error] No matching task descriptions found."); sys.exit(1)

    # ── 2. Feature cache ──
    cache_path = cache_root / "features.npz"
    if cache_path.exists() and not args.skip_feature_cache:
        print(f"\n[2/6] Loading cached features from {cache_path}")
        (action_feats, gt_actions,
         bb_f_list, bb_m_list, bb_proc_f_list, bb_proc_m_list,
         vlm_orig_f_list, vlm_orig_m_list, vlm_orig_proc_f_list, vlm_orig_proc_m_list,
         meta) = load_cache(cache_path)
        manifest = rebuild_manifest_from_meta(meta, desc_to_entries)
    else:
        print(f"\n[2/6] Building manifest ...")
        manifest = build_manifest_paired(desc_to_entries, groups, args.samples_per_desc, args.seed)
        print(f"  manifest: {len(manifest)} samples")

        print(f"\n[3/6] Building dataset stats ...")
        dataset_stats = build_dataset_stats(desc_to_entries)

        print(f"\n[4/6] Loading policy + preprocessor ...")
        policy, preprocessor = load_policy_and_preprocessor(
            args.checkpoint_dir, dataset_stats, args.device)

        print(f"\n[5/6] Extracting features (with_proc={not args.no_proc}) ...")
        (action_feats, gt_actions,
         bb_f_list, bb_m_list, bb_proc_f_list, bb_proc_m_list,
         vlm_orig_f_list, vlm_orig_m_list, vlm_orig_proc_f_list, vlm_orig_proc_m_list,
         meta) = extract_features(policy, preprocessor, manifest, args.batch_size, args.device,
                                  with_proc=not args.no_proc)
        save_cache(cache_path, action_feats, gt_actions,
                   bb_f_list, bb_m_list, bb_proc_f_list, bb_proc_m_list,
                   vlm_orig_f_list, vlm_orig_m_list, vlm_orig_proc_f_list, vlm_orig_proc_m_list,
                   meta)
        print(f"  [cache] saved → {cache_path}")
        del policy, preprocessor

    if args.no_proc:
        bb_proc_f_list, bb_proc_m_list = [], []
        vlm_orig_proc_f_list, vlm_orig_proc_m_list = [], []

    N = action_feats.shape[0]
    desc_counts = defaultdict(int)
    for d in meta["descriptions"]:
        desc_counts[d] += 1
    print(f"\n  N={N}  action_feats={action_feats.shape}  backbone batches={len(bb_f_list)}")
    print(f"  descriptions: {dict(desc_counts)}")
    drift_cos = np.asarray(meta.get("vlm_drift_cos", np.full(N, np.nan, dtype=np.float32)), dtype=np.float32)
    drift_l2 = np.asarray(meta.get("vlm_drift_l2", np.full(N, np.nan, dtype=np.float32)), dtype=np.float32)
    if np.isfinite(drift_cos).any() and np.isfinite(drift_l2).any():
        print(f"  vlm_drift: cos_mean={np.nanmean(drift_cos):.4f}  l2_mean={np.nanmean(drift_l2):.4f}")
    has_vlm_orig = bool(vlm_orig_f_list)

    zscore = not args.no_zscore

    # ── 3. Backbone agg (shared across all action_agg runs) ──
    print(f"\n[6/N] Aggregating backbone ({args.backbone_agg}) + t-SNE ...")
    X_backbone = agg_backbone(bb_f_list, bb_m_list, args.backbone_agg)
    X_pca_bb, X2d_bb = run_tsne(X_backbone, args.perplexity, args.tsne_n_iter, args.seed,
                                "backbone_raw", zscore=zscore)
    if X2d_bb is None:
        print("[error] backbone t-SNE failed."); sys.exit(1)
    # lora_on processed backbone (alt)
    X2d_bb_proc = None
    if bb_proc_f_list:
        X_backbone_proc = agg_backbone(bb_proc_f_list, bb_proc_m_list, args.backbone_agg)
        _, X2d_bb_proc = run_tsne(X_backbone_proc, args.perplexity, args.tsne_n_iter, args.seed,
                                  "backbone_proc", zscore=zscore)
    if has_vlm_orig:
        X_vlm_original = agg_backbone(vlm_orig_f_list, vlm_orig_m_list, args.backbone_agg)
        X_pca_vlm_orig, X2d_vlm_orig = run_tsne(
            X_vlm_original, args.perplexity, args.tsne_n_iter, args.seed,
            "backbone_original_raw", zscore=zscore)
        if X2d_vlm_orig is None:
            print("  [warn] original VLM t-SNE failed; skip original HTML")
            has_vlm_orig = False
        # lora_off processed backbone (alt)
        X2d_orig_proc = None
        if vlm_orig_proc_f_list:
            X_vlm_orig_proc = agg_backbone(vlm_orig_proc_f_list, vlm_orig_proc_m_list, args.backbone_agg)
            _, X2d_orig_proc = run_tsne(X_vlm_orig_proc, args.perplexity, args.tsne_n_iter, args.seed,
                                        "backbone_original_proc", zscore=zscore)
    else:
        X_vlm_original = None
        X_pca_vlm_orig, X2d_vlm_orig, X2d_orig_proc = None, None, None

    # ── 4. Video clips (shared) ──
    clips_dirs = {}
    if not args.skip_clips:
        print(f"\n  extracting video clips ...")
        w, h = map(int, args.clip_size.split("x"))
        clips_dirs = extract_video_clips(
            manifest, args.cam_keys, args.action_horizon,
            out_dir / "clips", fps=20, output_size=(w, h),
            reuse_existing=args.reuse_clips)

    # ── 5. Per action_agg: t-SNE + HTML ──
    tag_base = f"tsne_p{args.perplexity}" + ("_noz" if not zscore else "")
    print(f"\n  action_agg methods: {args.action_agg_list}")

    for action_agg in args.action_agg_list:
        print(f"\n── action_agg={action_agg} ──")
        if action_agg == "rp_flatten":
            X_action = random_project_flatten(action_feats, args.rp_dim, args.seed)
            action_tag = f"{action_agg}{args.rp_dim}"
        else:
            X_action = agg_action(action_feats, action_agg, gt_actions=gt_actions)
            action_tag = action_agg
        print(f"  X_action={X_action.shape}")
        X_pca_act, X2d_act = run_tsne(X_action, args.perplexity, args.tsne_n_iter, args.seed,
                                      f"action_encoder_{action_tag}", zscore=zscore)
        if X2d_act is None:
            print(f"  [skip] t-SNE failed for action_agg={action_agg}"); continue

        print(f"  computing distance matrices ...")
        dist_b64 = compute_dist_matrices(X_action, X_backbone, X_pca_act, X_pca_bb)

        tag = f"{action_tag}_{tag_base}"
        plot_overview_by_group(X2d_act, X2d_bb, meta["groups"], meta["descriptions"],
                               args.perplexity, out_dir / f"overview_by_group_{tag}.png")
        make_joint_html(
            X2d_act, X2d_bb, X_pca_act, X_pca_bb,
            groups=meta["groups"],
            descriptions=meta["descriptions"],
            frame_indices=meta["base_idxs"],
            action_agg=action_agg, backbone_agg="raw",
            perplexity=args.perplexity,
            out_path=out_dir / f"joint_{tag}.html",
            clips_dirs=clips_dirs,
            zscore=zscore,
            dist_b64=dist_b64,
            vlm_drift_cos=drift_cos,
            vlm_drift_l2=drift_l2,
            X2d_backbone_alt=X2d_bb_proc,
            backbone_agg_alt="processed",
        )
        if has_vlm_orig and X_vlm_original is not None:
            print(f"  computing original VLM distance matrices ...")
            dist_orig_b64 = compute_dist_matrices(X_action, X_vlm_original, X_pca_act, X_pca_vlm_orig)
            original_tag = f"{action_tag}_original_{tag_base}"
            plot_overview_by_group(
                X2d_act, X2d_vlm_orig, meta["groups"], meta["descriptions"],
                args.perplexity, out_dir / f"overview_by_group_{original_tag}.png",
                right_label="backbone original",
            )
            make_joint_html(
                X2d_act, X2d_vlm_orig, X_pca_act, X_pca_vlm_orig,
                groups=meta["groups"],
                descriptions=meta["descriptions"],
                frame_indices=meta["base_idxs"],
                action_agg=action_agg, backbone_agg="original_raw",
                perplexity=args.perplexity,
                out_path=out_dir / f"joint_{original_tag}.html",
                clips_dirs=clips_dirs,
                zscore=zscore,
                dist_b64=dist_orig_b64,
                vlm_drift_cos=drift_cos,
                vlm_drift_l2=drift_l2,
                backbone_label="backbone original",
                X2d_backbone_alt=X2d_orig_proc,
                backbone_agg_alt="original_processed",
            )

    print(f"\nDone. Output -> {out_dir}")


if __name__ == "__main__":
    main()
