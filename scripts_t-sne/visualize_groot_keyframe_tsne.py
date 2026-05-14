#!/usr/bin/env python3
"""
Paired task t-SNE / interactive visualization for GROOT keyframe model.
INSIGHTfixposV3: cabinet / door / bottle task groups.

Features:
  - Backbone + action_encoder dual t-SNE (side-by-side)
  - Per-description coloring with group toggle buttons
  - Click a point → video clip playback (A/B comparison)
  - A/B feature distance panel (cosine + L2, full + PCA)
  - Keyframe injection + task description replacement (same as training)

Usage:
    cd ~/ws3/lerobot
    conda run -n groot python -u src/lerobot/scripts/visualize_groot_keyframe_tsne.py \
        --checkpoint_dir /path/to/pretrained_model \
        --dataset_root /path/to/INSIGHTfixposV3 \
        --keyframe_registry_path /path/to/frame_index_registry.json \
        --samples_per_task 32 \
        --batch_size 16 \
        --task_prompt_mode non_guide \
        --output_dir ./outputs/tsne_keyframe_paired
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).parent
_KEYFRAME_PKG_DIR = Path(__file__).parent.parent / "policies" / "groot_keyframe"
TASK_DESCRIPTIONS_PATHS = {
    "guide":     _KEYFRAME_PKG_DIR / "task_descriptions.json",
    "non_guide": _KEYFRAME_PKG_DIR / "task_descriptions_non_guide.json",
}

# cabinet / door / bottle 그룹 정의
TASK_GROUPS: dict[str, list[str]] = {
    "cabinet": ["1ext"],
    "door":    ["3a", "3b", "3c", "3d"],
    "bottle":  ["5a", "5b", "5c", "5d", "5e", "5f", "5g", "5h"],
}

DEFAULT_CAM_KEYS = ["wrist", "right_shoulder", "guide", "keyframe"]

POP_KEYS = [
    "observation.images.wrist_semantic",
    "observation.images.right_shoulder_semantic",
    "observation.images.left_shoulder_semantic",
    "observation.images.guide_semantic",
    "observation.images.left_shoulder",
]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--dataset_repo_id", type=str, default="paragon7060/INSIGHTfixposV3")
    p.add_argument("--dataset_root", type=str,
                   default="/home/seonho/workspace/data/paragon7060/INSIGHTfixposV3")
    p.add_argument("--keyframe_registry_path", type=str,
                   default="~/clvla/memory_module/keyframe_output/frame_index_registry.json")
    p.add_argument("--task_descriptions_path", type=str, default=None)
    p.add_argument("--task_prompt_mode", type=str, default="non_guide",
                   choices=["guide", "non_guide"])
    p.add_argument("--action_dim",  type=int, default=8)
    p.add_argument("--state_dim",   type=int, default=16)
    p.add_argument("--samples_per_task", type=int, default=32,
                   help="Frames to sample per task_name")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--output_dir",  type=str, default="./outputs/tsne_keyframe_paired")
    p.add_argument("--cache_dir",   type=str, default=None,
                   help="Cache directory (default: output_dir/cache)")
    p.add_argument("--action_agg_list", type=str, nargs="+",
                   default=["flatten", "delta_mean"],
                   choices=["mean", "max", "last", "first_last", "delta_total",
                            "delta_mean", "flatten", "first_mid_last"])
    p.add_argument("--backbone_agg", type=str, default="mask_mean",
                   choices=["mask_mean", "mean", "max", "first", "last"])
    p.add_argument("--perplexity",  type=int, default=30)
    p.add_argument("--tsne_n_iter", type=int, default=1000)
    p.add_argument("--cam_keys", type=str, nargs="*", default=DEFAULT_CAM_KEYS,
                   help="Camera keys for video clip extraction (short names, e.g. wrist)")
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--clip_size", type=str, default="320x240")
    p.add_argument("--skip_clips",         action="store_true")
    p.add_argument("--skip_feature_cache", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Keyframe utilities
# ══════════════════════════════════════════════════════════════════════════════

def _remap_file_path(file_path: str, registry_path: Path) -> str:
    marker = "keyframe_output/"
    idx = file_path.find(marker)
    if idx == -1:
        return file_path
    return str(registry_path.parent / file_path[idx + len(marker):])


def load_keyframe_registry(registry_path: Path) -> dict:
    with open(registry_path) as f:
        entries = json.load(f)
    kf = {}
    for e in entries:
        if e["cropped"] and e["episode_id"] not in kf:
            m = re.search(r"task_(\w+)[/\\]", e["file_path"])
            task_id = m.group(1) if m else ""
            source_cam = "right_shoulder" if task_id.startswith("3") else "wrist"
            kf[e["episode_id"]] = {
                "frame_index": e["frame_index"],
                "file_path": _remap_file_path(e["file_path"], registry_path),
                "source_cam": source_cam,
            }
    return kf


def get_valid_episodes(registry_path: Path) -> list:
    with open(registry_path) as f:
        entries = json.load(f)
    return sorted({e["episode_id"] for e in entries if e["cropped"]})


def build_task_desc_map(task_descriptions_path: Path, dataset_root: Path) -> dict:
    with open(task_descriptions_path) as f:
        name_to_desc: dict = json.load(f)
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet").reset_index()
    return {row["index"]: name_to_desc.get(row["index"], row["index"])
            for _, row in tasks_df.iterrows()}


_kf_img_cache: dict = {}

def _load_keyframe_tensor(file_path: str, h: int, w: int) -> torch.Tensor:
    key = f"{file_path}_{h}_{w}"
    if key not in _kf_img_cache:
        img = Image.open(file_path).convert("RGB").resize((w, h), Image.BILINEAR)
        _kf_img_cache[key] = transforms.ToTensor()(img)
    return _kf_img_cache[key]


def inject_keyframe(batch: dict, keyframe_dict: dict) -> dict:
    wrist = batch["observation.images.wrist"]
    B, C, H, W = wrist.shape
    kf_imgs = []
    for i in range(B):
        ep_id = batch["episode_index"][i].item()
        fr_idx = batch["frame_index"][i].item()
        kf = keyframe_dict.get(ep_id)
        if kf is None or fr_idx < kf["frame_index"]:
            src_key = f"observation.images.{kf['source_cam'] if kf else 'wrist'}"
            kf_imgs.append(batch[src_key][i].clone())
        else:
            kf_imgs.append(_load_keyframe_tensor(kf["file_path"], H, W).to(wrist.device))
    batch["observation.images.keyframe"] = torch.stack(kf_imgs)
    return batch


def slice_feature_dims(batch: dict, action_dim: int, state_dim: int) -> dict:
    if "action" in batch and action_dim is not None:
        batch["action"] = batch["action"][..., :action_dim]
    if "observation.state" in batch and state_dim is not None:
        batch["observation.state"] = batch["observation.state"][..., :state_dim]
    return batch


def slice_dataset_stats(stats, action_dim: int, state_dim: int):
    if not stats:
        return stats
    def _trim(sub, d):
        return {k: v[..., :d].contiguous()
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[-1] > d else v
                for k, v in sub.items()}
    if "action" in stats and action_dim is not None:
        stats["action"] = _trim(stats["action"], action_dim)
    if "observation.state" in stats and state_dim is not None:
        stats["observation.state"] = _trim(stats["observation.state"], state_dim)
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_policy_with_vision_lora(checkpoint_dir: Path, device: str):
    from safetensors.torch import load_model as safetensors_load_model
    from lerobot.configs.policies import PreTrainedConfig

    with open(checkpoint_dir / "config.json") as f:
        config_data = json.load(f)
    lora_rank    = config_data.get("lora_rank", 0)
    lora_alpha   = config_data.get("lora_alpha", 16)
    lora_dropout = config_data.get("lora_dropout", 0.05)
    policy_type  = config_data.get("type", config_data.get("policy_type", "groot"))

    config = PreTrainedConfig.from_pretrained(str(checkpoint_dir))
    config.lora_rank = 0

    if policy_type == "groot_cl":
        from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
        policy = GrootCLPolicy(config)
    else:
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        policy = GrootPolicy(config)

    eagle = policy._groot_model.backbone.eagle_model
    lora_target = config_data.get("lora_target", "vision")
    if lora_target in ("vision", "both"):
        eagle.wrap_backbone_lora(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        print(f"  [load] vision LoRA: rank={lora_rank}")
    if lora_target in ("llm", "both"):
        eagle.wrap_llm_lora(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    missing, unexpected = safetensors_load_model(
        policy, str(checkpoint_dir / "model.safetensors"), strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)}")
    return policy


def load_model_and_preprocessor(checkpoint_dir: str, dataset_stats: dict, device: str):
    from lerobot.policies.factory import make_pre_post_processors
    checkpoint_dir = Path(checkpoint_dir)
    print(f"  loading policy from {checkpoint_dir} ...")

    with open(checkpoint_dir / "config.json") as f:
        config_data = json.load(f)
    lora_rank   = config_data.get("lora_rank", 0)
    lora_target = config_data.get("lora_target", "llm")

    if lora_rank > 0 and lora_target in ("vision", "both"):
        policy = _load_policy_with_vision_lora(checkpoint_dir, device)
    else:
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        policy = GrootPolicy.from_pretrained(str(checkpoint_dir))

    policy.eval().to(device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_dir),
        dataset_stats=dataset_stats,
    )
    print("  policy + preprocessor ready.")
    return policy, preprocessor


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading + manifest
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(repo_id, root, valid_episodes, delta_timestamps=None, pop_keys=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    kw = dict(repo_id=repo_id, root=root, video_backend="pyav", episodes=valid_episodes)
    if delta_timestamps:
        kw["delta_timestamps"] = delta_timestamps
    dataset = LeRobotDataset(**kw)
    for key in (pop_keys or POP_KEYS):
        for attr in ("features", "stats"):
            d = getattr(dataset.meta, attr, {})
            if key in d:
                d.pop(key)
        if dataset.delta_indices and key in dataset.delta_indices:
            dataset.delta_indices.pop(key)
        if dataset.delta_timestamps and key in dataset.delta_timestamps:
            dataset.delta_timestamps.pop(key)
    return dataset


def build_task_index_map(dataset) -> dict[str, list[int]]:
    """task_name → list of local (filtered) dataset indices"""
    # tasks DataFrame: index = task_name, column = task_index (int)
    task_name_to_int = {name: int(row.task_index)
                        for name, row in dataset.meta.tasks.iterrows()}
    int_to_task_name = {v: k for k, v in task_name_to_int.items()}

    task_to_local: dict[str, list[int]] = defaultdict(list)
    for local_idx, ti in enumerate(dataset.hf_dataset["task_index"]):
        name = int_to_task_name.get(int(ti))
        if name:
            task_to_local[name].append(local_idx)
    return dict(task_to_local)


def build_manifest(dataset, task_to_local: dict, task_groups: dict,
                   task_name_to_desc: dict, samples_per_task: int,
                   seed: int) -> list[dict]:
    rng = random.Random(seed)
    group_of_task = {t: g for g, tasks in task_groups.items() for t in tasks}
    hf = dataset.hf_dataset

    manifest = []
    for task_name, local_idxs in task_to_local.items():
        group = group_of_task.get(task_name)
        if group is None:
            continue
        desc = task_name_to_desc.get(task_name, task_name)
        chosen = rng.sample(local_idxs, min(samples_per_task, len(local_idxs)))
        for local_idx in chosen:
            row = hf[local_idx]
            manifest.append({
                "group":         group,
                "task_name":     task_name,
                "description":   desc,
                "local_idx":     local_idx,
                "episode_index": int(row["episode_index"]),
                "frame_index":   int(row["frame_index"]),
                "ds_index":      int(row["index"]),
            })
    rng.shuffle(manifest)
    print(f"  manifest: {len(manifest)} samples across "
          f"{len({m['task_name'] for m in manifest})} tasks")
    return manifest


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
def extract_features(policy, preprocessor, dataset, manifest, keyframe_dict,
                     task_name_to_desc, action_dim, state_dim, batch_size, device):
    use_bf16  = getattr(policy.config, "use_bf16", True)
    dev_type  = "cuda" if device != "cpu" else "cpu"
    backbone  = policy._groot_model.backbone
    action_encoder = getattr(policy._groot_model.action_head, "action_encoder", None)
    has_action_enc = action_encoder is not None
    if not has_action_enc:
        print("  [warn] no action_encoder found — action features will be empty")

    bb_f_list, bb_m_list = [], []
    ac_list, gt_list = [], []
    N = len(manifest)

    for start in range(0, N, batch_size):
        chunk = manifest[start:start + batch_size]
        # Collect raw samples from dataset
        raw_samples = [dataset[m["local_idx"]] for m in chunk]

        # Collate
        batch = {}
        for k in raw_samples[0].keys():
            vals = [s[k] for s in raw_samples]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals, dim=0)
            else:
                batch[k] = vals

        # Keyframe injection
        batch = inject_keyframe(batch, keyframe_dict)

        # Slice dims
        batch = slice_feature_dims(batch, action_dim, state_dim)

        # Task description replacement
        batch["task"] = [task_name_to_desc.get(m["task_name"], m["task_name"]) for m in chunk]

        # Preprocess
        processed = preprocessor(batch)

        # Backbone
        bin_ = _build_backbone_input(processed, device, use_bf16)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
            bb_out = backbone(bin_)
        bb_f_list.append(bb_out["backbone_features"].detach().float().cpu())
        bb_m_list.append(bb_out["backbone_attention_mask"].detach().cpu())

        # Action encoder
        if has_action_enc:
            try:
                _, action_inputs = policy._groot_model.prepare_input(processed)
                gt_actions    = action_inputs.action
                embodiment_id = action_inputs.embodiment_id
                B = gt_actions.shape[0]
                t_clean = torch.full((B,), 999, dtype=torch.long,
                                     device=gt_actions.device)
                with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
                    enc_out = action_encoder(gt_actions, t_clean, embodiment_id)
                ac_list.append(enc_out.float().cpu())
                gt_list.append(gt_actions.float().cpu())
            except Exception as e:
                print(f"\n  [warn] action_encoder failed: {e}")
                has_action_enc = False

        print(f"  [extract] {min(start + batch_size, N)}/{N} ...", end="\r")
    print()

    action_feats = (np.concatenate([x.numpy() for x in ac_list], axis=0)
                    if ac_list else np.zeros((N, 1), dtype=np.float32))
    gt_actions_np = (np.concatenate([x.numpy() for x in gt_list], axis=0)
                     if gt_list else np.zeros((N, 1, 1), dtype=np.float32))
    return action_feats, gt_actions_np, bb_f_list, bb_m_list


# ══════════════════════════════════════════════════════════════════════════════
# Cache I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_cache(path: Path, action_feats, gt_actions, bb_f_list, bb_m_list, manifest):
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
        groups=np.array([m["group"]       for m in manifest]),
        task_names=np.array([m["task_name"]   for m in manifest]),
        descriptions=np.array([m["description"] for m in manifest]),
        episode_indices=np.array([m["episode_index"] for m in manifest], dtype=np.int64),
        frame_indices=np.array([m["frame_index"]   for m in manifest], dtype=np.int64),
        ds_indices=np.array([m["ds_index"]      for m in manifest], dtype=np.int64),
    )
    print(f"  [cache] saved → {path}")


def load_cache(path: Path):
    npz = np.load(path, allow_pickle=True)
    bb_f = [torch.from_numpy(x.astype(np.float32)) for x in npz["bb_feats"]]
    bb_m = [torch.from_numpy(x) for x in npz["bb_masks"]]
    manifest = [{
        "group":         str(npz["groups"][i]),
        "task_name":     str(npz["task_names"][i]),
        "description":   str(npz["descriptions"][i]),
        "episode_index": int(npz["episode_indices"][i]),
        "frame_index":   int(npz["frame_indices"][i]),
        "ds_index":      int(npz["ds_indices"][i]),
        "local_idx":     -1,
    } for i in range(len(npz["groups"]))]
    return npz["action_feats"], npz["gt_actions"], bb_f, bb_m, manifest


# ══════════════════════════════════════════════════════════════════════════════
# Video clip extraction
# ══════════════════════════════════════════════════════════════════════════════

def load_episode_video_meta(dataset_root: Path) -> dict[int, dict]:
    """Read meta/episodes parquets → {episode_index: {obs_key: {chunk_index, file_index, from_timestamp}}}"""
    import pandas as pd
    ep_dir = dataset_root / "meta" / "episodes"
    if not ep_dir.exists():
        return {}
    result = {}
    for chunk_file in sorted(ep_dir.iterdir()):
        df = pd.read_parquet(chunk_file)
        for _, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            meta = {}
            for col in row.index:
                if col.startswith("videos/") and col.endswith("/chunk_index"):
                    obs_key = col[len("videos/"):-len("/chunk_index")]
                    meta[obs_key] = {
                        "chunk_index":    int(row[col]),
                        "file_index":     int(row[f"videos/{obs_key}/file_index"]),
                        "from_timestamp": float(row[f"videos/{obs_key}/from_timestamp"]),
                    }
            result[ep_idx] = meta
    return result


def _get_video_path(dataset_root: Path, obs_key: str, episode_index: int,
                    ep_video_meta: dict | None = None) -> tuple[Path | None, float]:
    """Returns (video_path, from_timestamp).
    Uses episodes parquet metadata (LeRobot v3 format) when available."""
    if ep_video_meta:
        ep_meta = ep_video_meta.get(episode_index, {}).get(obs_key)
        if ep_meta is not None:
            path = (dataset_root / "videos" / obs_key
                    / f"chunk-{ep_meta['chunk_index']:03d}"
                    / f"file-{ep_meta['file_index']:03d}.mp4")
            return (path if path.exists() else None), ep_meta["from_timestamp"]
    # fallback: legacy per-episode file naming
    chunk = episode_index // 1000
    for p in [
        dataset_root / "videos" / obs_key / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.mp4",
        dataset_root / "videos" / f"chunk-{chunk:03d}" / obs_key / f"episode_{episode_index:06d}.mp4",
        dataset_root / "videos" / obs_key / f"episode_{episode_index:06d}.mp4",
    ]:
        if p.exists():
            return p, 0.0
    return None, 0.0


def _extract_keyframe_clip(m: dict, keyframe_dict: dict,
                           dataset_root: Path, ep_video_meta: dict | None,
                           horizon: int, fps: int, W: int, H: int) -> list:
    """Build keyframe-slot frames: source_cam video before KF, static KF image after."""
    import av
    ep_id   = m["episode_index"]
    fr_idx  = m["frame_index"]
    kf      = keyframe_dict.get(ep_id)
    kf_fi   = kf["frame_index"] if kf else None

    src_ck  = kf["source_cam"] if kf else "wrist"
    src_obs = f"observation.images.{src_ck}"
    vpath, from_ts = _get_video_path(dataset_root, src_obs, ep_id, ep_video_meta)

    # Get source_cam frames
    src_frames = []
    if vpath is not None:
        start_sec = from_ts + fr_idx / fps
        try:
            with av.open(str(vpath)) as vc:
                stream = vc.streams.video[0]
                tb = float(stream.time_base)
                vc.seek(int(start_sec / tb), stream=stream, backward=True)
                for pkt in vc.demux(stream):
                    for frm in pkt.decode():
                        fts = float(frm.pts * tb) if frm.pts is not None else 0.0
                        if fts < start_sec - 1.0 / fps:
                            continue
                        src_frames.append(frm.to_ndarray(format="rgb24"))
                        if len(src_frames) >= horizon:
                            break
                    if len(src_frames) >= horizon:
                        break
        except Exception:
            pass

    # Load static keyframe image (shown from kf_frame_index onward)
    kf_np = None
    if kf and kf_fi is not None and fr_idx >= kf_fi:
        try:
            pil = Image.open(kf["file_path"]).convert("RGB").resize((W, H), Image.BILINEAR)
            kf_np = np.array(pil)
        except Exception:
            pass

    raw_frames = []
    for i in range(horizon):
        abs_fi = fr_idx + i
        if kf_np is not None and kf_fi is not None and abs_fi >= kf_fi:
            raw_frames.append(kf_np)
        elif i < len(src_frames):
            raw_frames.append(src_frames[i])
        elif src_frames:
            raw_frames.append(src_frames[-1])  # repeat last frame
    return raw_frames


def extract_video_clips(manifest: list[dict], cam_keys: list[str],
                        horizon: int, out_dir: Path,
                        fps: int = 20,
                        dataset_root: Path | None = None,
                        output_size=(320, 240),
                        ep_video_meta: dict | None = None,
                        keyframe_dict: dict | None = None):
    import av
    import io as _io

    W, H = output_size
    cam_dirs = {}
    for ck in cam_keys:
        d = out_dir / ck
        d.mkdir(parents=True, exist_ok=True)
        cam_dirs[ck] = d

    for si, m in enumerate(manifest):
        for ck in cam_keys:
            out_file = cam_dirs[ck] / f"{si:04d}.mp4"
            if out_file.exists():
                continue
            if dataset_root is None:
                continue

            # ── keyframe slot: synthetic clip from registry ───────────────
            if ck == "keyframe":
                if keyframe_dict is None:
                    continue
                raw_frames = _extract_keyframe_clip(
                    m, keyframe_dict, dataset_root, ep_video_meta, horizon, fps, W, H)
                if not raw_frames:
                    continue
                import io as _io2
                buf = _io2.BytesIO()
                with av.open(buf, mode="w", format="mp4") as oc:
                    ost = oc.add_stream("libx264", rate=fps)
                    ost.width = W; ost.height = H; ost.pix_fmt = "yuv420p"
                    ost.options = {"crf": "28", "preset": "ultrafast"}
                    for rf in raw_frames:
                        pil = Image.fromarray(rf)
                        if pil.size != (W, H):
                            pil = pil.resize((W, H), Image.LANCZOS)
                        avf = av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")
                        for pkt in ost.encode(avf): oc.mux(pkt)
                    for pkt in ost.encode(None): oc.mux(pkt)
                with open(out_file, "wb") as fh:
                    fh.write(buf.getvalue())
                continue

            obs_key = f"observation.images.{ck}"
            vpath, from_ts = _get_video_path(dataset_root, obs_key, m["episode_index"], ep_video_meta)
            if vpath is None:
                if si == 0:
                    print(f"  [clips] WARNING: no video found for cam='{ck}' ep={m['episode_index']} "
                          f"(obs_key={obs_key}) — skipping this camera")
                continue
            start_sec = from_ts + m["frame_index"] / fps
            try:
                with av.open(str(vpath)) as vc:
                    stream = vc.streams.video[0]
                    tb = float(stream.time_base)
                    vc.seek(int(start_sec / tb), stream=stream, backward=True)
                    raw_frames = []
                    for pkt in vc.demux(stream):
                        for frm in pkt.decode():
                            fts = float(frm.pts * tb) if frm.pts is not None else 0.0
                            if fts < start_sec - 1.0 / fps:
                                continue
                            raw_frames.append(frm.to_ndarray(format="rgb24"))
                            if len(raw_frames) >= horizon:
                                break
                        if len(raw_frames) >= horizon:
                            break
            except Exception as e:
                print(f"  [clips] {ck}/{si}: {e}")
                continue
            if not raw_frames:
                continue
            buf = _io.BytesIO()
            with av.open(buf, mode="w", format="mp4") as oc:
                ost = oc.add_stream("libx264", rate=fps)
                ost.width = W; ost.height = H; ost.pix_fmt = "yuv420p"
                ost.options = {"crf": "28", "preset": "ultrafast"}
                for rf in raw_frames[:horizon]:
                    pil = Image.fromarray(rf)
                    if pil.size != (W, H):
                        pil = pil.resize((W, H), Image.LANCZOS)
                    avf = av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")
                    for pkt in ost.encode(avf):
                        oc.mux(pkt)
                for pkt in ost.encode(None):
                    oc.mux(pkt)
            with open(out_file, "wb") as fh:
                fh.write(buf.getvalue())
        if (si + 1) % 20 == 0:
            print(f"  [clips] {si+1}/{len(manifest)}", end="\r")
    print()
    return cam_dirs


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def agg_backbone(feats_list, masks_list, method) -> np.ndarray:
    parts = []
    for f, m in zip(feats_list, masks_list):
        if method == "mask_mean":
            m_f = m.unsqueeze(-1).float()
            agg = (f * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)
        elif method == "mean":   agg = f.mean(dim=1)
        elif method == "max":
            m_f = m.unsqueeze(-1).float()
            agg = (f * m_f + (1.0 - m_f) * (-1e9)).max(dim=1).values
        elif method == "first":  agg = f[:, 0, :]
        elif method == "last":
            lengths = m.sum(dim=1).long().clamp(min=1)
            idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, f.shape[-1])
            agg = f.gather(dim=1, index=idx).squeeze(1)
        else: raise ValueError(method)
        parts.append(agg)
    return torch.cat(parts, dim=0).numpy()


def agg_action(feats: np.ndarray, method: str,
               gt_actions: np.ndarray | None = None) -> np.ndarray:
    t = torch.from_numpy(feats)
    if t.ndim == 2:
        return t.numpy()   # already 2D (no temporal dim)
    N, T, D = t.shape
    if method == "mean":          out = t.mean(dim=1)
    elif method == "max":         out = t.max(dim=1).values
    elif method == "last":        out = t[:, -1, :]
    elif method == "first_last":  out = (t[:, 0, :] + t[:, -1, :]) / 2.0
    elif method == "delta_total": out = t[:, -1, :] - t[:, 0, :]
    elif method == "delta_mean":  out = (t[:, 1:, :] - t[:, :-1, :]).mean(dim=1)
    elif method == "flatten":     out = t.reshape(N, T * D)
    elif method == "first_mid_last":
        mid = T // 2
        out = torch.cat([t[:, 0, :], t[:, mid, :], t[:, -1, :]], dim=-1)
    else: raise ValueError(method)
    return out.numpy()


# ══════════════════════════════════════════════════════════════════════════════
# PCA + t-SNE
# ══════════════════════════════════════════════════════════════════════════════

def pca_preprocess(X: np.ndarray, n_components: int = 50) -> np.ndarray | None:
    from sklearn.decomposition import PCA
    var = X.var(axis=0)
    X_c = X[:, var > 1e-10]
    if X_c.shape[1] == 0:
        return None
    mean = X_c.mean(axis=0); std = X_c.std(axis=0).clip(min=1e-10)
    X_s = (X_c - mean) / std
    n = min(n_components, X_s.shape[0] - 1, X_s.shape[1])
    out = PCA(n_components=n, random_state=0).fit_transform(X_s)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def run_tsne(X: np.ndarray, perplexity: int, n_iter: int,
             seed: int, label: str = "") -> tuple[np.ndarray | None, np.ndarray | None]:
    from sklearn.manifold import TSNE
    import sklearn
    X_pca = pca_preprocess(X)
    if X_pca is None:
        print(f"  [skip] {label}: all constant")
        return None, None
    perp = min(perplexity, max(5.0, (X_pca.shape[0] - 1) / 3.0 - 1))
    kw = dict(n_components=2, perplexity=perp, random_state=seed, verbose=0)
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
        kw["max_iter"] = n_iter
    else:
        kw["n_iter"] = n_iter
    print(f"  [t-SNE] {label} shape={X_pca.shape} perp={perp:.0f} ...")
    return X_pca, TSNE(**kw).fit_transform(X_pca)


# ══════════════════════════════════════════════════════════════════════════════
# Feature distance
# ══════════════════════════════════════════════════════════════════════════════

def _upper_tri_b64(mat: np.ndarray) -> str:
    idx = np.triu_indices(mat.shape[0], k=1)
    return base64.b64encode(mat[idx].astype(np.float32).tobytes()).decode("ascii")


def compute_dist_matrices(X_action, X_backbone, X_pca_action, X_pca_backbone) -> dict:
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
    out = {}
    for name, X in [("action_full", X_action),   ("action_pca", X_pca_action),
                    ("bb_full",     X_backbone),  ("bb_pca",     X_pca_backbone)]:
        print(f"  [dist] {name} shape={X.shape} ...", end=" ", flush=True)
        c = cosine_distances(X); l = euclidean_distances(X)
        out[f"{name}_cos"] = _upper_tri_b64(c)
        out[f"{name}_l2"]  = _upper_tri_b64(l)
        print(f"{(len(out[f'{name}_cos'])+len(out[f'{name}_l2']))/1e6:.1f} MB")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Static overview plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview(X2d_a, X2d_b, groups, task_names, descriptions,
                  perplexity, action_agg, backbone_agg, out_path):
    all_groups = sorted(set(groups))
    n = max(len(all_groups), 1)
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(n)
    gcolor = {g: cmap(i) for i, g in enumerate(all_groups)}
    arr_g = np.array(groups)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, X2d, title in [(axes[0], X2d_a, f"action_encoder ({action_agg})"),
                            (axes[1], X2d_b, f"backbone ({backbone_agg})")]:
        for g in all_groups:
            m = arr_g == g
            ax.scatter(X2d[m, 0], X2d[m, 1],
                       c=[gcolor[g]], label=g, s=12, alpha=0.7, linewidths=0)
        ax.legend(fontsize=9, markerscale=2)
        ax.set_title(title); ax.grid(True, alpha=0.2)
    fig.suptitle(f"t-SNE perp={perplexity}  N={len(groups)}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path}")


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
        l_val = light_cycle[i % len(light_cycle)]
        r, g, b = colorsys.hls_to_rgb(h, l_val, s)
        colors[did] = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
    return colors


# ══════════════════════════════════════════════════════════════════════════════
# Interactive HTML
# ══════════════════════════════════════════════════════════════════════════════

def make_joint_html(X2d_action, X2d_backbone,
                    X_pca_action, X_pca_backbone,
                    manifest, action_agg, backbone_agg,
                    perplexity, out_path: Path,
                    clips_dirs: dict | None = None,
                    dist_b64: dict | None = None):
    groups      = [m["group"]       for m in manifest]
    task_names  = [m["task_name"]   for m in manifest]
    descriptions = [m["description"] for m in manifest]
    frame_idxs  = [m["frame_index"] for m in manifest]
    N = len(manifest)

    all_groups = sorted(set(groups))
    all_task_names_set = sorted(set(task_names))
    task_color = _desc_color_map(all_task_names_set)

    # trace: one per (group, task_name)
    pair_to_idx: dict[tuple, list[int]] = defaultdict(list)
    for i, (g, t) in enumerate(zip(groups, task_names)):
        pair_to_idx[(g, t)].append(i)
    pairs = sorted(pair_to_idx.keys())
    group_of_trace = [g for g, _ in pairs]

    def _make_traces(X2d):
        traces = []
        for (g, t) in pairs:
            idxs = pair_to_idx[(g, t)]
            traces.append({
                "type": "scatter",
                "x": X2d[idxs, 0].tolist(),
                "y": X2d[idxs, 1].tolist(),
                "mode": "markers",
                "name": f"[{g}] {t}",
                "legendgroup": g,
                "legendgrouptitle": {"text": g},
                "marker": {"color": task_color[t], "size": 7, "opacity": 0.85},
                "customdata": [[descriptions[i], g, task_names[i], i, frame_idxs[i]]
                               for i in idxs],
                "hovertemplate": (
                    "<b>%{customdata[1]}</b> / %{customdata[2]}<br>"
                    "%{customdata[0]}<br>"
                    "sample: %{customdata[3]}  frame: %{customdata[4]}"
                    "<extra></extra>"
                ),
            })
        # A / B highlight traces
        for color in ["#2196F3", "#4CAF50"]:
            traces.append({
                "type": "scatter", "x": [], "y": [], "mode": "markers",
                "name": "", "showlegend": False, "hoverinfo": "skip",
                "marker": {"symbol": "circle-open", "size": 20, "color": color,
                           "line": {"color": color, "width": 3}},
            })
        return traces

    n_data = len(pairs)
    action_traces   = _make_traces(X2d_action)
    backbone_traces = _make_traces(X2d_backbone)

    has_clips = bool(clips_dirs)
    cam_shorts = list(clips_dirs.keys()) if has_clips else []
    video_panel_html = ""
    for cs in cam_shorts:
        video_panel_html += f"""
<div class="cam-group">
  <p class="cam-label">{cs}</p>
  <div class="cam-row">
    <div class="vid-box"><p class="vid-lbl" style="color:#2196F3">A</p>
      <video id="vid_A_{cs}" autoplay loop muted playsinline width="100%"></video></div>
    <div class="vid-box"><p class="vid-lbl" style="color:#4CAF50">B</p>
      <video id="vid_B_{cs}" autoplay loop muted playsinline width="100%"></video></div>
  </div>
</div>"""

    group_btns = "".join(
        f"<button class='cb' onclick='toggleGroup({json.dumps(g)})'>{g}</button>"
        for g in all_groups
    )

    shared_layout = {
        "height": 620, "hovermode": "closest",
        "xaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
        "yaxis": {"showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
        "legend": {"x": 1.01, "y": 1, "xanchor": "left", "font": {"size": 9},
                   "groupclick": "toggleitem", "tracegroupgap": 4},
        "margin": {"l": 50, "r": 280, "t": 60, "b": 40},
    }
    action_layout   = dict(shared_layout, title=f"action_encoder ({action_agg})")
    backbone_layout = dict(shared_layout, title=f"backbone ({backbone_agg})")

    title = (f"Keyframe t-SNE  |  action ({action_agg}) + backbone ({backbone_agg})"
             f"  |  perp={perplexity}  N={N}")

    has_dist = bool(dist_b64)
    dist_b64_json = json.dumps(dist_b64) if has_dist else "null"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body{{font-family:sans-serif;margin:0;padding:6px;background:#f5f5f5;}}
h3{{text-align:center;margin:4px 0;font-size:13px;color:#333;}}
#btn-row{{display:flex;flex-wrap:wrap;gap:4px;padding:5px 8px;background:#fff;
          border:1px solid #ddd;border-radius:6px;margin-bottom:5px;}}
.cb{{font-size:11px;padding:3px 8px;border:1px solid #bbb;border-radius:3px;
     background:#f0f0f0;cursor:pointer;white-space:nowrap;}}
.cb:hover{{background:#dde;}}
#plots-row{{display:flex;gap:6px;}}
#plots-row>div{{flex:1;min-width:0;background:#fff;border:1px solid #ddd;border-radius:6px;}}
#bottom-row{{display:flex;gap:10px;margin-top:8px;align-items:flex-start;}}
#video-panel{{display:flex;gap:12px;padding:8px 10px;background:#fff;
              border:1px solid #ddd;border-radius:6px;flex:3;min-height:60px;flex-wrap:wrap;}}
.cam-group{{display:flex;flex-direction:column;align-items:center;}}
.cam-label{{margin:0 0 3px;font-size:11px;font-weight:bold;color:#333;}}
.cam-row{{display:flex;gap:6px;}}
.vid-box{{text-align:center;}}
.vid-lbl{{margin:0 0 2px;font-size:12px;font-weight:bold;}}
.vid-box video{{width:150px;height:auto;border:2px solid #ccc;border-radius:4px;background:#111;}}
#right-panel{{flex:1;display:flex;flex-direction:column;gap:6px;min-width:240px;}}
#info-box{{padding:8px 10px;background:#fff;border:1px solid #ddd;border-radius:6px;
           font-size:12px;color:#444;white-space:pre-line;}}
#dist-panel{{padding:8px 10px;background:#fff;border:1px solid #ddd;border-radius:6px;
             font-size:11px;color:#333;}}
#dist-panel h4{{margin:0 0 5px;font-size:12px;color:#555;}}
.ds{{margin-bottom:6px;}}
.dst{{font-weight:bold;font-size:10px;color:#888;text-transform:uppercase;
      margin-bottom:2px;border-bottom:1px solid #eee;}}
.dr{{display:flex;justify-content:space-between;margin:2px 0;}}
.dv{{font-weight:bold;font-family:monospace;}}
.dv.f{{color:#1565C0;}} .dv.p{{color:#888;}}
#hint{{text-align:center;font-size:11px;color:#888;margin:2px 0;}}
</style></head><body>
<h3>{title}</h3>
<div id="btn-row">
  <button class="cb" onclick="setAll(true)">All ON</button>
  <button class="cb" onclick="setAll('legendonly')">All OFF</button>
  <span style="color:#bbb;padding:0 4px">|</span>
  {group_btns}
</div>
<div id="hint">click 1st: A (blue) &nbsp;|&nbsp; click 2nd: B (green) + distance &nbsp;|&nbsp; same point: reset</div>
<div id="plots-row">
  <div id="plot_action"></div>
  <div id="plot_backbone"></div>
</div>
<div id="bottom-row">
  <div id="video-panel">
{video_panel_html}
    <span id="vid-hint" style="color:#aaa;font-style:italic;font-size:12px;">← click a point to play clip</span>
  </div>
  <div id="right-panel">
    <div id="info-box"><span style="color:#aaa;font-style:italic;">click a point for details</span></div>
    <div id="dist-panel" style="display:none;">
      <h4>📐 Feature Distance (A vs B)</h4>
      <div class="ds"><div class="dst">Action Encoder ({action_agg})</div>
        <div class="dr"><span>Cos full</span><span class="dv f" id="d_af_c">—</span></div>
        <div class="dr"><span>L2 full</span><span class="dv f" id="d_af_l">—</span></div>
        <div class="dr"><span>Cos PCA50</span><span class="dv p" id="d_ap_c">—</span></div>
        <div class="dr"><span>L2 PCA50</span><span class="dv p" id="d_ap_l">—</span></div>
      </div>
      <div class="ds"><div class="dst">Backbone ({backbone_agg})</div>
        <div class="dr"><span>Cos full</span><span class="dv f" id="d_bf_c">—</span></div>
        <div class="dr"><span>L2 full</span><span class="dv f" id="d_bf_l">—</span></div>
        <div class="dr"><span>Cos PCA50</span><span class="dv p" id="d_bp_c">—</span></div>
        <div class="dr"><span>L2 PCA50</span><span class="dv p" id="d_bp_l">—</span></div>
      </div>
    </div>
  </div>
</div>
<script>
var N={N}, N_DATA={n_data};
var actionXY={json.dumps(X2d_action.tolist())};
var backboneXY={json.dumps(X2d_backbone.tolist())};
var pcaAction={json.dumps(X_pca_action.tolist())};
var pcaBackbone={json.dumps(X_pca_backbone.tolist())};
var descLabels={json.dumps(descriptions)};
var groupLabels={json.dumps(groups)};
var taskNames={json.dumps(task_names)};
var frameIdxs={json.dumps(frame_idxs)};
var camShorts={json.dumps(cam_shorts)};
var groupOfTrace={json.dumps(group_of_trace)};
var traceVis=new Array(N_DATA).fill(true);

function setAll(v){{for(var i=0;i<N_DATA;i++)traceVis[i]=v;
  Plotly.restyle('plot_action',{{visible:new Array(N_DATA).fill(v)}},Array.from({{length:N_DATA}},(_,i)=>i));
  Plotly.restyle('plot_backbone',{{visible:new Array(N_DATA).fill(v)}},Array.from({{length:N_DATA}},(_,i)=>i));}}
function toggleGroup(g){{
  var allOn=true;
  for(var i=0;i<N_DATA;i++){{if(groupOfTrace[i]===g&&traceVis[i]!==true){{allOn=false;break;}}}}
  var v=allOn?'legendonly':true;
  var idxs=[];
  for(var i=0;i<N_DATA;i++){{if(groupOfTrace[i]===g){{idxs.push(i);traceVis[i]=v;}}}}
  Plotly.restyle('plot_action',{{visible:idxs.map(_=>v)}},idxs);
  Plotly.restyle('plot_backbone',{{visible:idxs.map(_=>v)}},idxs);
}}

var distRaw={dist_b64_json};
var distMat={{}};
function decodeDist(b64){{var bin=atob(b64),bytes=new Uint8Array(bin.length);
  for(var i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);return new Float32Array(bytes.buffer);}}
if(distRaw!==null)for(var k in distRaw)distMat[k]=decodeDist(distRaw[k]);
function triIdx(i,j){{if(i>j){{var t=i;i=j;j=t;}}return i*(N-1)-Math.floor(i*(i-1)/2)+(j-i-1);}}
function getDist(k,i,j){{if(i===j)return 0.0;return distMat[k]?distMat[k][triIdx(i,j)]:NaN;}}
function cosPCA(a,b){{var d=0,na=0,nb=0;for(var i=0;i<a.length;i++){{d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}}
  return 1.0-Math.max(-1,Math.min(1,d/(Math.sqrt(na)*Math.sqrt(nb)+1e-10)));}}
function l2PCA(a,b){{var s=0;for(var i=0;i<a.length;i++){{var d=a[i]-b[i];s+=d*d;}}return Math.sqrt(s);}}
function fmt(v){{return isNaN(v)?'—':v.toFixed(4);}}
function pad4(n){{return String(n).padStart(4,'0');}}

var selA=null;
function hlSet(ti,si,pid,xy){{var p=xy[si];Plotly.restyle(pid,{{x:[[p[0]]],y:[[p[1]]]}},ti);}}
function hlClear(ti,pid){{Plotly.restyle(pid,{{x:[[]],y:[[]]}},ti);}}
function playVid(slot,si){{
  if(!camShorts.length)return;
  document.getElementById('vid-hint').style.display='none';
  camShorts.forEach(function(cs){{
    var el=document.getElementById('vid_'+slot+'_'+cs);
    if(!el)return;
    el.pause();el.src='./clips/'+cs+'/'+pad4(si)+'.mp4';
    el.style.border='2px solid '+(slot==='A'?'#2196F3':'#4CAF50');
    el.oncanplay=function(){{el.oncanplay=null;el.play().catch(function(){{}});}};el.load();
  }});
}}
function clearVid(slot){{camShorts.forEach(function(cs){{
  var el=document.getElementById('vid_'+slot+'_'+cs);
  if(el){{el.src='';el.style.border='2px solid #ccc';}}
}});}}
function sampleInfo(i){{return '<b>'+groupLabels[i]+'</b> / '+taskNames[i]+'\\n'+descLabels[i]+'\\nsample:'+i+' frame:'+frameIdxs[i];}}
function showInfo(a,b){{
  var h='<b style="color:#2196F3">A</b> '+sampleInfo(a);
  if(b!==null)h+='\\n\\n<b style="color:#4CAF50">B</b> '+sampleInfo(b);
  document.getElementById('info-box').innerHTML=h;
}}
function showDist(a,b){{
  document.getElementById('d_af_c').textContent=fmt(getDist('action_full_cos',a,b));
  document.getElementById('d_af_l').textContent=fmt(getDist('action_full_l2', a,b));
  document.getElementById('d_bf_c').textContent=fmt(getDist('bb_full_cos',    a,b));
  document.getElementById('d_bf_l').textContent=fmt(getDist('bb_full_l2',     a,b));
  document.getElementById('d_ap_c').textContent=fmt(distMat['action_pca_cos']?getDist('action_pca_cos',a,b):cosPCA(pcaAction[a],pcaAction[b]));
  document.getElementById('d_ap_l').textContent=fmt(distMat['action_pca_l2'] ?getDist('action_pca_l2', a,b):l2PCA(pcaAction[a],pcaAction[b]));
  document.getElementById('d_bp_c').textContent=fmt(distMat['bb_pca_cos']    ?getDist('bb_pca_cos',    a,b):cosPCA(pcaBackbone[a],pcaBackbone[b]));
  document.getElementById('d_bp_l').textContent=fmt(distMat['bb_pca_l2']     ?getDist('bb_pca_l2',     a,b):l2PCA(pcaBackbone[a],pcaBackbone[b]));
  document.getElementById('dist-panel').style.display='block';
}}
function resetSel(){{
  selA=null;
  hlClear(N_DATA,'plot_action');hlClear(N_DATA,'plot_backbone');
  hlClear(N_DATA+1,'plot_action');hlClear(N_DATA+1,'plot_backbone');
  clearVid('A');clearVid('B');
  document.getElementById('info-box').innerHTML='<span style="color:#aaa;font-style:italic;">click a point for details</span>';
  document.getElementById('dist-panel').style.display='none';
}}
function handleClick(si){{
  if(selA===null){{
    selA=si;
    hlSet(N_DATA,si,'plot_action',actionXY);hlSet(N_DATA,si,'plot_backbone',backboneXY);
    hlClear(N_DATA+1,'plot_action');hlClear(N_DATA+1,'plot_backbone');
    playVid('A',si);clearVid('B');showInfo(si,null);
    document.getElementById('dist-panel').style.display='none';
  }}else if(selA===si){{
    resetSel();
  }}else{{
    var a=selA;selA=null;
    hlSet(N_DATA+1,si,'plot_action',actionXY);hlSet(N_DATA+1,si,'plot_backbone',backboneXY);
    playVid('B',si);showInfo(a,si);showDist(a,si);
  }}
}}

var aData={json.dumps(action_traces)};
var bData={json.dumps(backbone_traces)};
var aLay={json.dumps(action_layout)};
var bLay={json.dumps(backbone_layout)};
var cfg={{responsive:true,displayModeBar:true}};
Plotly.newPlot('plot_action',aData,aLay,cfg).then(function(ga){{
  return Plotly.newPlot('plot_backbone',bData,bLay,cfg).then(function(gb){{
    ga.on('plotly_click',function(ev){{handleClick(ev.points[0].customdata[3]);}});
    gb.on('plotly_click',function(ev){{handleClick(ev.points[0].customdata[3]);}});
    ga.on('plotly_legendclick',function(ev){{
      var ti=ev.curveNumber;if(ti>=N_DATA)return false;
      var nv=(traceVis[ti]===true)?'legendonly':true;traceVis[ti]=nv;
      Plotly.restyle('plot_action',{{visible:[nv]}},[ti]);
      Plotly.restyle('plot_backbone',{{visible:[nv]}},[ti]);
      return false;
    }});
    gb.on('plotly_legendclick',function(ev){{
      var ti=ev.curveNumber;if(ti>=N_DATA)return false;
      var nv=(traceVis[ti]===true)?'legendonly':true;traceVis[ti]=nv;
      Plotly.restyle('plot_action',{{visible:[nv]}},[ti]);
      Plotly.restyle('plot_backbone',{{visible:[nv]}},[ti]);
      return false;
    }});
  }});
}});
</script></body></html>"""

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [html] {out_path}  ({Path(out_path).stat().st_size/1e6:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "features.npz"

    # ── Keyframe registry ─────────────────────────────────────────────────────
    print("[0] Loading keyframe registry ...")
    registry_path  = Path(args.keyframe_registry_path).expanduser()
    keyframe_dict  = load_keyframe_registry(registry_path)
    valid_episodes = get_valid_episodes(registry_path)
    print(f"  valid episodes: {len(valid_episodes)}")

    # ── Task descriptions ─────────────────────────────────────────────────────
    desc_path = (Path(args.task_descriptions_path) if args.task_descriptions_path
                 else TASK_DESCRIPTIONS_PATHS[args.task_prompt_mode])
    task_name_to_desc = build_task_desc_map(desc_path, Path(args.dataset_root))
    print(f"  task descriptions: {len(task_name_to_desc)} tasks (mode={args.task_prompt_mode})")

    # ── Feature cache ─────────────────────────────────────────────────────────
    if cache_path.exists() and not args.skip_feature_cache:
        print(f"\n[cache] Loading from {cache_path}")
        action_feats, gt_actions, bb_f_list, bb_m_list, manifest = load_cache(cache_path)
    else:
        # ── Dataset ───────────────────────────────────────────────────────────
        print("\n[1] Loading dataset ...")
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.factory import resolve_delta_timestamps

        ds_meta = LeRobotDatasetMetadata(
            repo_id=args.dataset_repo_id, root=args.dataset_root)
        dataset_plain = load_dataset(
            args.dataset_repo_id, args.dataset_root, valid_episodes)

        # ── Model ─────────────────────────────────────────────────────────────
        print("\n[2] Loading model and preprocessor ...")
        sliced_stats = slice_dataset_stats(
            dataset_plain.meta.stats, args.action_dim, args.state_dim)
        policy, preprocessor = load_model_and_preprocessor(
            args.checkpoint_dir, sliced_stats, args.device)

        # Load dataset with delta_timestamps for action horizon
    # Load dataset with delta_timestamps for action only (camera = current frame only)
    # Faster than resolve_delta_timestamps which loads all cameras
        delta_ts = {"action": [i / ds_meta.fps for i in range(args.action_horizon)]}
        dataset = load_dataset(
        args.dataset_repo_id, args.dataset_root, valid_episodes, delta_ts)

        # ── Manifest ──────────────────────────────────────────────────────────
        print("\n[3] Building manifest ...")
        task_to_local = build_task_index_map(dataset)
        manifest = build_manifest(
            dataset, task_to_local, TASK_GROUPS, task_name_to_desc,
            args.samples_per_task, args.seed)
        if not manifest:
            print("[error] Empty manifest."); sys.exit(1)

        # ── Feature extraction ────────────────────────────────────────────────
        print(f"\n[4] Extracting features ({len(manifest)} samples) ...")
        action_feats, gt_actions, bb_f_list, bb_m_list = extract_features(
            policy, preprocessor, dataset, manifest,
            keyframe_dict, task_name_to_desc,
            args.action_dim, args.state_dim,
            args.batch_size, args.device)
        save_cache(cache_path, action_feats, gt_actions, bb_f_list, bb_m_list, manifest)
        del policy, preprocessor

    N = len(manifest)
    desc_counts = defaultdict(int)
    for m in manifest:
        desc_counts[m["task_name"]] += 1
    print(f"\n  N={N}  action_feats={action_feats.shape}  backbone batches={len(bb_f_list)}")
    print(f"  task distribution: {dict(sorted(desc_counts.items()))}")

    # ── Video clips ───────────────────────────────────────────────────────────
    clips_dirs = {}
    if not args.skip_clips and args.cam_keys:
        print(f"\n[5] Extracting video clips (cams={args.cam_keys}) ...")
        print("  [5a] Loading episode video metadata ...")
        ep_video_meta = load_episode_video_meta(Path(args.dataset_root))
        print(f"  [5a] Loaded metadata for {len(ep_video_meta)} episodes")
        W, H = map(int, args.clip_size.split("x"))
        clips_dirs = extract_video_clips(
            manifest, args.cam_keys, args.action_horizon,
            out_dir / "clips", fps=20,
            dataset_root=Path(args.dataset_root),
            output_size=(W, H),
            ep_video_meta=ep_video_meta,
            keyframe_dict=keyframe_dict)
        clips_dirs = {ck: d for ck, d in clips_dirs.items()
                      if d.exists() and any(d.glob("*.mp4"))}
        if clips_dirs:
            print(f"  [clips] available cams: {list(clips_dirs.keys())}")
        else:
            print("  [clips] WARNING: no clips were generated for any camera")

    # ── Backbone t-SNE (shared across action_agg runs) ─────────────────────────
    print(f"\n[6] Aggregating backbone ({args.backbone_agg}) + t-SNE ...")
    X_backbone = agg_backbone(bb_f_list, bb_m_list, args.backbone_agg)
    X_pca_bb, X2d_bb = run_tsne(X_backbone, args.perplexity, args.tsne_n_iter,
                                  args.seed, "backbone")
    if X2d_bb is None:
        print("[error] backbone t-SNE failed."); sys.exit(1)

    groups      = [m["group"]       for m in manifest]
    task_names  = [m["task_name"]   for m in manifest]
    descriptions = [m["description"] for m in manifest]

    # ── Per action_agg: t-SNE + HTML ──────────────────────────────────────────
    print(f"\n[7] Generating HTML (action_agg methods: {args.action_agg_list}) ...")
    for action_agg in args.action_agg_list:
        print(f"\n  ── action_agg={action_agg} ──")
        X_action = agg_action(action_feats, action_agg, gt_actions=gt_actions)
        print(f"  X_action={X_action.shape}")
        X_pca_act, X2d_act = run_tsne(X_action, args.perplexity, args.tsne_n_iter,
                                       args.seed, f"action_{action_agg}")
        if X2d_act is None:
            print(f"  [skip] t-SNE failed"); continue

        print("  computing distance matrices ...")
        dist_b64 = compute_dist_matrices(X_action, X_backbone, X_pca_act, X_pca_bb)

        tag = f"{action_agg}_p{args.perplexity}"
        plot_overview(X2d_act, X2d_bb, groups, task_names, descriptions,
                      args.perplexity, action_agg, args.backbone_agg,
                      out_dir / f"overview_{tag}.png")
        make_joint_html(
            X2d_act, X2d_bb, X_pca_act, X_pca_bb,
            manifest=manifest,
            action_agg=action_agg, backbone_agg=args.backbone_agg,
            perplexity=args.perplexity,
            out_path=out_dir / f"joint_{tag}.html",
            clips_dirs=clips_dirs,
            dist_b64=dist_b64,
        )

    print(f"\nDone. → {out_dir}")


if __name__ == "__main__":
    main()
