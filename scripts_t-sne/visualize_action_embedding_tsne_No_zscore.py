#!/usr/bin/env python3
"""
Visualize GROOT action expert representations using t-SNE and UMAP.

Three representation types compared:
  ① action_encoder output  — motion-only MLP, no backbone needed, fast
  ② DiT output (action slice) — VL-conditioned, requires backbone (--run_dit)
  ③ Temporal slice — time-based slicing of action_encoder output

Analysis modes:
  - Per-representation aggregation plots (mean / max / last / first_last)
  - Temporal slice 4-panel: early[0:10] / mid[20:30] / late[40:50] / full[0:50]
  - Timestep comparison: clean(t=999) / half(t=500) / noisy(t=0)
  - 3-way comparison: action_encoder vs DiT vs backbone (--run_dit only)

Usage (fast, no backbone):
    conda run -n groot python src/lerobot/scripts/visualize_action_embedding_tsne.py \\
        --checkpoint_dir /home/seonho/ws3/outputs/groot_guide/checkpoints/050000/pretrained_model \\
        --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \\
        --output_dir /home/seonho/ws3/outputs/action_emb_vis

Usage (with DiT output, includes backbone):
    conda run -n groot python src/lerobot/scripts/visualize_action_embedding_tsne.py \\
        --checkpoint_dir /home/seonho/ws3/outputs/groot_guide/checkpoints/050000/pretrained_model \\
        --dataset_root /home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \\
        --output_dir /home/seonho/ws3/outputs/action_emb_vis \\
        --run_dit --batch_size 16
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prompt import SCENE_TASK_PROMPT_GUIDE
from sampler import ProportionalTaskSampler

# ── Constants ──────────────────────────────────────────────────────────────────

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

POP_KEYS = [
    "observation.images.wrist_semantic",
    "observation.images.right_shoulder_semantic",
    "observation.images.left_shoulder_semantic",
    "observation.images.guide_semantic",
    "observation.images.left_shoulder",
    "observation.images.guide",
]

AGGREGATION_METHODS = ["mean", "max", "last", "first_last", "delta_total", "delta_mean", "flatten"]

TEMPORAL_SLICES_DEFAULT = {
    "early": (0,  10),
    "mid":   (20, 30),
    "late":  (40, 50),
    "full":  (0,  50),
}


def get_temporal_slices(T_action: int) -> dict:
    """Compute temporal slice boundaries based on actual T_action."""
    q = max(1, T_action // 4)
    return {
        "early": (0,        q),
        "mid":   (q,        2 * q),
        "late":  (2 * q,    3 * q),
        "full":  (0,        T_action),
    }

TIMESTEP_CONFIGS = {
    "clean_999": 999,
    "half_500":  500,
    "noisy_0":   0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="t-SNE/UMAP of GROOT action representations")
    p.add_argument("--checkpoint_dir", type=str,
                   default="/home/seonho/ws3/outputs/groot_guide/checkpoints/050000/pretrained_model")
    p.add_argument("--dataset_repo_id", type=str, default="paragon7060/INSIGHTfixposV3")
    p.add_argument("--dataset_root", type=str,
                   default="/home/seonho/workspace/data/paragon7060/INSIGHTfixposV3")
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  type=str,
                   default="/home/seonho/ws3/outputs/action_emb_vis")
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--run_dit",     action="store_true",
                   help="Also extract DiT output (requires backbone, slow ~10 min)")
    p.add_argument("--skip_cache",  action="store_true",
                   help="Re-extract features even if cache exists")
    p.add_argument("--tsne_perplexities", type=int, nargs="+", default=[10, 30, 50])
    p.add_argument("--umap_neighbors",    type=int, nargs="+", default=[5, 15, 30])
    p.add_argument("--tsne_n_iter",       type=int, default=1000)
    p.add_argument("--pop_keys", type=str, nargs="*", default=None,
                   help="Keys to pop from dataset. Defaults to POP_KEYS if not specified.")
    p.add_argument("--image_keys", type=str, nargs="*",
                   default=["observation.images.right_shoulder", "observation.images.wrist"],
                   help="Image keys to embed in interactive HTML (shown on point click).")
    p.add_argument("--image_size", type=int, default=192,
                   help="Max image size (px) for HTML thumbnails.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Dataset / model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(repo_id: str, root: str, pop_keys=None, action_horizon: int = 16):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    # Load metadata first to get fps, then build delta_timestamps so that
    # each sample returns the actual future action chunk (not just 1 repeated frame).
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    delta_timestamps = {"action": [i / meta.fps for i in range(action_horizon)]}
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav",
                             delta_timestamps=delta_timestamps)
    keys_to_pop = pop_keys if pop_keys is not None else POP_KEYS
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


_CL_ONLY_FIELDS = {
    # fields present in checkpoint config.json but NOT in current GrootCLConfig/GrootConfig
    "use_contrastive", "contrastive_latent_dim", "contrastive_vlm_input_dim",
    "contrastive_cnn_hidden_dim", "contrastive_proj_hidden_dim", "contrastive_triplet_margin",
    "contrastive_loss_weight", "contrastive_phase", "contrastive_backprop_backbone",
    "contrastive_fallback_to_in_batch", "groot_pretrained_path",
}


def _load_policy_with_vision_lora(ckpt: Path, device: str):
    """CL checkpoints (lora_target='vision'): apply backbone LoRA before loading weights."""
    import json
    import tempfile
    from safetensors.torch import load_model as safetensors_load_model

    with open(ckpt / "config.json") as f:
        config_data = json.load(f)

    lora_rank    = config_data.get("lora_rank", 0)
    lora_alpha   = config_data.get("lora_alpha", 16)
    lora_dropout = config_data.get("lora_dropout", 0.05)
    policy_type  = config_data.get("type", config_data.get("policy_type", "groot"))

    # Strip fields unknown to the current GrootCLConfig/GrootConfig class, then parse
    # directly with draccus to avoid stale-field DecodingErrors.
    _extra = _CL_ONLY_FIELDS | {"type"}   # "type" is handled by ChoiceRegistry; skip here
    sanitized = {k: v for k, v in config_data.items() if k not in _extra}
    sanitized["lora_rank"] = 0  # prevent _create_groot_model from applying LLM LoRA

    import draccus
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cfg = Path(tmpdir) / "config.json"
        with open(tmp_cfg, "w") as f:
            json.dump(sanitized, f)
        if policy_type == "groot_cl":
            from lerobot.policies.groot_cl.configuration_groot_cl import GrootCLConfig
            config = draccus.parse(GrootCLConfig, tmp_cfg, args=[])
        else:
            from lerobot.policies.groot.configuration_groot import GrootConfig
            config = draccus.parse(GrootConfig, tmp_cfg, args=[])

    config.lora_rank = 0  # prevent _create_groot_model from applying LLM LoRA

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
        print(f"[load] Applied vision LoRA: rank={lora_rank}, alpha={lora_alpha}")
    if lora_target in ("llm", "both"):
        eagle.wrap_llm_lora(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        print(f"[load] Applied LLM LoRA: rank={lora_rank}, alpha={lora_alpha}")

    # Load weights, skipping any shape-mismatched keys (e.g. contrastive head built
    # with a different contrastive_vlm_input_dim than the stripped config provides).
    from safetensors import safe_open
    ckpt_sd = {}
    with safe_open(str(ckpt / "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            ckpt_sd[key] = f.get_tensor(key)

    model_sd = policy.state_dict()
    skipped  = []
    filtered = {}
    for k, v in ckpt_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in model_sd:
            skipped.append(k)

    missing_keys = set(model_sd.keys()) - set(filtered.keys())
    policy.load_state_dict(filtered, strict=False)

    if skipped:
        print(f"[load] Skipped {len(skipped)} shape-mismatched keys (e.g. {skipped[:2]})")
    lora_keys = [k for k in missing_keys if "lora" not in k]
    if lora_keys:
        print(f"[load] Missing {len(lora_keys)} non-LoRA keys (e.g. {lora_keys[:2]})")
    return policy


def load_model_and_preprocessor(checkpoint_dir: str, dataset_stats: dict, device: str):
    from lerobot.policies.factory import make_pre_post_processors
    ckpt = Path(checkpoint_dir)
    print(f"[load] Loading policy from {ckpt} ...")

    import json
    with open(ckpt / "config.json") as f:
        config_data = json.load(f)
    lora_rank   = config_data.get("lora_rank", 0)
    lora_target = config_data.get("lora_target", "llm")

    if lora_rank > 0 and lora_target in ("vision", "both"):
        policy = _load_policy_with_vision_lora(ckpt, device)
    else:
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        policy = GrootPolicy.from_pretrained(str(ckpt))

    policy.eval()
    policy = policy.to(device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        dataset_stats=dataset_stats,
    )
    print("[load] Done.")
    return policy, preprocessor


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(
    policy, preprocessor, dataloader,
    num_samples: int, device: str,
    run_dit: bool = False,
    seed: int = 42,
):
    """
    Returns:
      action_feats  : np.ndarray  (N, T_action, D_enc)   — action_encoder output
      dit_feats     : np.ndarray  (N, T_action, D_dit) or None
      bb_feats_list : list[(B,T_vl,D)] or None  — backbone (variable T)
      bb_masks_list : list[(B,T_vl)]   or None
      task_labels   : list[str]
      frame_indices : list[int]
    """
    torch.manual_seed(seed)

    action_head    = policy._groot_model.action_head
    action_encoder = action_head.action_encoder
    action_horizon = action_head.action_horizon

    acenc_list  = []          # list of (B, T_action, D_enc) float32 cpu
    gt_act_list = []          # list of (B, T_action, action_dim) float32 cpu — raw GT actions
    dit_list    = []          # list of (B, T_action, D_dit) float32 cpu
    bb_f_list   = []          # list of (B, T_vl, D) float32 cpu
    bb_m_list   = []          # list of (B, T_vl) cpu
    task_labels   = []
    frame_indices = []

    # ── Hooks for DiT and backbone ──────────────────────────────────────────
    _dit_buf  = []
    _bb_buf   = []

    def _dit_hook(module, inp, out):
        # out: (B, state+future+action, hidden_size)
        _dit_buf.append(out[:, -action_horizon:, :].detach().float().cpu())

    def _bb_hook(module, inp, out):
        _bb_buf.append((
            out["backbone_features"].detach().float().cpu(),
            out["backbone_attention_mask"].detach().cpu(),
        ))

    if run_dit:
        _h_dit = action_head.model.register_forward_hook(_dit_hook)
        _h_bb  = policy._groot_model.backbone.register_forward_hook(_bb_hook)

    use_bf16 = getattr(policy.config, "use_bf16", True)
    dev_type = "cuda" if device != "cpu" else "cpu"

    total = 0
    for batch in dataloader:
        if total >= num_samples:
            break

        raw_task_keys = list(batch["task"])
        raw_indices   = batch["index"].tolist()
        batch["task"] = [PROMPT_MAP[t] for t in raw_task_keys]
        processed     = preprocessor(batch)

        # ── Prepare action inputs (without running backbone) ──────────────
        _, action_inputs = policy._groot_model.prepare_input(processed)
        gt_actions    = action_inputs.action         # (B, T_action, action_dim)
        embodiment_id = action_inputs.embodiment_id  # (B,)
        B = gt_actions.shape[0]

        # ── ① action_encoder: clean t=999 ─────────────────────────────────
        t_clean = torch.full((B,), 999, dtype=torch.long, device=gt_actions.device)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
            enc_out = action_encoder(gt_actions, t_clean, embodiment_id)  # (B, T_action, D)
        acenc_list.append(enc_out.float().cpu())
        gt_act_list.append(gt_actions.float().cpu())   # raw GT action (B, T_action, action_dim)

        # ── ② DiT output via full forward (monkey-patch sample_time) ──────
        if run_dit:
            _dit_buf.clear()
            _bb_buf.clear()

            # Patch: force t = 999/1000 ≈ 1.0 → noisy_trajectory ≈ actions
            _orig_st = action_head.sample_time
            action_head.sample_time = (
                lambda batch_size, device, dtype:
                torch.full((batch_size,), 999.0 / 1000.0, device=device, dtype=dtype)
            )
            try:
                policy.forward(processed)   # returns (loss, loss_dict), discarded
            finally:
                action_head.sample_time = _orig_st

            if _dit_buf:
                dit_list.append(_dit_buf[0])
            if _bb_buf:
                bf, bm = _bb_buf[0]
                bb_f_list.append(bf)
                bb_m_list.append(bm)

        task_labels.extend(raw_task_keys)
        frame_indices.extend(raw_indices)
        total += B
        print(f"  collected {min(total, num_samples)}/{num_samples} ...", end="\r")

    print()

    if run_dit:
        _h_dit.remove()
        _h_bb.remove()

    # ── Trim to num_samples ─────────────────────────────────────────────────
    if total > num_samples:
        excess = total - num_samples
        keep   = acenc_list[-1].shape[0] - excess
        acenc_list[-1]  = acenc_list[-1][:keep]
        gt_act_list[-1] = gt_act_list[-1][:keep]
        if dit_list:
            dit_list[-1] = dit_list[-1][:keep]
        if bb_f_list:
            bb_f_list[-1] = bb_f_list[-1][:keep]
            bb_m_list[-1] = bb_m_list[-1][:keep]
        task_labels   = task_labels[:num_samples]
        frame_indices = frame_indices[:num_samples]

    action_feats = np.concatenate([x.numpy() for x in acenc_list], axis=0)  # (N, T, D_enc)
    gt_actions   = np.concatenate([x.numpy() for x in gt_act_list], axis=0) # (N, T, action_dim)
    dit_feats    = (np.concatenate([x.numpy() for x in dit_list], axis=0)
                    if dit_list else None)

    D = action_feats.shape[-1]
    D_dit = dit_feats.shape[-1] if dit_feats is not None else 0
    print(f"[extract] N={action_feats.shape[0]}, T_action={action_feats.shape[1]}, "
          f"D_enc={D}, action_dim={gt_actions.shape[-1]}"
          + (f", D_dit={D_dit}" if run_dit else ""))
    return (action_feats, gt_actions, dit_feats,
            bb_f_list if bb_f_list else None,
            bb_m_list if bb_m_list else None,
            task_labels, frame_indices)


@torch.no_grad()
def extract_features_multi_timestep(
    policy, preprocessor, dataloader,
    num_samples: int, device: str,
    timestep_configs: dict,   # {"clean_999": 999, "half_500": 500, "noisy_0": 0}
    seed: int = 42,
):
    """
    Extract action_encoder features for multiple fixed timestep values.
    Returns dict: {name: np.ndarray (N, T_action, D)}
    """
    torch.manual_seed(seed)

    action_head    = policy._groot_model.action_head
    action_encoder = action_head.action_encoder

    use_bf16 = getattr(policy.config, "use_bf16", True)
    dev_type = "cuda" if device != "cpu" else "cpu"

    buffers     = {name: [] for name in timestep_configs}
    task_labels = []

    total = 0
    for batch in dataloader:
        if total >= num_samples:
            break

        raw_task_keys = list(batch["task"])
        batch["task"] = [PROMPT_MAP[t] for t in raw_task_keys]
        processed     = preprocessor(batch)

        _, action_inputs = policy._groot_model.prepare_input(processed)
        gt_actions    = action_inputs.action
        embodiment_id = action_inputs.embodiment_id
        B = gt_actions.shape[0]

        for name, t_val in timestep_configs.items():
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=gt_actions.device)
            if t_val == 0:
                noise = torch.randn_like(gt_actions)
                inp   = noise
            else:
                t_cont = t_val / 1000.0
                noise  = torch.randn_like(gt_actions)
                inp    = (1 - t_cont) * noise + t_cont * gt_actions
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
                out = action_encoder(inp, t_tensor, embodiment_id)
            buffers[name].append(out.float().cpu())

        task_labels.extend(raw_task_keys)
        total += B
        print(f"  timestep comparison: {min(total, num_samples)}/{num_samples} ...", end="\r")

    print()

    result = {}
    for name, lst in buffers.items():
        arr = np.concatenate([x.numpy() for x in lst], axis=0)
        result[name] = arr[:num_samples]
    task_labels = task_labels[:num_samples]
    return result, task_labels


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(feats: np.ndarray, method: str) -> np.ndarray:
    """(N, T, D) → (N, D)
    Delta methods preserve sign: negative = decreasing, positive = increasing.
      delta_total : feat[:,-1] - feat[:,0]          — net change over full horizon
      delta_mean  : mean of step-by-step differences — average rate of change
    """
    t = torch.from_numpy(feats)
    if method == "mean":
        out = t.mean(dim=1)
    elif method == "max":
        out = t.max(dim=1).values
    elif method == "last":
        out = t[:, -1, :]
    elif method == "first_last":
        out = (t[:, 0, :] + t[:, -1, :]) / 2.0
    elif method == "delta_total":
        # Net change: last step minus first step.  Sign preserved.
        out = t[:, -1, :] - t[:, 0, :]
    elif method == "delta_mean":
        # Average per-step change.  Sign preserved.
        out = (t[:, 1:, :] - t[:, :-1, :]).mean(dim=1)
    elif method == "flatten":
        N, T, D = t.shape
        return t.reshape(N, T * D).numpy()
    else:
        raise ValueError(f"Unknown aggregation: {method}")
    return out.numpy()


RAW_AGGREGATION_METHODS = ["raw_delta_mean", "raw_delta_total", "raw_flatten"]


def agg_raw_action(gt_actions: np.ndarray, method: str) -> np.ndarray:
    """gt_actions (N, T, action_dim) → (N, D*)
      raw_delta_mean  : mean per-step velocity in joint space → (N, action_dim)
      raw_delta_total : net displacement in joint space       → (N, action_dim)
      raw_flatten     : full trajectory flattened             → (N, T*action_dim)
    """
    t = torch.from_numpy(gt_actions.astype(np.float32))
    N, T, D = t.shape
    if method == "raw_delta_mean":
        out = (t[:, 1:, :] - t[:, :-1, :]).mean(dim=1)
    elif method == "raw_delta_total":
        out = t[:, -1, :] - t[:, 0, :]
    elif method == "raw_flatten":
        out = t.reshape(N, T * D)
    else:
        raise ValueError(f"Unknown raw method: {method}")
    return out.numpy()


def visualise_raw_action(
    gt_actions: np.ndarray,
    task_labels: list,
    out_dir: Path,
    args,
    episode_progress: np.ndarray = None,
    frame_indices: list = None,
    images_dict: dict = None,
):
    """raw GT action space t-SNE/UMAP 시각화. out_dir/raw_action/<method>/ 에 저장."""
    umap_ok = True
    try:
        import umap  # noqa: F401
    except ImportError:
        umap_ok = False
        print("  [warn] umap not installed, skipping UMAP plots")

    for raw_agg in RAW_AGGREGATION_METHODS:
        print(f"\n  ── raw_agg={raw_agg} ──")
        X_raw = agg_raw_action(gt_actions, raw_agg)
        X = sanitize(X_raw, f"raw_action/{raw_agg}")
        if X is None:
            continue

        agg_dir = out_dir / raw_agg
        agg_dir.mkdir(parents=True, exist_ok=True)

        for perp in args.tsne_perplexities:
            print(f"    [t-SNE] perp={perp} ...")
            try:
                X2d = run_tsne(X, perplexity=perp, n_iter=args.tsne_n_iter,
                               seed=args.seed, label=f"raw_action/{raw_agg}/tsne_p{perp}")
            except Exception as e:
                print(f"    [error] t-SNE failed: {e}"); continue
            if X2d is None:
                continue
            tag   = f"tsne_p{perp}"
            title = f"raw_action  |  agg={raw_agg}  |  t-SNE perp={perp}"
            plot_combined(X2d, task_labels, title, agg_dir / f"{tag}_combined.png")
            plot_by_task(X2d, task_labels, title + " (task)", agg_dir / f"{tag}_task.png")
            plot_by_category(X2d, task_labels, title + " (cat)", agg_dir / f"{tag}_category.png")
            if episode_progress is not None:
                make_progress_html(X2d, task_labels, episode_progress,
                                   frame_indices or list(range(len(task_labels))),
                                   title, agg_dir / f"{tag}_progress.html",
                                   images_dict=images_dict)

        if umap_ok:
            for nb in args.umap_neighbors:
                print(f"    [UMAP] n_neighbors={nb} ...")
                try:
                    X2d = run_umap(X, n_neighbors=nb, seed=args.seed,
                                   label=f"raw_action/{raw_agg}/umap_nb{nb}")
                except Exception as e:
                    print(f"    [error] UMAP failed: {e}"); continue
                if X2d is None:
                    continue
                tag   = f"umap_nb{nb}"
                title = f"raw_action  |  agg={raw_agg}  |  UMAP n_neighbors={nb}"
                plot_combined(X2d, task_labels, title, agg_dir / f"{tag}_combined.png")
                plot_by_task(X2d, task_labels, title + " (task)", agg_dir / f"{tag}_task.png")
                plot_by_category(X2d, task_labels, title + " (cat)", agg_dir / f"{tag}_category.png")
                if episode_progress is not None:
                    make_progress_html(X2d, task_labels, episode_progress,
                                       frame_indices or list(range(len(task_labels))),
                                       title, agg_dir / f"{tag}_progress.html",
                                       images_dict=images_dict)


def compute_episode_progress(frame_indices, dataset) -> np.ndarray:
    """
    For each sampled frame_index (global dataset index), return the fraction
    through its episode: 0.0 = first frame, 1.0 = last frame.

    Uses hf_dataset columns 'frame_index' (position within episode) and
    'episode_index' to compute progress without needing episode_data_index.
    """
    import numpy as np

    hf = dataset.hf_dataset
    # Build per-episode length array: ep_lengths[i] = number of frames in episode i
    ep_indices_all = np.array([x.item() if hasattr(x, "item") else int(x)
                                for x in hf["episode_index"]], dtype=np.int64)
    ep_lengths = np.bincount(ep_indices_all)   # shape: (num_episodes,)

    fi_arr = np.array(frame_indices, dtype=np.int64)  # global frame indices

    # For each global index, look up which episode it is in and its within-episode position
    frame_pos_arr = np.array([x.item() if hasattr(x, "item") else int(x)
                               for x in hf["frame_index"]], dtype=np.int64)
    ep_of_frame   = ep_indices_all   # ep_indices_all[global_idx] = episode index

    ep_of_sample  = ep_of_frame[fi_arr]      # episode index for each sampled frame
    pos_of_sample = frame_pos_arr[fi_arr]    # within-episode position
    ep_len_of_sample = ep_lengths[ep_of_sample]

    progress = np.where(ep_len_of_sample > 1,
                        pos_of_sample / (ep_len_of_sample - 1).clip(min=1),
                        0.0)
    return progress.clip(0.0, 1.0)


def extract_images(dataset, frame_indices: list, image_keys: list,
                   max_size: int = 192) -> dict:
    """
    For each global frame index in frame_indices, load the image tensors from
    the dataset and return as uint8 HWC numpy arrays.

    Returns:
        dict mapping short key (e.g. "right_shoulder") → np.ndarray (N, H, W, 3) uint8
        Only keys present in dataset.meta.features are included.
    """
    from PIL import Image as PILImage

    valid_keys = [k for k in image_keys if k in dataset.meta.features]
    if not valid_keys:
        print("  [images] No valid image keys found in dataset, skipping.")
        return {}

    N = len(frame_indices)
    print(f"  [images] Extracting {N} frames × {len(valid_keys)} cameras ...")
    buffers = {k: [] for k in valid_keys}

    for i, global_idx in enumerate(frame_indices):
        sample = dataset[global_idx]
        for k in valid_keys:
            img = sample[k]
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)  # CHW → HWC
                img = img.numpy()
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            if img.ndim == 2:                      # grayscale → RGB
                img = np.stack([img] * 3, axis=-1)
            if max(img.shape[:2]) > max_size:
                pil = PILImage.fromarray(img)
                pil.thumbnail((max_size, max_size), PILImage.LANCZOS)
                img = np.array(pil)
            buffers[k].append(img)
        if (i + 1) % 64 == 0 or (i + 1) == N:
            print(f"  [images] {i+1}/{N} ...", end="\r")
    print()

    result = {}
    for k, imgs in buffers.items():
        short = k.split(".")[-1]   # "observation.images.right_shoulder" → "right_shoulder"
        # Stack: images may vary slightly in HW after thumbnail, so use object array
        arr = np.empty(len(imgs), dtype=object)
        for j, im in enumerate(imgs):
            arr[j] = im
        result[short] = arr
    return result


def _imgs_to_b64(img_arr: np.ndarray, quality: int = 70) -> list:
    """Convert object array of uint8 HWC images → list of base64 JPEG data-URI strings."""
    import base64
    import io
    from PIL import Image as PILImage

    out = []
    for img in img_arr:
        pil = PILImage.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        out.append(f"data:image/jpeg;base64,{b64}")
    return out


def make_progress_html(
    X_2d: np.ndarray,           # (N, 2)
    task_labels: list,
    episode_progress: np.ndarray,  # (N,) in [0, 1]
    frame_indices: list,
    title: str,
    out_path: Path,
    images_dict: dict = None,   # {short_key: object array of uint8 HWC images}
):
    """
    Interactive Plotly HTML with two subplots side by side:
      Left  — colored by episode progress (viridis, continuous)
      Right — colored by task_id (categorical)
    Hover on every point shows: task, category, progress %, frame_idx.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print(f"  [skip] plotly not installed, skipping HTML: {out_path.name}")
        return

    import json as _json

    task_arr     = np.array(task_labels)
    task_ids     = sorted(set(task_labels))
    progress_pct = (episode_progress * 100).round(1)
    cats         = np.array([TASK_TO_CATEGORY.get(t, "?") for t in task_labels])
    sample_idxs  = list(range(len(task_labels)))   # 0..N-1 for image lookup

    # ── build tab20 colour map for tasks ──────────────────────────────────────
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(task_ids), 1))
    task_color = {t: f"rgba({int(cmap(i)[0]*255)},{int(cmap(i)[1]*255)},{int(cmap(i)[2]*255)},0.85)"
                  for i, t in enumerate(task_ids)}

    # customdata layout: [task, category, progress_pct, frame_idx, sample_idx]
    def _cd_full(indices):
        return list(zip(
            task_arr[indices].tolist(), cats[indices].tolist(),
            progress_pct[indices].tolist(),
            [frame_indices[i] for i in indices],
            [sample_idxs[i]   for i in indices],
        ))

    hover_tmpl = (
        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
        "progress: %{customdata[2]:.1f}%<br>"
        "frame: %{customdata[3]}<extra></extra>"
    )

    # ── subplot layout ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Colored by episode progress", "Colored by task"],
        horizontal_spacing=0.08,
    )

    # ── Left: continuous progress colour ──────────────────────────────────────
    all_idx = list(range(len(task_labels)))
    fig.add_trace(
        go.Scatter(
            x=X_2d[:, 0].tolist(), y=X_2d[:, 1].tolist(),
            mode="markers",
            marker=dict(
                color=episode_progress.tolist(),
                colorscale="Viridis",
                size=6,
                opacity=0.85,
                showscale=True,
                colorbar=dict(title="progress", x=0.44, len=0.9),
                cmin=0.0, cmax=1.0,
            ),
            customdata=_cd_full(all_idx),
            hovertemplate=hover_tmpl,
            showlegend=False,
            name="progress",
        ),
        row=1, col=1,
    )

    # ── Right: categorical task colour ────────────────────────────────────────
    for tid in task_ids:
        mask = task_arr == tid
        idx  = np.where(mask)[0].tolist()
        fig.add_trace(
            go.Scatter(
                x=X_2d[idx, 0].tolist(), y=X_2d[idx, 1].tolist(),
                mode="markers",
                marker=dict(color=task_color[tid], size=6, opacity=0.85),
                name=tid,
                customdata=_cd_full(idx),
                hovertemplate=hover_tmpl,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title_text=title,
        title_font_size=14,
        height=600,
        width=1400,
        legend=dict(x=1.01, y=1, xanchor="left"),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")

    # ── If no images: simple HTML ─────────────────────────────────────────────
    if not images_dict:
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"  [html] {out_path.name}")
        return

    # ── With images: custom HTML with click-to-show panel ────────────────────
    print(f"  [html] encoding {len(images_dict)} × {len(task_labels)} images ...")
    b64_data = {}
    for short_key, img_arr in images_dict.items():
        b64_data[short_key] = _imgs_to_b64(img_arr)

    fig_json = fig.to_json()
    img_panel_html = "\n".join(
        f'<div class="img-box"><p class="img-label">{k}</p>'
        f'<img id="img_{k}" src="" width="100%"></div>'
        for k in b64_data
    )
    img_keys_js = _json.dumps(list(b64_data.keys()))
    img_data_js = _json.dumps(b64_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 8px; background: #fafafa; }}
  #plot-wrap {{ width: 100%; }}
  #image-panel {{
    display: flex; align-items: flex-start; gap: 16px;
    padding: 12px 16px; margin-top: 8px;
    background: #fff; border: 1px solid #ddd; border-radius: 8px;
  }}
  .img-box {{ text-align: center; flex: 1; min-width: 0; }}
  .img-label {{ margin: 0 0 4px; font-size: 12px; font-weight: bold; color: #555; }}
  .img-box img {{
    width: 100%; max-width: 320px; height: auto;
    border: 2px solid #ccc; border-radius: 4px;
    background: #eee;
  }}
  #img-info {{
    flex: 1.2; font-size: 13px; color: #444; padding: 4px 8px;
    white-space: pre-line;
  }}
  #img-hint {{ color: #aaa; font-style: italic; }}
</style>
</head>
<body>
<div id="plot-wrap"><div id="plotDiv"></div></div>
<div id="image-panel">
{img_panel_html}
  <div id="img-info"><span id="img-hint">← click a point to see images</span></div>
</div>

<script>
var figData = {fig_json};
var imgKeys = {img_keys_js};
var imgData = {img_data_js};

Plotly.newPlot('plotDiv', figData.data, figData.layout, {{responsive: true}})
.then(function(gd) {{
  gd.on('plotly_click', function(eventData) {{
    var pt = eventData.points[0];
    var cd = pt.customdata;
    var sampleIdx = cd[4];

    imgKeys.forEach(function(key) {{
      var el = document.getElementById('img_' + key);
      if (el && imgData[key] && sampleIdx < imgData[key].length) {{
        el.src = imgData[key][sampleIdx];
        el.style.border = '2px solid #4a90d9';
      }}
    }});

    var info = document.getElementById('img-info');
    info.innerHTML =
      '<b>' + cd[0] + '</b> (' + cd[1] + ')<br>' +
      'progress: ' + parseFloat(cd[2]).toFixed(1) + '%<br>' +
      'frame idx: ' + cd[3] + '<br>' +
      'sample: ' + sampleIdx;
  }});
}});
</script>
</body>
</html>"""

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [html] {out_path.name} (with images)")


def aggregate_backbone(feats_list, masks_list, method: str = "mask_mean") -> np.ndarray:
    """Variable-T backbone features → (N, D) using mask_mean."""
    parts = []
    for f, m in zip(feats_list, masks_list):
        # f: (B, T_vl, D), m: (B, T_vl)
        m_f = m.unsqueeze(-1).float()
        agg = (f * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)
        parts.append(agg)
    return torch.cat(parts, dim=0).numpy()


def aggregate_temporal_slice(feats: np.ndarray, start: int, end: int) -> np.ndarray:
    """(N, T, D) → (N, D) using mean over [start:end]."""
    return feats[:, start:end, :].mean(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Dimensionality reduction (reused from backbone script)
# ══════════════════════════════════════════════════════════════════════════════

def sanitize(X: np.ndarray, label: str) -> np.ndarray | None:
    bad_rows = np.any(~np.isfinite(X), axis=1)
    n_bad = bad_rows.sum()
    if n_bad > 0:
        print(f"    [warn] {n_bad}/{len(X)} rows have NaN/inf in '{label}', replacing with 0")
        if n_bad > len(X) // 2:
            print(f"    [skip] too many bad rows, skipping")
            return None
        X = X.copy()
        X[bad_rows] = 0.0
    return X


def pca_preprocess(X: np.ndarray, n_components: int = 50) -> np.ndarray | None:
    from sklearn.decomposition import PCA
    var = X.var(axis=0)
    X_clean = X[:, var > 1e-10]
    if X_clean.shape[1] == 0:
        return None
    n_comp = min(n_components, X_clean.shape[0] - 1, X_clean.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    X_pca = pca.fit_transform(X_clean)
    return np.nan_to_num(X_pca, nan=0.0, posinf=0.0, neginf=0.0)


def _add_jitter_if_needed(X_pca: np.ndarray, label: str) -> np.ndarray | None:
    stds = X_pca.std(axis=0)
    if stds[0] < 1e-12:
        rng   = np.random.RandomState(42)
        X_pca = X_pca + rng.randn(*X_pca.shape) * 1e-6
        if X_pca.std(axis=0)[0] < 1e-12:
            print(f"    [skip] {label}: degenerate after jitter")
            return None
    return X_pca


def run_tsne(X: np.ndarray, perplexity: int, n_iter: int, seed: int,
             label: str = "") -> np.ndarray | None:
    from sklearn.manifold import TSNE
    import sklearn
    X_pca = pca_preprocess(X)
    if X_pca is None:
        print(f"    [skip] {label}: all features constant")
        return None
    X_pca = _add_jitter_if_needed(X_pca, label)
    if X_pca is None:
        return None
    max_perp = (X_pca.shape[0] - 1) / 3.0
    perplexity = min(perplexity, max(5.0, max_perp - 1))
    kw = dict(n_components=2, perplexity=perplexity, random_state=seed, verbose=0)
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5):
        kw["max_iter"] = n_iter
    else:
        kw["n_iter"] = n_iter
    return TSNE(**kw).fit_transform(X_pca)


def run_umap(X: np.ndarray, n_neighbors: int, seed: int,
             label: str = "") -> np.ndarray | None:
    import umap as umap_lib
    X_pca = pca_preprocess(X)
    if X_pca is None:
        print(f"    [skip] {label}: all features constant")
        return None
    X_pca = _add_jitter_if_needed(X_pca, label)
    if X_pca is None:
        return None
    return umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                         min_dist=0.1, random_state=seed, verbose=False).fit_transform(X_pca)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _task_cmap(task_ids):
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(task_ids), 1))
    return {t: cmap(i) for i, t in enumerate(task_ids)}


def _scatter_by_task(ax, X_2d, task_labels, alpha=0.65, s=15):
    task_ids  = sorted(set(task_labels))
    cmap      = _task_cmap(task_ids)
    labels_a  = np.array(task_labels)
    for tid in task_ids:
        mask = labels_a == tid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap[tid]], label=tid, alpha=alpha, s=s, linewidths=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)


def _scatter_by_category(ax, X_2d, task_labels, alpha=0.65, s=15):
    cats = np.array([TASK_TO_CATEGORY[t] for t in task_labels])
    for cat, color in CATEGORY_COLORS.items():
        mask = cats == cat
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=cat, alpha=alpha, s=s, linewidths=0)
    ax.legend(fontsize=10, markerscale=2)


def plot_combined(X_2d, task_labels, title, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    _scatter_by_task(ax, X_2d, task_labels)
    ax.set_title("by task_id"); ax.grid(True, alpha=0.2)
    ax = axes[1]
    _scatter_by_category(ax, X_2d, task_labels)
    ax.set_title("by category"); ax.grid(True, alpha=0.2)
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_by_task(X_2d, task_labels, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_by_task(ax, X_2d, task_labels)
    ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_by_category(X_2d, task_labels, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_by_category(ax, X_2d, task_labels)
    ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_temporal_slices_panel(
    feats: np.ndarray,        # (N, T_action, D)
    task_labels: list,
    reduction_fn,             # callable: X (N,D) → X_2d (N,2) or None
    title_prefix: str,
    out_path: Path,
    slices: dict = None,     # if None, computed from feats.shape[1]
):
    """4-panel plot: one subplot per temporal slice (early / mid / late / full)."""
    if slices is None:
        slices = get_temporal_slices(feats.shape[1])
    n_slices = len(slices)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes_flat = axes.flatten()

    for ax_i, (name, (s, e)) in enumerate(slices.items()):
        ax  = axes_flat[ax_i]
        X   = aggregate_temporal_slice(feats, s, e)   # (N, D)
        X   = sanitize(X, f"{name}")
        if X is None:
            ax.set_title(f"[{name}] SKIPPED"); continue
        X2d = reduction_fn(X)
        if X2d is None:
            ax.set_title(f"[{name}] FAILED"); continue
        cats = np.array([TASK_TO_CATEGORY[t] for t in task_labels])
        for cat, color in CATEGORY_COLORS.items():
            mask = cats == cat
            ax.scatter(X2d[mask, 0], X2d[mask, 1],
                       c=color, label=cat, alpha=0.65, s=15, linewidths=0)
        ax.legend(fontsize=9, markerscale=2)
        ax.set_title(f"temporal slice [{s}:{e}]  ({name})", fontsize=11)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{title_prefix}  |  temporal slices  (by category)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_temporal_slices_task_panel(
    feats: np.ndarray,
    task_labels: list,
    reduction_fn,
    title_prefix: str,
    out_path: Path,
    slices: dict = None,
):
    """4-panel plot of temporal slices, colored by task_id."""
    if slices is None:
        slices = get_temporal_slices(feats.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes_flat = axes.flatten()

    for ax_i, (name, (s, e)) in enumerate(slices.items()):
        ax  = axes_flat[ax_i]
        X   = aggregate_temporal_slice(feats, s, e)
        X   = sanitize(X, name)
        if X is None:
            ax.set_title(f"[{name}] SKIPPED"); continue
        X2d = reduction_fn(X)
        if X2d is None:
            ax.set_title(f"[{name}] FAILED"); continue
        _scatter_by_task(ax, X2d, task_labels)
        ax.set_title(f"temporal slice [{s}:{e}]  ({name})", fontsize=11)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{title_prefix}  |  temporal slices  (by task_id)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_timestep_comparison(
    timestep_feats: dict,    # {name: (N, T, D)}
    task_labels: list,
    reduction_fn,
    agg_method: str,
    out_path: Path,
):
    """3-panel: clean_999 / half_500 / noisy_0, by category."""
    n = len(timestep_feats)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (name, feats) in zip(axes, timestep_feats.items()):
        X   = aggregate(feats, agg_method)
        X   = sanitize(X, name)
        if X is None:
            ax.set_title(f"[{name}] SKIPPED"); continue
        X2d = reduction_fn(X)
        if X2d is None:
            ax.set_title(f"[{name}] FAILED"); continue
        cats = np.array([TASK_TO_CATEGORY[t] for t in task_labels])
        for cat, color in CATEGORY_COLORS.items():
            mask = cats == cat
            ax.scatter(X2d[mask, 0], X2d[mask, 1],
                       c=color, label=cat, alpha=0.65, s=15, linewidths=0)
        ax.legend(fontsize=10, markerscale=2)
        t_val = TIMESTEP_CONFIGS[name]
        t_pct = t_val / 10
        ax.set_title(f"t={t_val}  (GT ratio={t_pct:.0f}%)", fontsize=12)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Timestep comparison  |  action_encoder  |  agg={agg_method}  (by category)",
        fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_3way_comparison(
    acenc_X: np.ndarray,   # (N, D)
    dit_X:   np.ndarray,   # (N, D)
    bb_X:    np.ndarray,   # (N, D)
    task_labels: list,
    reduction_fn,
    title: str,
    out_path: Path,
):
    """3-panel: action_encoder vs DiT vs backbone, by category and task."""
    datasets = [
        ("① action_encoder", acenc_X),
        ("② DiT output",     dit_X),
        ("③ backbone",        bb_X),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(27, 14))
    cats = np.array([TASK_TO_CATEGORY[t] for t in task_labels])

    for col, (label, X) in enumerate(datasets):
        X = sanitize(X, label)
        if X is None:
            for row in range(2):
                axes[row][col].set_title(f"[{label}] SKIPPED")
            continue
        X2d = reduction_fn(X)
        if X2d is None:
            for row in range(2):
                axes[row][col].set_title(f"[{label}] FAILED")
            continue

        # Row 0: by category
        ax = axes[0][col]
        for cat, color in CATEGORY_COLORS.items():
            mask = cats == cat
            ax.scatter(X2d[mask, 0], X2d[mask, 1],
                       c=color, label=cat, alpha=0.65, s=15, linewidths=0)
        ax.legend(fontsize=9, markerscale=2)
        ax.set_title(f"{label}\n(by category)", fontsize=11)
        ax.grid(True, alpha=0.2)

        # Row 1: by task_id
        ax = axes[1][col]
        _scatter_by_task(ax, X2d, task_labels)
        ax.set_title(f"{label}\n(by task_id)", fontsize=11)
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-representation visualisation loop
# ══════════════════════════════════════════════════════════════════════════════

def visualise_representation(
    feats: np.ndarray,          # (N, T, D)
    task_labels: list,
    repr_name: str,
    out_dir: Path,
    args,
    agg_methods=AGGREGATION_METHODS,
    episode_progress: np.ndarray = None,  # (N,) in [0,1], optional
    frame_indices: list = None,
    images_dict: dict = None,   # {short_key: object array of uint8 images}
):
    """Standard per-aggregation-method t-SNE/UMAP plots for one representation.
    If episode_progress is provided, also generates interactive HTML.
    If images_dict is also provided, clicking a point shows that frame's images.
    """
    umap_ok = True
    try:
        import umap  # noqa: F401
    except ImportError:
        umap_ok = False
        print("  [warn] umap not installed, skipping UMAP plots")

    for agg in agg_methods:
        print(f"\n  ── agg={agg} ──")
        X_raw = aggregate(feats, agg)
        X = sanitize(X_raw, f"{repr_name}/{agg}")
        if X is None:
            continue

        agg_dir = out_dir / agg
        agg_dir.mkdir(exist_ok=True)

        for perp in args.tsne_perplexities:
            print(f"    [t-SNE] perp={perp} ...")
            try:
                X2d = run_tsne(X, perplexity=perp, n_iter=args.tsne_n_iter,
                               seed=args.seed, label=f"{repr_name}/{agg}/tsne_p{perp}")
            except Exception as e:
                print(f"    [error] t-SNE failed: {e}"); continue
            if X2d is None:
                continue
            tag   = f"tsne_p{perp}"
            title = f"{repr_name}  |  agg={agg}  |  t-SNE perp={perp}"
            plot_combined(X2d, task_labels, title, agg_dir / f"{tag}_combined.png")
            plot_by_task(X2d, task_labels, title + " (task)", agg_dir / f"{tag}_task.png")
            plot_by_category(X2d, task_labels, title + " (cat)", agg_dir / f"{tag}_category.png")
            if episode_progress is not None:
                make_progress_html(X2d, task_labels, episode_progress,
                                   frame_indices or list(range(len(task_labels))),
                                   title, agg_dir / f"{tag}_progress.html",
                                   images_dict=images_dict)

        if umap_ok:
            for nb in args.umap_neighbors:
                print(f"    [UMAP] n_neighbors={nb} ...")
                try:
                    X2d = run_umap(X, n_neighbors=nb, seed=args.seed,
                                   label=f"{repr_name}/{agg}/umap_nb{nb}")
                except Exception as e:
                    print(f"    [error] UMAP failed: {e}"); continue
                if X2d is None:
                    continue
                tag   = f"umap_nb{nb}"
                title = f"{repr_name}  |  agg={agg}  |  UMAP n_neighbors={nb}"
                plot_combined(X2d, task_labels, title, agg_dir / f"{tag}_combined.png")
                plot_by_task(X2d, task_labels, title + " (task)", agg_dir / f"{tag}_task.png")
                plot_by_category(X2d, task_labels, title + " (cat)", agg_dir / f"{tag}_category.png")
                if episode_progress is not None:
                    make_progress_html(X2d, task_labels, episode_progress,
                                       frame_indices or list(range(len(task_labels))),
                                       title, agg_dir / f"{tag}_progress.html",
                                       images_dict=images_dict)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_enc = out_dir / "action_encoder_cache.npz"
    cache_dit = out_dir / "dit_features_cache.npz"
    cache_ts  = out_dir / "timestep_cache.npz"
    cache_img = out_dir / "image_cache.npz"

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("[1/6] Loading dataset ...")
    # action_horizon=16: GROOT architecture hard cap; loads actual future 16 frames
    # so dataset["action"] shape is (16, action_dim) not (action_dim,).
    dataset = load_dataset(args.dataset_repo_id, args.dataset_root,
                           pop_keys=args.pop_keys, action_horizon=16)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[2/6] Loading model and preprocessor ...")
    policy, preprocessor = load_model_and_preprocessor(
        args.checkpoint_dir, dataset.meta.stats, args.device)

    # ── DataLoader ────────────────────────────────────────────────────────────
    print("[3/6] Building dataloader ...")
    corrupted_path = Path(args.dataset_root) / "all_excluded_indices_seed2.json"
    corrupted = set()
    if corrupted_path.exists():
        with open(corrupted_path) as f:
            corrupted = set(json.load(f))

    valid_episodes = [i for i in range(dataset.num_episodes) if i not in corrupted]
    sampler = ProportionalTaskSampler(
        dataset=dataset,
        valid_episode_indices=valid_episodes,
        target_proportions={"cabinet": 1.0, "door": 1.0, "bottle": 1.0},
        epoch_size=args.num_samples * 4,
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
    print("[4/6] Extracting features ...")

    # ① action_encoder + (② DiT) from same dataloader pass
    need_main_extract = (
        args.skip_cache
        or not cache_enc.exists()
        or (args.run_dit and not cache_dit.exists())
    )

    if need_main_extract:
        action_feats, gt_actions, dit_feats, bb_f, bb_m, task_labels, frame_indices = extract_features(
            policy, preprocessor, dataloader,
            num_samples=args.num_samples,
            device=args.device,
            run_dit=args.run_dit,
            seed=args.seed,
        )
        episode_progress = compute_episode_progress(frame_indices, dataset)
        # Save action_encoder cache (also saves raw GT actions for direction analysis)
        np.savez_compressed(
            cache_enc,
            action_feats=action_feats,
            gt_actions=gt_actions,
            task_labels=np.array(task_labels),
            frame_indices=np.array(frame_indices, dtype=np.int64),
            episode_progress=episode_progress,
        )
        print(f"  [cache] action_encoder → {cache_enc}")
        # Save DiT cache
        if dit_feats is not None:
            bb_f_np = np.empty(len(bb_f), dtype=object)
            bb_m_np = np.empty(len(bb_m), dtype=object)
            for i, (f, m) in enumerate(zip(bb_f, bb_m)):
                bb_f_np[i] = f.numpy() if isinstance(f, torch.Tensor) else f
                bb_m_np[i] = m.numpy() if isinstance(m, torch.Tensor) else m
            np.savez_compressed(
                cache_dit,
                dit_feats=dit_feats,
                bb_feats=bb_f_np,
                bb_masks=bb_m_np,
                task_labels=np.array(task_labels),
            )
            print(f"  [cache] DiT → {cache_dit}")
    else:
        print(f"  Loading action_encoder cache from {cache_enc}")
        npz = np.load(cache_enc, allow_pickle=True)
        action_feats  = npz["action_feats"]
        gt_actions    = npz["gt_actions"] if "gt_actions" in npz else None
        task_labels   = list(npz["task_labels"].astype(str))
        frame_indices = list(npz["frame_indices"].astype(int))
        if "episode_progress" in npz:
            episode_progress = npz["episode_progress"]
        else:
            episode_progress = compute_episode_progress(frame_indices, dataset)

        dit_feats = None
        bb_f      = None
        bb_m      = None
        if args.run_dit and cache_dit.exists():
            print(f"  Loading DiT cache from {cache_dit}")
            npz2       = np.load(cache_dit, allow_pickle=True)
            dit_feats  = npz2["dit_feats"]
            bb_f       = [torch.from_numpy(x) for x in npz2["bb_feats"]]
            bb_m       = [torch.from_numpy(x) for x in npz2["bb_masks"]]

    N, T_action, D_enc = action_feats.shape
    print(f"  N={N}, T_action={T_action}, D_enc={D_enc}")
    task_dist = {t: task_labels.count(t) for t in sorted(set(task_labels))}
    print(f"  task distribution: {task_dist}")

    # ── Image extraction ──────────────────────────────────────────────────────
    images_dict = {}
    if args.image_keys:
        if not args.skip_cache and cache_img.exists():
            print(f"  Loading image cache from {cache_img}")
            npz_img = np.load(cache_img, allow_pickle=True)
            images_dict = {k: npz_img[k] for k in npz_img.files}
        else:
            images_dict = extract_images(
                dataset, frame_indices,
                image_keys=args.image_keys,
                max_size=args.image_size,
            )
            if images_dict:
                np.savez_compressed(cache_img, **images_dict)
                print(f"  [cache] images → {cache_img}")

    # ── Timestep comparison cache ─────────────────────────────────────────────
    if args.skip_cache or not cache_ts.exists():
        print("[4b/6] Extracting timestep comparison features ...")
        # Reset dataloader with same seed
        sampler2 = ProportionalTaskSampler(
            dataset=dataset,
            valid_episode_indices=valid_episodes,
            target_proportions={"cabinet": 1.0, "door": 1.0, "bottle": 1.0},
            epoch_size=args.num_samples * 4,
            shuffle=True,
        )
        dl2 = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler2,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            drop_last=False,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        ts_feats, ts_labels = extract_features_multi_timestep(
            policy, preprocessor, dl2,
            num_samples=args.num_samples,
            device=args.device,
            timestep_configs=TIMESTEP_CONFIGS,
            seed=args.seed,
        )
        print()
        save_dict = {name: arr for name, arr in ts_feats.items()}
        save_dict["task_labels"] = np.array(ts_labels)
        np.savez_compressed(cache_ts, **save_dict)
        print(f"  [cache] timestep → {cache_ts}")
    else:
        print(f"  Loading timestep cache from {cache_ts}")
        npz3     = np.load(cache_ts, allow_pickle=True)
        ts_feats = {name: npz3[name] for name in TIMESTEP_CONFIGS}
        ts_labels = list(npz3["task_labels"].astype(str))

    # ── UMAP availability ─────────────────────────────────────────────────────
    umap_available = True
    try:
        import umap  # noqa: F401
    except ImportError:
        umap_available = False

    # Pick a "default" reduction for multi-panel plots
    def default_reduce(X, nb=15):
        if umap_available:
            return run_umap(X, n_neighbors=nb, seed=args.seed)
        else:
            return run_tsne(X, perplexity=30, n_iter=args.tsne_n_iter, seed=args.seed)
    reduce_label = "umap_nb15" if umap_available else "tsne_p30"

    # ══════════════════════════════════════════════════════════════════════════
    # [5/6] Per-representation standard plots
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[5/6] Generating per-representation plots ...")

    # ① action_encoder
    enc_dir = out_dir / "action_encoder"
    enc_dir.mkdir(exist_ok=True)
    print(f"\n── ① action_encoder ──")
    visualise_representation(
        action_feats, task_labels, "action_encoder", enc_dir, args,
        episode_progress=episode_progress, frame_indices=frame_indices,
        images_dict=images_dict)

    # ② DiT output
    if dit_feats is not None:
        dit_dir = out_dir / "dit_output"
        dit_dir.mkdir(exist_ok=True)
        print(f"\n── ② DiT output ──")
        visualise_representation(
            dit_feats, task_labels, "DiT_output", dit_dir, args,
            episode_progress=episode_progress, frame_indices=frame_indices,
            images_dict=images_dict)

    # ③ raw action (flatten / delta)
    if gt_actions is not None:
        raw_dir = out_dir / "raw_action"
        raw_dir.mkdir(exist_ok=True)
        print(f"\n── ③ raw_action ──")
        visualise_raw_action(
            gt_actions, task_labels,
            out_dir=raw_dir, args=args,
            episode_progress=episode_progress, frame_indices=frame_indices,
            images_dict=images_dict)

    # ══════════════════════════════════════════════════════════════════════════
    # [6/6] Special analysis plots
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[6/6] Generating special analysis plots ...")

    spec_dir = out_dir / "analysis"
    spec_dir.mkdir(exist_ok=True)

    # ── ③ Temporal slice 4-panel ─────────────────────────────────────────────
    print("\n  ③ Temporal slice analysis ...")
    t_slices = get_temporal_slices(T_action)
    print(f"  T_action={T_action}, slices: {t_slices}")

    umap_nbs_for_slice = args.umap_neighbors if umap_available else []
    tsne_perps_for_slice = args.tsne_perplexities if not umap_available else []

    for nb in umap_nbs_for_slice:
        def _red_nb(X, _nb=nb):
            return run_umap(X, n_neighbors=_nb, seed=args.seed)
        plot_temporal_slices_panel(
            action_feats, task_labels, _red_nb,
            title_prefix=f"action_encoder  |  UMAP n_neighbors={nb}",
            out_path=spec_dir / f"temporal_slice_category_umap_nb{nb}.png",
            slices=t_slices,
        )
        plot_temporal_slices_task_panel(
            action_feats, task_labels, _red_nb,
            title_prefix=f"action_encoder  |  UMAP n_neighbors={nb}",
            out_path=spec_dir / f"temporal_slice_task_umap_nb{nb}.png",
            slices=t_slices,
        )
    for perp in tsne_perps_for_slice:
        def _red_tsne(X, _p=perp):
            return run_tsne(X, perplexity=_p, n_iter=args.tsne_n_iter, seed=args.seed)
        plot_temporal_slices_panel(
            action_feats, task_labels, _red_tsne,
            title_prefix=f"action_encoder  |  t-SNE perp={perp}",
            out_path=spec_dir / f"temporal_slice_category_tsne_p{perp}.png",
            slices=t_slices,
        )
        plot_temporal_slices_task_panel(
            action_feats, task_labels, _red_tsne,
            title_prefix=f"action_encoder  |  t-SNE perp={perp}",
            out_path=spec_dir / f"temporal_slice_task_tsne_p{perp}.png",
            slices=t_slices,
        )

    # ── Timestep comparison 3-panel ──────────────────────────────────────────
    print("\n  Timestep comparison ...")
    for nb in (args.umap_neighbors if umap_available else []):
        def _red_nb(X, _nb=nb):
            return run_umap(X, n_neighbors=_nb, seed=args.seed)
        for agg in ["mean", "last"]:
            plot_timestep_comparison(
                ts_feats, ts_labels, _red_nb, agg,
                out_path=spec_dir / f"timestep_compare_{agg}_umap_nb{nb}.png",
            )
    if not umap_available:
        def _red_tsne(X):
            return run_tsne(X, perplexity=30, n_iter=args.tsne_n_iter, seed=args.seed)
        for agg in ["mean", "last"]:
            plot_timestep_comparison(
                ts_feats, ts_labels, _red_tsne, agg,
                out_path=spec_dir / f"timestep_compare_{agg}_tsne_p30.png",
            )

    # ── 3-way comparison: action_encoder vs DiT vs backbone ──────────────────
    if dit_feats is not None and bb_f is not None:
        print("\n  3-way comparison (action_encoder / DiT / backbone) ...")
        bb_X   = aggregate_backbone(bb_f, bb_m, method="mask_mean")   # (N, D_bb)
        for nb in ([15] if umap_available else []):
            def _red3(X, _nb=nb):
                return run_umap(X, n_neighbors=_nb, seed=args.seed)
            for agg in ["mean", "last"]:
                acenc_X = aggregate(action_feats, agg)
                dit_X   = aggregate(dit_feats,    agg)
                title = (f"3-way comparison  |  agg={agg}  |  UMAP n_neighbors={nb}\n"
                         f"① action_encoder  |  ② DiT output  |  ③ backbone")
                plot_3way_comparison(
                    acenc_X, dit_X, bb_X,
                    task_labels=task_labels,
                    reduction_fn=_red3,
                    title=title,
                    out_path=spec_dir / f"3way_compare_{agg}_umap_nb{nb}.png",
                )

    print(f"\n✅ Done. All outputs saved to: {out_dir}")
    print(f"  ├── action_encoder/   — ① per-agg t-SNE/UMAP plots")
    if dit_feats is not None:
        print(f"  ├── dit_output/      — ② DiT per-agg plots")
    print(f"  ├── analysis/")
    print(f"  │   ├── temporal_slice_*  — ③ early/mid/late/full panels")
    print(f"  │   ├── timestep_compare_* — clean vs half vs noisy")
    if dit_feats is not None:
        print(f"  │   └── 3way_compare_*  — action_encoder vs DiT vs backbone")
    print(f"  ├── action_encoder_cache.npz")
    if dit_feats is not None:
        print(f"  └── dit_features_cache.npz")


if __name__ == "__main__":
    main()
