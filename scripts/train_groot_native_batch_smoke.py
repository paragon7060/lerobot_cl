#!/usr/bin/env python
"""Phase-1 smoke: LeRobot-native batch + GrootPolicy one-step training check.

This script intentionally avoids official direct dataset/collate bridge.
It validates only the minimal trainability path on a LeRobot-native batch:
import -> preprocess -> forward -> backward -> optimizer step -> native save.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.policies.groot_common import LeRobotNativeBatchBuilder, make_robocasa_preset, tensor_report
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot_robocasa.configuration_groot import GrootRobocasaConfig
from lerobot.policies.groot_robocasa.processor_groot import make_groot_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model_path", type=str, default="paragon7060/Robocasa_baseline")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    return p.parse_args()


def _changed_param_count(before: dict[str, torch.Tensor], model: torch.nn.Module) -> int:
    changed = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        prev = before[name]
        if not torch.equal(prev, p.detach().cpu()):
            changed += 1
    return changed


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "phase1_manifest.json"

    # Robocasa-aligned config values for batch semantics.
    preset_cfg = GrootRobocasaConfig()
    cfg = GrootConfig(base_model_path=args.model_path)
    cfg.chunk_size = preset_cfg.chunk_size
    cfg.n_action_steps = preset_cfg.n_action_steps
    cfg.max_action_dim = preset_cfg.max_action_dim
    cfg.max_state_dim = preset_cfg.max_state_dim
    cfg.embodiment_tag = preset_cfg.embodiment_tag
    cfg.tokenizer_assets_repo = preset_cfg.tokenizer_assets_repo
    cfg.video_backend = "pyav"
    cfg.input_features = {
        f"{OBS_IMAGES}.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.left_shoulder": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.right_shoulder": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))}
    cfg.device = args.device

    policy = GrootPolicy.from_pretrained(args.model_path, config=cfg)
    policy.train()
    policy.to(args.device)

    ds_meta = LeRobotDatasetMetadata(
        repo_id=args.repo_id,
        root=args.dataset_root / args.repo_id,
    )
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    dataset = MultiLeRobotDataset(
        repo_ids=[args.repo_id],
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=policy.config.video_backend,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    pre, _ = make_groot_pre_post_processors(policy.config, dataset_stats=dataset.stats)
    raw_batch = next(iter(dataloader))
    batch = pre(raw_batch)
    batch_builder = LeRobotNativeBatchBuilder(make_robocasa_preset(video_backend=policy.config.video_backend))

    batch_report = tensor_report(batch_builder.build_parity_view(batch))
    optimizer = AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    before = {
        n: p.detach().cpu().clone()
        for n, p in policy.named_parameters()
        if p.requires_grad
    }

    loss, loss_dict = policy(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {float(loss)}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
    if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
        raise RuntimeError(f"Non-finite grad norm: {float(grad_norm)}")
    optimizer.step()

    changed_count = _changed_param_count(before, policy)

    native_dir = args.output_dir / "native_checkpoint"
    policy.save_pretrained(native_dir)

    manifest = {
        "model_path": args.model_path,
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "batch_size": args.batch_size,
        "loss": float(loss.detach().cpu().item()),
        "flow_matching_loss": float(
            loss_dict.get("flow_matching_loss", loss_dict.get("loss", loss)).detach().cpu().item()
            if isinstance(loss_dict.get("flow_matching_loss", loss_dict.get("loss", loss)), torch.Tensor)
            else loss_dict.get("flow_matching_loss", loss_dict.get("loss", float(loss.detach().cpu().item())))
        ),
        "grad_norm": float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        "changed_param_count": int(changed_count),
        "native_save_dir": str(native_dir),
        "batch_report": batch_report,
    }
    log_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"ok": True, "manifest": str(log_path), "native_dir": str(native_dir)}))


if __name__ == "__main__":
    main()
