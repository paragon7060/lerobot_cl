#!/usr/bin/env python
"""Phase-6 smoke skeleton: GrootMGDPolicy + LeRobot-native batch one-step check.

Scope:
- No official eval / no parity runner / no autoshim export.
- No non-zero custom loss; custom_loss_weight must stay 0.0 in this phase.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.policies.groot_common import (
    LeRobotNativeBatchBuilder,
    apply_to_policy_config,
    assert_groot_compatible,
    assert_official_config_match,
    forward_with_mgd_custom_loss_hook,
    make_robocasa_preset,
    tensor_report,
)
from lerobot.policies.groot_mgd.configuration_groot import GrootMGDConfig
from lerobot.policies.groot_mgd.modeling_groot import GrootMGDPolicy
from lerobot.policies.groot_mgd.processor_groot import make_groot_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--model_path", type=str, default="paragon7060/Robocasa_baseline")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--custom_loss_weight", type=float, default=0.0)
    p.add_argument("--video_backend", type=str, default="pyav")
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


def _build_run_dir(output_root: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name if run_name else "default"
    run_dir = output_root / f"phase6_mgd_native_smoke_{stamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def main() -> None:
    args = _parse_args()
    if args.custom_loss_weight != 0.0:
        raise ValueError("Phase-6 smoke enforces custom_loss_weight=0.0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = _build_run_dir(args.output_root, args.run_name)
    manifest_path = run_dir / "phase6_manifest.json"
    report_path = run_dir / "phase6_batch_report.json"

    preset = make_robocasa_preset(video_backend=args.video_backend)
    cfg = GrootMGDConfig(base_model_path=args.model_path, device=args.device)
    apply_to_policy_config(cfg, preset)
    cfg.video_backend = args.video_backend
    # Phase-7 gate: keep MGD branch connected but no-op in optimization.
    cfg.mgd_enabled = True
    cfg.mgd_loss_weight = 0.0
    cfg.mgd_fm_loss_weight = 1.0
    cfg.mgd_preserve_weight = 0.0
    cfg.input_features = {
        f"{OBS_IMAGES}.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.left_shoulder": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.right_shoulder": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))}

    policy = GrootMGDPolicy.from_pretrained(args.model_path, config=cfg)
    assert_groot_compatible(policy)
    assert_official_config_match(policy, preset)
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

    batch_builder = LeRobotNativeBatchBuilder(preset)
    pre, _ = make_groot_pre_post_processors(policy.config, dataset_stats=dataset.stats)
    raw_batch = next(iter(dataloader))
    batch = batch_builder.build_train_batch(raw_batch, pre)
    parity_view = batch_builder.build_parity_view(batch)
    batch_report = tensor_report(parity_view)
    report_path.write_text(json.dumps(batch_report, indent=2))

    optimizer = AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    before = {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}

    total_loss, loss_dict = forward_with_mgd_custom_loss_hook(
        policy,
        batch,
        custom_loss_weight=args.custom_loss_weight,
    )
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"Non-finite total loss: {float(total_loss)}")

    flow_matching_loss = _to_float(loss_dict["flow_matching_loss"])
    total_loss_value = _to_float(total_loss)
    if abs(total_loss_value - flow_matching_loss) > 1e-8:
        raise RuntimeError(
            f"Expected total_loss == flow_matching_loss at custom_loss_weight=0.0, "
            f"got total={total_loss_value} flow={flow_matching_loss}"
        )

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
    if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
        raise RuntimeError(f"Non-finite grad norm: {float(grad_norm)}")
    optimizer.step()

    changed_count = _changed_param_count(before, policy)

    native_dir = run_dir / "native_checkpoint"
    policy.save_pretrained(native_dir)

    training_only_key_prefix_candidates = [
        "action_target_projector.",
        "mgd_recon_head.",
        "channel_mask.",
    ]
    manifest = {
        "script": "scripts/train_groot_mgd_native_batch_smoke.py",
        "run_dir": str(run_dir),
        "model_path": args.model_path,
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "batch_size": args.batch_size,
        "custom_loss_weight": args.custom_loss_weight,
        "total_loss": total_loss_value,
        "flow_matching_loss": flow_matching_loss,
        "mgd_loss": float(loss_dict.get("mgd_loss", 0.0)),
        "custom_loss": float(loss_dict.get("custom_loss", 0.0)),
        "grad_norm": _to_float(grad_norm),
        "changed_param_count": int(changed_count),
        "native_save_dir": str(native_dir),
        "batch_report_path": str(report_path),
        "trainable_param_count": int(sum(p.numel() for p in policy.parameters() if p.requires_grad)),
        "state_dict_key_count": int(len(policy.state_dict())),
        "batch_spec": {
            "camera_order": list(batch_builder.batch_spec.camera_order),
            "state_order": list(batch_builder.batch_spec.state_order),
            "action_order": list(batch_builder.batch_spec.action_order),
            "action_horizon": int(batch_builder.batch_spec.action_horizon),
            "chunk_size": int(batch_builder.batch_spec.chunk_size),
            "padded_state_dim": int(batch_builder.batch_spec.padded_state_dim),
            "padded_action_dim": int(batch_builder.batch_spec.padded_action_dim),
        },
        "training_only_key_prefix_candidates_for_export_exclusion": training_only_key_prefix_candidates,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"ok": True, "manifest": str(manifest_path), "run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
