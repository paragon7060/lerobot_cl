"""Batch report and parity-view builders for LeRobot-native GR00T training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .batch_spec import BatchSpec
from .robocasa_preset import ROBOCASA_VIDEO_FEATURE_KEYS, RobocasaPreset


def tensor_report(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            report[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    return report


@dataclass
class LeRobotNativeBatchBuilder:
    """Build report-oriented batch views for parity checks."""

    preset: RobocasaPreset

    @property
    def batch_spec(self) -> BatchSpec:
        return BatchSpec(
            camera_order=tuple(self.preset.video_feature_keys),
            state_order=("robot_state",),
            action_order=("robot_action",),
            action_horizon=self.preset.n_action_steps,
            chunk_size=self.preset.chunk_size,
            padded_state_dim=self.preset.max_state_dim,
            padded_action_dim=self.preset.max_action_dim,
        )

    def build_parity_view(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Return a parity-aligned batch view without mutating the train batch."""
        view = dict(batch)
        action = view.get("action")
        if not isinstance(action, torch.Tensor):
            raise KeyError("batch['action'] tensor is required to build parity view")
        bsz = int(action.shape[0])
        device = action.device

        if "has_real_action" not in view:
            view["has_real_action"] = torch.ones((bsz,), dtype=torch.bool, device=device)
        if "segmentation_target" not in view:
            view["segmentation_target"] = torch.zeros((bsz, 2), dtype=torch.float64, device=device)
        if "segmentation_target_mask" not in view:
            view["segmentation_target_mask"] = torch.zeros((bsz, 1), dtype=torch.float64, device=device)

        for key in ("state", "action"):
            if key in view and isinstance(view[key], torch.Tensor):
                view[key] = view[key].to(torch.float64)
        return view

    def build_train_batch(
        self,
        raw_batch: dict[str, Any],
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the training batch using the caller-provided preprocessor."""
        batch = preprocessor(raw_batch)
        if "state" not in batch or "action" not in batch:
            raise KeyError("Preprocessed batch must contain 'state' and 'action'")
        return batch


def make_robocasa_preset(video_backend: str = "opencv") -> RobocasaPreset:
    """Factory used by scripts to avoid duplicating preset defaults."""
    return RobocasaPreset(
        video_backend=video_backend,
        video_feature_keys=ROBOCASA_VIDEO_FEATURE_KEYS,
    )
