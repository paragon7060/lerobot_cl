"""Batch parity check for the official-style training path.

Verifies that the new LeRobot official adapter produces *the same* batch as
`gr00t_finetune.py`'s reference path on the same raw sample. This is **not** a
comparison against the existing LeRobot processor (`processor_groot`) — that
path is preserved for inference / eval only.

Reference (path A): build `LeRobotSingleDataset` + `DATA_CONFIG_MAP[name]` +
`gr00t.model.transforms.collate` directly, mirroring `gr00t_finetune.py`.

Adapter (path B): same pieces, but obtained through this package's
`build_official_dataset` + `build_official_collate` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .official_data import (
    build_official_collate,
    build_official_dataset,
    ensure_isaac_gr00t_on_path,
)
from .robocasa_preset import RobocasaPreset


# Keys that should match exactly across the two paths. The first group is
# numerical (compared with allclose), the second is integer (torch.equal).
_FLOAT_KEYS = ("state", "action")
_BOOL_OR_INT_KEYS = (
    "state_mask",
    "action_mask",
    "embodiment_id",
    "has_real_action",
    "segmentation_target",
    "segmentation_target_mask",
    "eagle_input_ids",
    "eagle_attention_mask",
    "eagle_image_grid_thw",
)
_PIXEL_KEY = "eagle_pixel_values"


@dataclass
class ParityReport:
    keys_only_in_reference: list[str] = field(default_factory=list)
    keys_only_in_adapter: list[str] = field(default_factory=list)
    shape_mismatches: dict[str, tuple] = field(default_factory=dict)
    dtype_mismatches: dict[str, tuple] = field(default_factory=dict)
    value_mismatches: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (
            self.keys_only_in_reference
            or self.keys_only_in_adapter
            or self.shape_mismatches
            or self.dtype_mismatches
            or self.value_mismatches
        )

    def summary(self) -> str:
        if self.ok:
            return "[parity] OK — reference and adapter batches match."
        lines = ["[parity] MISMATCH:"]
        if self.keys_only_in_reference:
            lines.append(f"  only in reference: {self.keys_only_in_reference}")
        if self.keys_only_in_adapter:
            lines.append(f"  only in adapter:   {self.keys_only_in_adapter}")
        for key, (a, b) in self.shape_mismatches.items():
            lines.append(f"  shape mismatch [{key}]: reference={a}, adapter={b}")
        for key, (a, b) in self.dtype_mismatches.items():
            lines.append(f"  dtype mismatch [{key}]: reference={a}, adapter={b}")
        for key, msg in self.value_mismatches.items():
            lines.append(f"  value  mismatch [{key}]: {msg}")
        return "\n".join(lines)


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))


def _compare_tensor(key: str, a, b, *, atol: float, report: ParityReport) -> None:
    a = _to_tensor(a)
    b = _to_tensor(b)
    if a.shape != b.shape:
        report.shape_mismatches[key] = (tuple(a.shape), tuple(b.shape))
        return
    if a.dtype != b.dtype:
        report.dtype_mismatches[key] = (str(a.dtype), str(b.dtype))
        # Cast b to a.dtype so we can still compare values; flag separately.
        b = b.to(a.dtype)
    if a.is_floating_point():
        if not torch.allclose(a, b, atol=atol, rtol=0.0):
            diff = (a - b).abs().max().item()
            report.value_mismatches[key] = f"max abs diff = {diff:.6g} (atol={atol})"
    else:
        if not torch.equal(a, b):
            diff_count = (a != b).sum().item()
            report.value_mismatches[key] = f"{diff_count} elements differ"


def check_batch_parity(
    *,
    reference_batch: dict,
    adapter_batch: dict,
    atol: float = 1e-4,
) -> ParityReport:
    """Compare the official reference batch (path A) against the adapter batch (path B)."""
    report = ParityReport()

    ref_keys = set(reference_batch.keys())
    adp_keys = set(adapter_batch.keys())
    report.keys_only_in_reference = sorted(ref_keys - adp_keys)
    report.keys_only_in_adapter = sorted(adp_keys - ref_keys)

    common = ref_keys & adp_keys
    for key in sorted(common):
        if key == "eagle_content":
            # Pre-collate structure; should not appear post-collate. Skip.
            continue
        if key in _FLOAT_KEYS or key == _PIXEL_KEY:
            _compare_tensor(key, reference_batch[key], adapter_batch[key], atol=atol, report=report)
        elif key in _BOOL_OR_INT_KEYS:
            _compare_tensor(key, reference_batch[key], adapter_batch[key], atol=0.0, report=report)
        else:
            # Best-effort tensor comparison for keys we did not enumerate above.
            try:
                _compare_tensor(key, reference_batch[key], adapter_batch[key], atol=atol, report=report)
            except Exception as e:  # noqa: BLE001
                report.value_mismatches[key] = f"could not compare: {e}"

    return report


def _build_reference_batch(
    dataset_path: str | Path,
    *,
    preset: RobocasaPreset,
    sample_index: int = 0,
) -> dict:
    """Path A — wire the official pieces directly, the way `gr00t_finetune.py` does."""
    ensure_isaac_gr00t_on_path()
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
    from gr00t.model.transforms import build_eagle_processor, collate

    data_config = DATA_CONFIG_MAP[preset.data_config_name]
    dataset = LeRobotSingleDataset(
        dataset_path=str(dataset_path),
        modality_configs=data_config.modality_config(),
        transforms=data_config.transform(),
        embodiment_tag=EmbodimentTag(preset.embodiment_tag),
        video_backend=preset.video_backend,
    )
    eagle_processor = build_eagle_processor(DEFAULT_EAGLE_PATH)
    feature = dataset[sample_index]
    return collate([feature], eagle_processor)


def _build_adapter_batch(
    dataset_path: str | Path,
    *,
    preset: RobocasaPreset,
    sample_index: int = 0,
) -> dict:
    """Path B — go through the helpers in `groot_common.official_data`."""
    dataset = build_official_dataset([dataset_path], preset=preset)
    collate_fn = build_official_collate()
    feature = dataset[sample_index]
    return collate_fn([feature])


def run_parity_smoke_test(
    dataset_path: str | Path,
    *,
    preset: RobocasaPreset,
    sample_index: int = 0,
    atol: float = 1e-4,
    seed: int = 0,
) -> ParityReport:
    """Build one sample through both paths and return a comparison report.

    Both paths share the same RNG-affected components (VideoColorJitter); seed
    torch / numpy / random before each build so the reference and adapter
    sample identical augmentations.
    """
    import random

    def _seed_all() -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    _seed_all()
    reference_batch = _build_reference_batch(
        dataset_path, preset=preset, sample_index=sample_index
    )
    _seed_all()
    adapter_batch = _build_adapter_batch(
        dataset_path, preset=preset, sample_index=sample_index
    )

    return check_batch_parity(
        reference_batch=reference_batch,
        adapter_batch=adapter_batch,
        atol=atol,
    )


def compare_tensor_reports(
    phase1_report: dict[str, dict[str, Any]],
    phase2_report: dict[str, dict[str, Any]],
    *,
    required_keys_only: bool = False,
    ignore_batch_dim: bool = False,
    exclude_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Compare tensor metadata reports (shape/dtype) from phase1/phase2 scripts."""
    k1 = set(phase1_report.keys())
    k2 = set(phase2_report.keys())

    if required_keys_only:
        k1 = k1 & k2
        k2 = set(k2)
    if exclude_keys:
        k1 = {k for k in k1 if k not in exclude_keys}
        k2 = {k for k in k2 if k not in exclude_keys}

    common = sorted(k1 & k2)
    missing_in_phase1 = sorted(k2 - k1)
    extra_in_phase1 = sorted(k1 - k2)

    shape_mismatch = []
    dtype_mismatch = []
    for key in common:
        s1 = phase1_report[key]["shape"]
        s2 = phase2_report[key]["shape"]
        if ignore_batch_dim and len(s1) > 0 and len(s2) > 0:
            s1 = s1[1:]
            s2 = s2[1:]
        if s1 != s2:
            shape_mismatch.append(
                {"key": key, "phase1": phase1_report[key]["shape"], "phase2": phase2_report[key]["shape"]}
            )
        if phase1_report[key]["dtype"] != phase2_report[key]["dtype"]:
            dtype_mismatch.append(
                {"key": key, "phase1": phase1_report[key]["dtype"], "phase2": phase2_report[key]["dtype"]}
            )

    return {
        "common_key_count": len(common),
        "missing_in_phase1_count": len(missing_in_phase1),
        "extra_in_phase1_count": len(extra_in_phase1),
        "shape_mismatch_count": len(shape_mismatch),
        "dtype_mismatch_count": len(dtype_mismatch),
        "missing_in_phase1": missing_in_phase1,
        "extra_in_phase1": extra_in_phase1,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
    }
