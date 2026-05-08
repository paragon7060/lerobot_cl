"""Bridges to the official Isaac-GR00T data + collate stack.

The training script wires LeRobot's parser/Accelerator loop to Isaac-GR00T's
`LeRobotSingleDataset` / `LeRobotMixtureDataset`, the `DATA_CONFIG_MAP` data
configs, and `gr00t.model.transforms.collate`. This module is the only place
those imports live.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .robocasa_preset import RobocasaPreset


# The official package lives next to `lerobot_cl/` inside the `clvla` workspace.
# Resolve relative to this file so the helper works on both the local mount and
# the remote 48server checkout without hard-coding home paths.
_THIS_FILE = Path(__file__).resolve()
_ISAAC_GR00T_REL = ("benchmarks", "robocasa365", "Isaac-GR00T")


def _candidate_isaac_gr00t_paths() -> list[Path]:
    """Likely on-disk locations of the Isaac-GR00T repo.

    1. Sibling of `lerobot_cl/` under `clvla/` (preferred — server layout).
    2. `ISAAC_GR00T_PATH` environment override.
    """
    candidates: list[Path] = []
    env_override = os.environ.get("ISAAC_GR00T_PATH")
    if env_override:
        candidates.append(Path(env_override).resolve())
    # parents[5] = clvla/ (groot_common/ -> policies/ -> lerobot/ -> src/ -> lerobot_cl/ -> clvla/)
    if len(_THIS_FILE.parents) >= 6:
        candidates.append(_THIS_FILE.parents[5].joinpath(*_ISAAC_GR00T_REL))
    return candidates


def ensure_isaac_gr00t_on_path() -> Path:
    """Make `import gr00t` work, returning the resolved repo root.

    No-op when `gr00t` is already importable. Otherwise prepends the first
    existing candidate path to `sys.path`.
    """
    try:
        import gr00t  # noqa: F401

        return Path(gr00t.__file__).resolve().parents[1]
    except ImportError:
        pass

    for candidate in _candidate_isaac_gr00t_paths():
        if candidate.exists() and (candidate / "gr00t").is_dir():
            sys.path.insert(0, str(candidate))
            import gr00t  # noqa: F401

            return candidate

    searched = "\n  - ".join(str(c) for c in _candidate_isaac_gr00t_paths())
    raise ImportError(
        "Could not locate the Isaac-GR00T repo. Tried:\n  - "
        + searched
        + "\nSet ISAAC_GR00T_PATH to override."
    )


def _build_single_dataset(
    dataset_path: str | Path,
    *,
    preset: RobocasaPreset,
    filter_key: str | None = None,
):
    """Instantiate an `LeRobotSingleDataset` matching `gr00t_finetune.py`."""
    ensure_isaac_gr00t_on_path()
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP

    data_config = DATA_CONFIG_MAP[preset.data_config_name]
    return LeRobotSingleDataset(
        dataset_path=str(dataset_path),
        modality_configs=data_config.modality_config(),
        transforms=data_config.transform(),
        embodiment_tag=EmbodimentTag(preset.embodiment_tag),
        video_backend=preset.video_backend,
        filter_key=filter_key,
    )


def build_official_dataset(
    dataset_paths: Sequence[str | Path],
    *,
    preset: RobocasaPreset,
    filter_keys: Sequence[str | None] | None = None,
    ds_weights_alpha: float = 0.4,
    balance_dataset_weights: bool = True,
    balance_trajectory_weights: bool = True,
    seed: int = 42,
):
    """Mirror `main()` in `gr00t_finetune.py`: single dataset or weighted mixture.

    Returns:
        Either an `LeRobotSingleDataset` (when one path is supplied) or an
        `LeRobotMixtureDataset` with `balance_*_weights` and the same
        `np.power(len, alpha)` weight scheme as the official script.
    """
    if len(dataset_paths) == 0:
        raise ValueError("dataset_paths must contain at least one entry")

    ensure_isaac_gr00t_on_path()
    from gr00t.data.dataset import LeRobotMixtureDataset

    if filter_keys is None:
        filter_keys = [None] * len(dataset_paths)
    if len(filter_keys) != len(dataset_paths):
        raise ValueError("filter_keys length must match dataset_paths length")

    if len(dataset_paths) == 1:
        return _build_single_dataset(
            dataset_paths[0], preset=preset, filter_key=filter_keys[0]
        )

    singles = [
        _build_single_dataset(p, preset=preset, filter_key=fk)
        for p, fk in zip(dataset_paths, filter_keys)
    ]

    weights = np.array([np.power(len(d), ds_weights_alpha) for d in singles])
    # Match `gr00t_finetune.py`: the mixture loader requires at least one
    # weight to equal 1.0, so we normalise against the first dataset.
    weights = weights / weights[0]

    return LeRobotMixtureDataset(
        data_mixture=list(zip(singles, weights)),
        mode="train",
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )


def _default_eagle_processor():
    ensure_isaac_gr00t_on_path()
    from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
    from gr00t.model.transforms import build_eagle_processor

    return build_eagle_processor(DEFAULT_EAGLE_PATH)


def extract_eagle_processor(policy) -> Any:
    """Return the Eagle processor a policy's GR00T backbone uses, if available.

    Falls back to a freshly-built processor at `DEFAULT_EAGLE_PATH`. Both must
    produce identical tokenisation for parity, so either is fine — preferring
    the in-policy instance avoids loading the HF checkpoint twice.
    """
    backbone = getattr(getattr(policy, "_groot_model", None), "backbone", None)
    if backbone is not None:
        proc = getattr(backbone, "eagle_processor", None)
        if proc is not None:
            return proc
    return _default_eagle_processor()


def build_official_collate(eagle_processor=None) -> Callable[[list[dict]], dict]:
    """Closure of `gr00t.model.transforms.collate` for use as DataLoader's `collate_fn`."""
    ensure_isaac_gr00t_on_path()
    from gr00t.model.transforms import collate as _collate

    proc = eagle_processor if eagle_processor is not None else _default_eagle_processor()

    def _fn(features: list[dict]) -> dict:
        return _collate(features, proc)

    return _fn
