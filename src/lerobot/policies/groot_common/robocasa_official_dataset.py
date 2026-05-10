"""Official-style RoboCasa dataset discovery for LeRobot leaf datasets.

This module intentionally follows RoboCasa/Isaac-GR00T conventions:
- split is explicit (`pretrain`, `target`, `real`)
- task category buckets are explicit (`atomic`, `composite`)
- leaf dataset is `.../<split>/<category>/<Task>/<Date>/lerobot`

It does not try to auto-infer arbitrary layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


SUPPORTED_SPLITS: tuple[str, ...] = ("pretrain", "target", "real")
SUPPORTED_CATEGORIES: tuple[str, ...] = ("atomic", "composite")


@dataclass(frozen=True)
class RoboCasaOfficialDiscovery:
    root: Path
    split: str
    repo_ids: list[str]
    dataset_paths: list[Path]
    atomic_repo_ids: list[str]
    composite_repo_ids: list[str]
    atomic_count: int
    composite_count: int
    total_count: int
    v3_compatible_count: int
    non_v3_count: int


def _is_official_leaf_valid(dataset_path: Path) -> bool:
    """Minimum checks aligned with GR00T official loader expectations."""
    meta = dataset_path / "meta"
    required = (
        meta / "modality.json",
        meta / "info.json",
        meta / "episodes.jsonl",
        meta / "tasks.jsonl",
    )
    return all(path.exists() for path in required)


def _is_v3_compatible(dataset_path: Path) -> bool:
    """Check LeRobot v3 compatibility via meta/info.json codebase version."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return False
    try:
        info = json.loads(info_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    codebase_version = str(info.get("codebase_version", "")).strip()
    return codebase_version.startswith("v3.")


def _discover_category(root: Path, split: str, category: str) -> list[Path]:
    category_root = root / split / category
    if not category_root.exists():
        return []

    # official v1.0 layout: <split>/<category>/<Task>/<Date>/lerobot
    candidates = sorted(category_root.glob("*/*/lerobot"))
    valid_paths: list[Path] = []
    for path in candidates:
        if not path.is_dir():
            continue
        if _is_official_leaf_valid(path):
            valid_paths.append(path)
    return valid_paths


def discover_robocasa_official_lerobot_repos(
    root: str | Path,
    split: str = "pretrain",
) -> RoboCasaOfficialDiscovery:
    """Discover official RoboCasa LeRobot datasets for a given split.

    Args:
        root: RoboCasa dataset root that contains split directories (e.g. `v1.0` root).
        split: One of `pretrain`, `target`, `real`.
    """
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split={split!r}. Expected one of {SUPPORTED_SPLITS}.")

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root must be a directory: {root_path}")

    # Root is expected to be the version root (e.g. .../robocasa_dataset/v1.0),
    # not a split directory.
    if root_path.name in SUPPORTED_SPLITS:
        raise ValueError(
            "root must point to the version directory (e.g. .../robocasa_dataset/v1.0), "
            f"not split dir {root_path.name!r}. Received: {root_path}"
        )

    split_root = root_path / split
    if not split_root.exists():
        raise FileNotFoundError(
            f"Split directory does not exist: {split_root}. "
            f"Expected root to contain split dirs like {SUPPORTED_SPLITS}."
        )

    atomic_paths = _discover_category(root_path, split, "atomic")
    composite_paths = _discover_category(root_path, split, "composite")
    dataset_paths = [*atomic_paths, *composite_paths]

    repo_ids = [path.relative_to(root_path).as_posix() for path in dataset_paths]
    atomic_repo_ids = [path.relative_to(root_path).as_posix() for path in atomic_paths]
    composite_repo_ids = [path.relative_to(root_path).as_posix() for path in composite_paths]

    v3_compatible_count = sum(1 for path in dataset_paths if _is_v3_compatible(path))
    total_count = len(dataset_paths)

    return RoboCasaOfficialDiscovery(
        root=root_path,
        split=split,
        repo_ids=repo_ids,
        dataset_paths=dataset_paths,
        atomic_repo_ids=atomic_repo_ids,
        composite_repo_ids=composite_repo_ids,
        atomic_count=len(atomic_repo_ids),
        composite_count=len(composite_repo_ids),
        total_count=total_count,
        v3_compatible_count=v3_compatible_count,
        non_v3_count=total_count - v3_compatible_count,
    )
