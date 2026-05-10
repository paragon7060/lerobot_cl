"""Runtime dataset selection helpers aligned with official RoboCasa split semantics.

This module targets converted LeRobot v3 datasets laid out as:
  <root>/robocasa_<split>_human_atomic/task_XXXX
  <root>/robocasa_<split>_human_composite/task_XXXX

The output repo_ids are training-ready for MultiLeRobotDataset:
  repo_id = "robocasa_pretrain_human_atomic/task_0001"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pyarrow.parquet as pq

from .robocasa_official_dataset import SUPPORTED_SPLITS


@dataclass(frozen=True)
class RoboCasaOfficialRuntimeDiscovery:
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
    task_label_compatible_count: int
    task_label_missing_count: int


def _is_v3_leaf_valid(task_dir: Path) -> bool:
    meta = task_dir / "meta"
    required = (
        meta / "info.json",
        meta / "tasks.parquet",
        meta / "stats.json",
        meta / "episodes",
        task_dir / "data",
    )
    if not all(path.exists() for path in required):
        return False

    # Require at least one episodes parquet shard (v3 structure).
    if not list((meta / "episodes").glob("chunk-*/file-*.parquet")):
        return False

    try:
        info = json.loads((meta / "info.json").read_text())
    except (json.JSONDecodeError, OSError):
        return False
    codebase_version = str(info.get("codebase_version", "")).strip()
    return codebase_version.startswith("v3.")


def _discover_split_category(root: Path, split: str, category: str) -> list[Path]:
    ds_name = f"robocasa_{split}_human_{category}"
    ds_root = root / ds_name
    if not ds_root.exists():
        return []
    task_dirs = sorted([p for p in ds_root.iterdir() if p.is_dir() and p.name.startswith("task_")])
    return [p for p in task_dirs if _is_v3_leaf_valid(p)]


def _has_task_label_column(task_dir: Path) -> bool:
    tasks_path = task_dir / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return False
    try:
        schema = pq.read_schema(tasks_path)
    except Exception:
        return False
    names = set(schema.names)
    return ("task" in names) or ("task_name" in names)


def discover_robocasa_official_runtime_repos(
    root: str | Path,
    split: str = "pretrain",
) -> RoboCasaOfficialRuntimeDiscovery:
    """Discover training-ready v3 repo_ids for a given split.

    Args:
        root: Converted v3 root, e.g. `.../robocasa_dataset/robocasa_human_v3`.
        split: Split name. Defaults to `pretrain` for the common Robocasa pretrain case.
    """
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split={split!r}. Expected one of {SUPPORTED_SPLITS}.")

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root must be a directory: {root_path}")

    atomic_paths = _discover_split_category(root_path, split, "atomic")
    composite_paths = _discover_split_category(root_path, split, "composite")
    dataset_paths = [*atomic_paths, *composite_paths]

    repo_ids = [path.relative_to(root_path).as_posix() for path in dataset_paths]
    atomic_repo_ids = [path.relative_to(root_path).as_posix() for path in atomic_paths]
    composite_repo_ids = [path.relative_to(root_path).as_posix() for path in composite_paths]

    # Valid set is already v3-checked by _is_v3_leaf_valid; still keep explicit fields.
    v3_compatible_count = len(dataset_paths)
    total_count = len(dataset_paths)
    task_label_compatible_count = sum(1 for path in dataset_paths if _has_task_label_column(path))

    return RoboCasaOfficialRuntimeDiscovery(
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
        task_label_compatible_count=task_label_compatible_count,
        task_label_missing_count=total_count - task_label_compatible_count,
    )
