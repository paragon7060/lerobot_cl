from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


def parse_num_demos_from_filter_key(filter_key: str | None) -> int | None:
    """Parse `<N>_demos` style filter keys used in official Robocasa soups."""
    if filter_key is None:
        return None
    if not filter_key.endswith("_demos"):
        raise ValueError(f"Unsupported filter_key format: {filter_key!r}")
    return int(filter_key.split("_", 1)[0])


def load_episode_ids(repo_root: Path) -> list[int]:
    """Load episode ids from either `meta/episodes.jsonl` or v3 parquet files."""
    episodes_jsonl = repo_root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        with episodes_jsonl.open("r") as f:
            return [int(json.loads(line)["episode_index"]) for line in f]

    parquet_files = sorted((repo_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No episode metadata found under {repo_root}. "
            "Expected meta/episodes.jsonl or data/**/*.parquet."
        )

    import pyarrow.parquet as pq

    episode_ids: set[int] = set()
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=["episode_index"])
        episode_ids.update(int(x) for x in table["episode_index"].to_pylist())
    return sorted(episode_ids)


def sample_episode_ids_from_filter_key(
    repo_root: Path,
    filter_key: str | None,
    *,
    seed: int | None = 0,
) -> list[int] | None:
    """Apply official-style random subset selection for one dataset repo.

    Returns:
        - `None` when no filtering is requested or when `N >= total`.
        - Selected episode ids otherwise.
    """
    num_demos = parse_num_demos_from_filter_key(filter_key)
    if num_demos is None:
        return None

    all_demo_ids = load_episode_ids(repo_root)
    if num_demos >= len(all_demo_ids):
        return None

    state = random.getstate()
    try:
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_demo_ids)
    finally:
        random.setstate(state)
    return all_demo_ids[:num_demos]


def subset_manifest_path(
    dataset_root: Path,
    *,
    split: str,
    filter_key: str,
    seed: int | None,
) -> Path:
    seed_tag = "none" if seed is None else str(seed)
    return dataset_root / f"subset_{split}_{filter_key}_seed{seed_tag}.json"


def load_or_create_subset_manifest(
    dataset_root: Path,
    repo_ids: Sequence[str],
    *,
    split: str,
    filter_key: str,
    seed: int | None,
    create_if_missing: bool = True,
) -> tuple[dict[str, list[int] | None], Path, str]:
    """Load cached subset manifest or create it once.

    Returns:
        tuple of `(episodes_by_repo, manifest_path, source)` where source is
        `"cache"` or `"created"`.
    """
    manifest_path = subset_manifest_path(
        dataset_root, split=split, filter_key=filter_key, seed=seed
    )
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            payload = json.load(f)
        episodes_by_repo = payload["episodes_by_repo"]
        return episodes_by_repo, manifest_path, "cache"

    if not create_if_missing:
        raise FileNotFoundError(
            f"Subset manifest not found: {manifest_path}. "
            "Main process should create it first."
        )

    episodes_by_repo: dict[str, list[int] | None] = {}
    for repo_id in repo_ids:
        repo_root = dataset_root / repo_id
        episodes_by_repo[repo_id] = sample_episode_ids_from_filter_key(
            repo_root, filter_key, seed=seed
        )

    payload = {
        "split": split,
        "filter_key": filter_key,
        "seed": seed,
        "repo_ids": list(repo_ids),
        "episodes_by_repo": episodes_by_repo,
    }
    with manifest_path.open("w") as f:
        json.dump(payload, f)
    return episodes_by_repo, manifest_path, "created"
