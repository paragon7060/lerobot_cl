#!/usr/bin/env python
"""Materialize a pre-sliced RoboCasa v3 dataset root from cached subset manifest.

Keeps the source dataset root intact and writes a new dataset root where each
task repo contains only selected episodes (official-style `<N>_demos` subset).
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.groot_common.filter_key_subset import load_or_create_subset_manifest
from lerobot.policies.groot_common.robocasa_official_runtime import discover_robocasa_official_runtime_repos


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MaterializeConfig:
    src_root: str = "/home/seonho/groot_robocasa/robocasa_dataset/robocasa_human_v3"
    dst_root: str = "/home/seonho/groot_robocasa/robocasa_dataset/robocasa_human_v3_presliced_100"
    split: str = "pretrain"
    subset_demos_per_task: int = 100
    subset_seed: int = 0
    overwrite_existing: bool = False
    dry_run: bool = False


def _copy_repo_tree(src_repo_root: Path, dst_repo_root: Path) -> None:
    dst_repo_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_repo_root, dst_repo_root, dirs_exist_ok=False)


def run(cfg: MaterializeConfig) -> None:
    src_root = Path(cfg.src_root)
    dst_root = Path(cfg.dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"src_root does not exist: {src_root}")
    if dst_root.exists() and not cfg.overwrite_existing:
        raise FileExistsError(
            f"dst_root already exists: {dst_root}. "
            "Set --overwrite_existing=true after removing or backing up the directory."
        )
    if dst_root.exists() and cfg.overwrite_existing:
        logger.warning("Removing existing dst_root due to overwrite_existing=true: %s", dst_root)
        if not cfg.dry_run:
            shutil.rmtree(dst_root)

    discovery = discover_robocasa_official_runtime_repos(root=src_root, split=cfg.split)
    repo_ids = discovery.repo_ids
    if not repo_ids:
        raise RuntimeError(f"No runtime repos found under {src_root} split={cfg.split}")

    filter_key = f"{cfg.subset_demos_per_task}_demos"
    episodes_by_repo, manifest_path, source = load_or_create_subset_manifest(
        src_root,
        repo_ids,
        split=cfg.split,
        filter_key=filter_key,
        seed=cfg.subset_seed,
        create_if_missing=True,
    )
    logger.info("Loaded subset manifest (%s): %s", source, manifest_path)
    logger.info("Materializing %d repos from %s -> %s", len(repo_ids), src_root, dst_root)

    kept_as_full_copy = 0
    sliced_count = 0
    skipped_existing = 0

    for idx, repo_id in enumerate(repo_ids, start=1):
        src_repo_root = src_root / repo_id
        dst_repo_root = dst_root / repo_id
        selected = episodes_by_repo.get(repo_id)

        if dst_repo_root.exists():
            logger.info("[%d/%d] skip existing repo: %s", idx, len(repo_ids), repo_id)
            skipped_existing += 1
            continue

        if selected is None:
            logger.info("[%d/%d] copy full repo (<=N demos): %s", idx, len(repo_ids), repo_id)
            kept_as_full_copy += 1
            if not cfg.dry_run:
                _copy_repo_tree(src_repo_root, dst_repo_root)
            continue

        logger.info(
            "[%d/%d] slice repo: %s (keep=%d episodes)",
            idx,
            len(repo_ids),
            repo_id,
            len(selected),
        )
        sliced_count += 1
        if cfg.dry_run:
            continue

        dataset = LeRobotDataset(repo_id=repo_id, root=src_repo_root)
        all_episode_ids = list(range(dataset.meta.total_episodes))
        selected_set = set(int(x) for x in selected)
        delete_ids = [ep for ep in all_episode_ids if ep not in selected_set]
        if not delete_ids:
            _copy_repo_tree(src_repo_root, dst_repo_root)
            kept_as_full_copy += 1
            sliced_count -= 1
            continue

        delete_episodes(
            dataset=dataset,
            episode_indices=delete_ids,
            output_dir=dst_repo_root,
            repo_id=repo_id,
        )

    logger.info("Done. total=%d, sliced=%d, full_copy=%d, skipped_existing=%d", len(repo_ids), sliced_count, kept_as_full_copy, skipped_existing)


def parse_args() -> MaterializeConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, default=MaterializeConfig.src_root)
    parser.add_argument("--dst_root", type=str, default=MaterializeConfig.dst_root)
    parser.add_argument("--split", type=str, default=MaterializeConfig.split)
    parser.add_argument("--subset_demos_per_task", type=int, default=MaterializeConfig.subset_demos_per_task)
    parser.add_argument("--subset_seed", type=int, default=MaterializeConfig.subset_seed)
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    return MaterializeConfig(
        src_root=args.src_root,
        dst_root=args.dst_root,
        split=args.split,
        subset_demos_per_task=args.subset_demos_per_task,
        subset_seed=args.subset_seed,
        overwrite_existing=args.overwrite_existing,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    run(parse_args())
