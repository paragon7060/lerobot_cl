#!/usr/bin/env python
"""Pre-compute hard-negative pairs for ContrastiveLeRobotDataset.

Output format (JSON):
{
  "<repo_id>": {
    "<episode_idx>_<frame_idx>": {
      "neg_episode_idx": int,
      "neg_frame_idx": int
    },
    ...
  }
}

Usage:
    python scripts/precompute_negative_pairs.py \\
        --repo_id lerobot/my_dataset \\
        --output_path negative_pairs.json \\
        [--root /path/to/local/dataset]
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def build_negative_pairs(dataset: LeRobotDataset) -> dict:
    """Build a hard-negative mapping for all (episode, frame) pairs.

    Matching strategy:
    - Group episodes by task_index.
    - For each anchor (ep_i, frame_j), randomly pick a different episode with the same task.
    - Match by relative timestep ratio: neg_frame = round(ratio * neg_ep_len)

    Returns:
        dict mapping "<ep_idx>_<frame_idx>" → {"neg_episode_idx": int, "neg_frame_idx": int}
    """
    episodes_meta = dataset.meta.episodes

    # Build task_index → [ep_idx, ...] mapping using the hf_dataset
    dataset._ensure_hf_dataset_loaded()
    task_by_episode: dict[int, int] = {}
    for ep_idx in range(dataset.meta.total_episodes):
        ep = episodes_meta[ep_idx]
        from_idx = ep["dataset_from_index"]
        row = dataset.hf_dataset[int(from_idx)]
        task_idx = int(row["task_index"].item()) if hasattr(row["task_index"], "item") else int(row["task_index"])
        task_by_episode[ep_idx] = task_idx

    task_to_episodes: dict[int, list[int]] = defaultdict(list)
    for ep_idx, task_idx in task_by_episode.items():
        task_to_episodes[task_idx].append(ep_idx)

    unique_tasks = set(task_to_episodes.keys())
    if len(unique_tasks) == 1:
        logger.warning(
            "Dataset has only one unique task_index. Hard-negative pairing is meaningless. "
            "Consider using contrastive_fallback_to_in_batch=True instead."
        )

    pairs: dict[str, dict] = {}

    for ep_idx, ep_meta in episodes_meta.items():
        ep_len = ep_meta["dataset_to_index"] - ep_meta["dataset_from_index"]
        task_idx = task_by_episode[ep_idx]
        candidates = [e for e in task_to_episodes[task_idx] if e != ep_idx]

        if not candidates:
            logger.debug("Episode %d has no negative candidates (only episode with task %d).", ep_idx, task_idx)
            continue

        neg_ep_idx = random.choice(candidates)
        neg_ep_meta = episodes_meta[neg_ep_idx]
        neg_ep_len = neg_ep_meta["dataset_to_index"] - neg_ep_meta["dataset_from_index"]

        for frame_idx in range(ep_len):
            ratio = frame_idx / max(ep_len - 1, 1)
            neg_frame_idx = min(round(ratio * (neg_ep_len - 1)), neg_ep_len - 1)
            key = f"{ep_idx}_{frame_idx}"
            pairs[key] = {
                "neg_episode_idx": neg_ep_idx,
                "neg_frame_idx": neg_frame_idx,
            }

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Pre-compute hard-negative pairs for contrastive training.")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace dataset repo ID.")
    parser.add_argument("--output_path", type=str, default="negative_pairs.json")
    parser.add_argument("--root", type=str, default=None, help="Local dataset root (optional).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    random.seed(args.seed)

    logger.info("Loading dataset %s ...", args.repo_id)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    logger.info("Building negative pairs ...")
    pairs = build_negative_pairs(dataset)
    logger.info("Built %d pairs.", len(pairs))

    output = {args.repo_id: pairs}
    output_path = Path(args.output_path)
    with open(output_path, "w") as f:
        json.dump(output, f)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
