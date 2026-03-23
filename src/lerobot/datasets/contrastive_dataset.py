import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class ContrastiveLeRobotDataset(LeRobotDataset):
    """LeRobotDataset subclass that appends a `negative_action` field to each sample.

    If `negative_pairs_path` is provided, the pre-computed hard-negative mapping is used.
    Otherwise `negative_action` is set to None and the policy falls back to in-batch negatives.
    """

    def __init__(self, *args, negative_pairs_path: str | Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._negative_pairs: dict | None = None
        if negative_pairs_path is not None:
            with open(negative_pairs_path) as f:
                data = json.load(f)
            repo_id = self.repo_id
            self._negative_pairs = data.get(repo_id, data)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)

        if self._negative_pairs is None:
            item["negative_action"] = None
            return item

        ep_idx = int(item["episode_index"].item())
        fr_idx = int(item["frame_index"].item())
        key = f"{ep_idx}_{fr_idx}"

        pair = self._negative_pairs.get(key)
        if pair is None:
            item["negative_action"] = None
            return item

        neg_global_idx = self._ep_frame_to_global_idx(pair["neg_episode_idx"], pair["neg_frame_idx"])
        neg_item = super().__getitem__(neg_global_idx)
        item["negative_action"] = neg_item["action"]
        return item

    def _ep_frame_to_global_idx(self, ep_idx: int, frame_idx: int) -> int:
        ep_start = self.meta.episodes[ep_idx]["dataset_from_index"]
        return int(ep_start) + frame_idx
