#!/usr/bin/env python
"""Pre-compute hard-negative pairs for ContrastiveLeRobotDataset using a custom task mapping.

Usage:
    python scripts/precompute_negative_pairs.py \
        --repo_id lerobot/my_dataset \
        --task_mapping task_mapping.json \
        --output_path negative_pairs.json \
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

def build_negative_pairs(dataset: LeRobotDataset, task_mapping: dict) -> dict:
    episodes_meta = dataset.meta.episodes
    dataset._ensure_hf_dataset_loaded()

    # 1. 태스크 이름(String)을 내부 task_index(Int)로 자동 연결해주는 사전 생성
    # tasks DataFrame: index=task_name(str), column="task_index"(int)
    # iloc 위치(positional)가 아닌 실제 task_index 컬럼값을 사용해야
    # task_by_episode / task_to_episodes와 키 체계가 일치함
    task_name_to_idx = {
        task_name: int(row["task_index"])
        for task_name, row in dataset.meta.tasks.iterrows()
    }

    # 2. episode별 task_index 수집 및 task_index별 episode 그룹화
    task_by_episode = {}
    task_to_episodes = defaultdict(list)

    for ep_idx in range(dataset.meta.total_episodes):
        ep = episodes_meta[ep_idx]
        from_idx = ep["dataset_from_index"]
        row = dataset.hf_dataset[int(from_idx)]
        task_idx = int(row["task_index"].item()) if hasattr(row["task_index"], "item") else int(row["task_index"])
        
        task_by_episode[ep_idx] = task_idx
        task_to_episodes[task_idx].append(ep_idx)

    neg_task_map = {}
    for task_str, neg_task_strs in task_mapping.items():
        if task_str not in task_name_to_idx:
            logger.warning("Task '%s' not found in dataset metadata. Skipping.", task_str)
            continue
        
        k_idx = task_name_to_idx[task_str]
        v_indices = []
        for v_str in neg_task_strs:
            if v_str in task_name_to_idx:
                v_indices.append(task_name_to_idx[v_str])
            else:
                logger.warning("Negative task '%s' not found in dataset. Skipping.", v_str)
        
        neg_task_map[k_idx] = v_indices

    pairs: dict[str, dict] = {}

    # 4. 지정된 매핑에 따라 Hard Negative 매칭
    # episodes_meta는 HuggingFace datasets.Dataset — .items() 없음, 정수 인덱싱 사용
    for ep_idx in range(dataset.meta.total_episodes):
        ep_meta = episodes_meta[ep_idx]
        task_idx = task_by_episode[ep_idx]
        ep_len = int(ep_meta["dataset_to_index"]) - int(ep_meta["dataset_from_index"])

        # 지정된 Negative task_index 리스트 가져오기
        allowed_neg_tasks = neg_task_map.get(task_idx, [])
        
        candidates = []
        for neg_t_idx in allowed_neg_tasks:
            candidates.extend(task_to_episodes.get(neg_t_idx, []))

        if not candidates:
            logger.warning(
                "Episode %d (task_index=%d) has no negative candidates. "
                "Check that the task mapping covers this task_index.",
                ep_idx, task_idx,
            )
            continue

        # Hard negative 에피소드 1개 랜덤 선택
        neg_ep_idx = random.choice(candidates)
        neg_ep_meta = episodes_meta[neg_ep_idx]
        neg_ep_len = int(neg_ep_meta["dataset_to_index"]) - int(neg_ep_meta["dataset_from_index"])

        # 5. 프레임 비율 매칭 (Time-alignment)
        for frame_idx in range(ep_len):
            ratio = frame_idx / max(ep_len - 1, 1)
            neg_frame_idx = min(round(ratio * max(neg_ep_len - 1, 1)), neg_ep_len - 1)
            
            key = f"{ep_idx}_{frame_idx}"
            pairs[key] = {
                "neg_episode_idx": neg_ep_idx,
                "neg_frame_idx": neg_frame_idx,
            }

    return pairs
    
def main():
    parser = argparse.ArgumentParser(description="Pre-compute hard-negative pairs using a task mapping JSON.")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace dataset repo ID.")
    parser.add_argument("--task_mapping", type=str, required=True, help="Path to JSON file defining negative task_index pairs.")
    parser.add_argument("--output_path", type=str, default="negative_pairs.json")
    parser.add_argument("--root", type=str, default=None, help="Local dataset root (optional).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    random.seed(args.seed)

    # 매핑 파일 로드
    logger.info("Loading task mapping from %s ...", args.task_mapping)
    with open(args.task_mapping, "r") as f:
        task_mapping = json.load(f)

    logger.info("Loading dataset %s ...", args.repo_id)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    logger.info("Building negative pairs based on mapping ...")
    pairs = build_negative_pairs(dataset, task_mapping)
    logger.info("Built %d pairs.", len(pairs))

    output = {args.repo_id: pairs}
    output_path = Path(args.output_path)
    with open(output_path, "w") as f:
        json.dump(output, f)
    logger.info("Saved to %s", output_path)

if __name__ == "__main__":
    main()