#!/usr/bin/env python
"""MultiLeRobotDataset dry-run 검증 스크립트

학습 전에 아래 두 가지를 확인한다:
  1. dataset_index 분포 — 배치 안에 각 데이터셋이 고루 섞이는지
  2. feature 누락 여부 — disabled_features로 날아간 키가 없는지

데이터셋 구조:
  {root}/
    robocasa_pretrain_human_atomic/task_0001/meta/info.json  ← 각 task가 독립 LeRobot 데이터셋
    robocasa_pretrain_human_atomic/task_0002/...
    robocasa_pretrain_human_composite/task_0001/...
    ...

실행:
  python scripts/check_multi_dataset.py \
      --root /home/seonho/slicing_robocasa_human_v3 \
      --batch_size 32 \
      --num_batches 10
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.multi_dataset import MultiLeRobotDataset

TOP_LEVEL_DIRS = [
    "robocasa_pretrain_human_atomic",
    "robocasa_pretrain_human_composite",
    #"robocasa_target_human_atomic",
    #"robocasa_target_human_composite",
]

REQUIRED_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.state",
    "action",
    "dataset_index",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/home/seonho/slicing_robocasa_human_v3")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_batches", type=int, default=10,
                   help="검사할 배치 수. 많을수록 분포가 정확해짐")
    p.add_argument("--num_workers", type=int, default=0,
                   help="0이면 메인 프로세스에서 직접 로드 (디버깅에 적합)")
    return p.parse_args()


def get_task_repo_ids(root: Path) -> tuple[list[str], dict[str, list[str]]]:
    """root 아래 각 top-level dir의 task_XXXX 서브디렉토리를 열거한다.

    Returns:
        repo_ids: ["robocasa_pretrain_human_atomic/task_0001", ...] 형식의 전체 목록
        group_map: {top_level_dir: [repo_id, ...]} — 그룹별 집계용
    """
    repo_ids = []
    group_map: dict[str, list[str]] = {}
    for top_dir in TOP_LEVEL_DIRS:
        top_path = root / top_dir
        if not top_path.exists():
            print(f"  ⚠️  경로 없음: {top_path}")
            group_map[top_dir] = []
            continue
        tasks = sorted(
            p.name for p in top_path.iterdir()
            if p.is_dir() and p.name.startswith("task_")
            and (p / "meta" / "info.json").exists()
            and (p / "meta" / "episodes").exists()
        )
        skipped = sum(
            1 for p in top_path.iterdir()
            if p.is_dir() and p.name.startswith("task_")
            and not ((p / "meta" / "info.json").exists() and (p / "meta" / "episodes").exists())
        )
        if skipped:
            print(f"  ⚠️  {top_dir}: {skipped}개 task에 meta/info.json 없음 (아직 다운로드 안 됨) → 건너뜀")
        group_ids = [f"{top_dir}/{t}" for t in tasks]
        repo_ids.extend(group_ids)
        group_map[top_dir] = group_ids
    return repo_ids, group_map


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    args = parse_args()
    root = Path(args.root)

    # ── 0. task 경로 열거 ────────────────────────────────────────
    section("0. task 디렉토리 열거")
    repo_ids, group_map = get_task_repo_ids(root)
    for top_dir, gids in group_map.items():
        print(f"  {top_dir:<45}  {len(gids)} tasks")
    print(f"\n  총 {len(repo_ids)}개 task 데이터셋")

    # ── 1. 샘플 메타 확인 (첫 번째 task만) ─────────────────────
    section("1. 샘플 메타 확인 (각 그룹 첫 번째 task)")
    for top_dir, gids in group_map.items():
        if not gids:
            continue
        rid = gids[0]
        try:
            meta = LeRobotDatasetMetadata(repo_id=rid, root=root / rid)
            print(f"  [{rid}]  fps={meta.fps}  episodes={meta.total_episodes}  frames={meta.total_frames}")
        except Exception as e:
            print(f"  [{rid}]  ERROR: {e}")

    # ── 2. MultiLeRobotDataset 생성 ────────────────────────────
    section(f"2. MultiLeRobotDataset 생성 ({len(repo_ids)}개 task)")
    print("  (처음 로드 시 stats 집계로 시간이 걸릴 수 있음)")
    try:
        dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            root=root,
            video_backend="pyav",
        )
    except Exception as e:
        print(f"  ERROR 데이터셋 생성 실패: {e}")
        return

    print(f"  total frames   : {dataset.num_frames:,}")
    print(f"  total episodes : {dataset.num_episodes:,}")
    print(f"  fps            : {dataset.fps}")

    # 그룹별(top_level_dir) 프레임 수 및 비율
    print("\n  [그룹별 프레임 수 및 비율]")
    total = dataset.num_frames
    offset = 0
    for top_dir, gids in group_map.items():
        group_frames = sum(dataset._datasets[offset + i].num_frames for i in range(len(gids)))
        ratio = group_frames / total * 100 if total > 0 else 0
        print(f"    {top_dir:<45}  {group_frames:>8,} frames  ({ratio:.1f}%)  ({len(gids)} tasks)")
        offset += len(gids)

    # ── 3. Feature 누락 확인 ─────────────────────────────────────
    section("3. Feature 확인")
    print(f"  활성 features ({len(dataset.features)}):")
    for k in sorted(dataset.features.keys()):
        print(f"    {k}")

    if dataset.disabled_features:
        print(f"\n  ⚠️  disabled_features (공통 아님 → 누락됨) ({len(dataset.disabled_features)}):")
        for k in sorted(dataset.disabled_features):
            print(f"    {k}")
    else:
        print("\n  ✓  disabled_features 없음 — 모든 키 공통")

    print("\n  [필수 키 체크]")
    all_ok = True
    for key in REQUIRED_KEYS:
        present = key in dataset.features or key == "dataset_index"  # dataset_index는 __getitem__에서 추가
        status = "✓" if present else "✗ MISSING"
        if not present:
            all_ok = False
        print(f"    {status}  {key}")
    if not all_ok:
        print("\n  ⚠️  필수 키 누락 — 학습 전 데이터셋 확인 필요")

    # ── 4. 첫 번째 샘플 shape 확인 ──────────────────────────────
    section("4. 첫 번째 샘플 shape")
    try:
        sample = dataset[0]
        for k, v in sorted(sample.items()):
            shape = v.shape if isinstance(v, torch.Tensor) else type(v).__name__
            print(f"  {k:<50}  {shape}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 5. DataLoader 배치 dataset_index 분포 ───────────────────
    section(f"5. dataset_index 분포 (batch_size={args.batch_size}, batches={args.num_batches})")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    counter = Counter()
    try:
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            indices = batch["dataset_index"].tolist()
            counter.update(indices)
    except Exception as e:
        print(f"  ERROR DataLoader 실패: {e}")
        return

    total_samples = sum(counter.values())
    print(f"\n  총 {total_samples}개 샘플 ({args.num_batches}개 배치) 분석 결과:")

    # task 단위 분포
    for idx in sorted(counter.keys()):
        rid = repo_ids[idx] if idx < len(repo_ids) else f"unknown({idx})"
        count = counter[idx]
        ratio = count / total_samples * 100
        print(f"  [{idx:>3}] {rid:<55}  {count:>4}  ({ratio:4.1f}%)")

    # 그룹(top_level_dir) 단위로 집계
    print(f"\n  [그룹별 집계]")
    offset = 0
    for top_dir, gids in group_map.items():
        n = len(gids)
        group_count = sum(counter.get(offset + i, 0) for i in range(n))
        ratio = group_count / total_samples * 100 if total_samples > 0 else 0
        bar = "█" * int(ratio / 2)
        print(f"  {top_dir:<45}  {group_count:>5}  ({ratio:5.1f}%)  {bar}")
        offset += n

    expected_ratio = 100.0 / len(TOP_LEVEL_DIRS)
    offset = 0
    group_ratios = []
    for top_dir, gids in group_map.items():
        group_count = sum(counter.get(offset + i, 0) for i in range(len(gids)))
        group_ratios.append(group_count / total_samples * 100 if total_samples > 0 else 0)
        offset += len(gids)
    max_deviation = max(abs(r - expected_ratio) for r in group_ratios)
    print(f"\n  그룹 균등 분포 기대값: {expected_ratio:.1f}%  |  최대 편차: {max_deviation:.1f}%")
    if max_deviation > 20:
        print("  ⚠️  그룹 간 편차가 큼 — 데이터셋 크기 불균형 (WeightedRandomSampler 고려)")
    else:
        print("  ✓  편차 허용 범위 내")

    section("완료")
    print("  문제 없으면 train_groot_multi.py 실행하세요.\n")


if __name__ == "__main__":
    main()
