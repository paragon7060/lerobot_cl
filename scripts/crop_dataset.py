#!/usr/bin/env python3
"""Crop action and observation.state dimensions in a LeRobot dataset.

Usage:
    python scripts/crop_dataset.py \
        --dataset-dir /path/to/dataset \
        --action-size 7 \
        --state-size 7 \
        [--output-dir /path/to/output] \
        [--dry-run]
"""
import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _crop_list_column(col: pa.ChunkedArray, size: int) -> pa.Array:
    return pa.array([row[:size] for row in col.to_pylist()], type=pa.list_(pa.float32()))


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def crop_data_parquets(out_dir: Path, action_size: int | None, state_size: int | None) -> None:
    paths = sorted((out_dir / "data").glob("*/*.parquet"))
    if not paths:
        print("  [data] parquet 파일 없음, 건너뜀")
        return

    for pq_path in paths:
        table = pq.read_table(pq_path)
        columns: dict[str, pa.Array] = {}
        for col_name in table.column_names:
            col = table.column(col_name)
            if col_name == "action" and action_size is not None:
                col = _crop_list_column(col, action_size)
            elif col_name == "observation.state" and state_size is not None:
                col = _crop_list_column(col, state_size)
            columns[col_name] = col
        pq.write_table(pa.table(columns), pq_path, compression="snappy")
        print(f"  [data] {pq_path.relative_to(out_dir)}")


def crop_info_json(out_dir: Path, action_size: int | None, state_size: int | None) -> None:
    info_path = out_dir / "meta/info.json"
    info = _load_json(info_path)
    features = info["features"]

    if "action" in features and action_size is not None:
        features["action"]["shape"] = [action_size]
        if features["action"].get("names"):
            features["action"]["names"] = features["action"]["names"][:action_size]

    if "observation.state" in features and state_size is not None:
        features["observation.state"]["shape"] = [state_size]
        if features["observation.state"].get("names"):
            features["observation.state"]["names"] = features["observation.state"]["names"][:state_size]

    _write_json(info, info_path)
    print(f"  [meta] info.json 업데이트")


_STAT_KEYS = ["mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99"]


def crop_stats_json(out_dir: Path, action_size: int | None, state_size: int | None) -> None:
    stats_path = out_dir / "meta/stats.json"
    if not stats_path.exists():
        print("  [meta] stats.json 없음, 건너뜀")
        return

    stats = _load_json(stats_path)

    for stat_key in _STAT_KEYS:
        if "action" in stats and action_size is not None:
            if stat_key in stats["action"]:
                stats["action"][stat_key] = stats["action"][stat_key][:action_size]
        if "observation.state" in stats and state_size is not None:
            if stat_key in stats["observation.state"]:
                stats["observation.state"][stat_key] = stats["observation.state"][stat_key][:state_size]

    _write_json(stats, stats_path)
    print(f"  [meta] stats.json 업데이트")


def crop_episode_parquets(out_dir: Path, action_size: int | None, state_size: int | None) -> None:
    paths = sorted((out_dir / "meta/episodes").glob("*/*.parquet"))
    if not paths:
        print("  [episodes] parquet 파일 없음, 건너뜀")
        return

    for pq_path in paths:
        table = pq.read_table(pq_path)
        columns: dict[str, pa.Array] = {}
        for col_name in table.column_names:
            col = table.column(col_name)
            if col_name.startswith("stats/action/") and action_size is not None:
                col = _crop_list_column(col, action_size)
            elif col_name.startswith("stats/observation.state/") and state_size is not None:
                col = _crop_list_column(col, state_size)
            columns[col_name] = col
        pq.write_table(pa.table(columns), pq_path, compression="snappy")
        print(f"  [episodes] {pq_path.relative_to(out_dir)}")


def _validate_and_print(dataset_dir: Path, action_size: int | None, state_size: int | None) -> None:
    info = _load_json(dataset_dir / "meta/info.json")
    features = info["features"]

    print("\n현재 features:")
    for key in ["action", "observation.state"]:
        if key in features:
            ft = features[key]
            print(f"  {key}: shape={ft['shape']}, names={ft.get('names')}")

    if "action" in features and action_size is not None:
        cur = features["action"]["shape"][0]
        if action_size > cur:
            raise ValueError(f"action-size {action_size} > 현재 크기 {cur}")
        print(f"\naction: {cur} → {action_size}")

    if "observation.state" in features and state_size is not None:
        cur = features["observation.state"]["shape"][0]
        if state_size > cur:
            raise ValueError(f"state-size {state_size} > 현재 크기 {cur}")
        print(f"observation.state: {cur} → {state_size}")


def crop_dataset(
    dataset_dir: Path,
    output_dir: Path | None,
    action_size: int | None,
    state_size: int | None,
    dry_run: bool,
) -> None:
    if action_size is None and state_size is None:
        raise ValueError("--action-size 또는 --state-size 중 하나 이상을 지정해야 합니다.")

    _validate_and_print(dataset_dir, action_size, state_size)

    if dry_run:
        print("\n[dry-run] 실제 수정 없이 종료합니다.")
        return

    if output_dir is None:
        out_dir = dataset_dir
        print(f"\nin-place 수정: {out_dir}")
    else:
        out_dir = output_dir
        print(f"\n{dataset_dir} → {out_dir} 복사 중...")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(dataset_dir, out_dir)
        print("복사 완료")

    print("\n[1/4] data parquet crop...")
    crop_data_parquets(out_dir, action_size, state_size)

    print("\n[2/4] info.json crop...")
    crop_info_json(out_dir, action_size, state_size)

    print("\n[3/4] stats.json crop...")
    crop_stats_json(out_dir, action_size, state_size)

    print("\n[4/4] episodes parquet crop...")
    crop_episode_parquets(out_dir, action_size, state_size)

    print(f"\n완료: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeRobot 데이터셋의 action/observation.state를 지정한 크기로 crop합니다."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="원본 데이터셋 경로")
    parser.add_argument("--action-size", type=int, default=None, help="action을 앞 N개 차원으로 crop")
    parser.add_argument("--state-size", type=int, default=None, help="observation.state를 앞 N개 차원으로 crop")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="출력 경로 (생략 시 in-place 수정)",
    )
    parser.add_argument("--dry-run", action="store_true", help="수정 없이 현재 shape/names만 출력")
    args = parser.parse_args()

    crop_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        action_size=args.action_size,
        state_size=args.state_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
