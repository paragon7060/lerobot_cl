#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.transforms import DEFAULT_EAGLE_PATH, build_eagle_processor, collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--data_config", type=str, default="panda_omron")
    p.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    p.add_argument("--video_backend", type=str, default="opencv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_config_cls = DATA_CONFIG_MAP[args.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    ds = LeRobotSingleDataset(
        dataset_path=str(args.dataset_path),
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        video_backend=args.video_backend,
        filter_key=None,
    )

    sample = ds[args.sample_index]
    eagle_processor = build_eagle_processor(DEFAULT_EAGLE_PATH)
    batch = collate([sample], eagle_processor=eagle_processor)
    report = {
        "dataset_path": str(args.dataset_path),
        "sample_index": int(args.sample_index),
        "keys": sorted(list(batch.keys())),
        "tensor_report": {},
    }

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            report["tensor_report"][k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "device": str(v.device),
            }

    out = args.output_dir / "phase2_reference_batch_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"ok": True, "report": str(out)}))


if __name__ == "__main__":
    main()
