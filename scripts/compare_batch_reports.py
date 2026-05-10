#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.policies.groot_common import compare_tensor_reports


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_manifest", type=Path, required=True)
    p.add_argument("--phase2_report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--required_keys_only", action="store_true")
    p.add_argument("--ignore_batch_dim", action="store_true")
    p.add_argument("--exclude_keys", type=str, nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    p1 = json.loads(args.phase1_manifest.read_text())["batch_report"]
    p2 = json.loads(args.phase2_report.read_text())["tensor_report"]
    comparison = compare_tensor_reports(
        p1,
        p2,
        required_keys_only=args.required_keys_only,
        ignore_batch_dim=args.ignore_batch_dim,
        exclude_keys=set(args.exclude_keys),
    )

    report = {
        "phase1_manifest": str(args.phase1_manifest),
        "phase2_report": str(args.phase2_report),
        **comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"ok": True, "report": str(args.output)}))


if __name__ == "__main__":
    main()
