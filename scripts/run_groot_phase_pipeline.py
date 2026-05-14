#!/usr/bin/env python3
"""Run multi-stage GR00T training with checkpoint handoff.

Each stage is passed as JSON via --stage-json (repeatable), for example:

python scripts/run_groot_phase_pipeline.py \
  --stage-json '{"name":"phase2_rkd","script":"clvla/lerobot_cl/src/lerobot/scripts/train_groot_processed_rkd.py","cuda_visible_devices":"0","mode_arg":"policy.cl_v2_trainable_mode","mode":"processed_only","steps":120000,"save_freq":30000,"output_dir":"/home/seonho/groot_robocasa/processed_rkd_ckpt/phase2","log_path":"/home/seonho/groot_robocasa/processed_rkd_ckpt/phase2/train.log","extra_args":["--dataset.root=/home/seonho/groot_robocasa/robocasa_dataset/robocasa_v3_official_presliced","--data_split=pretrain","--num_workers=2","--policy.type=groot_processed_rkd","--policy.groot_pretrained_path=paragon7060/Robocasa_baseline","--policy.cl_v2_phase=phase2","--policy.cl_v2_student_repr=flatten","--policy.cl_v2_action_repr=raw_flatten","--policy.cl_v2_loss_weight=0.2","--policy.cl_v2_fm_loss_weight=0.01","--policy.lora_rank=0","--policy.tune_visual=false","--policy.tune_llm=false","--batch_size=64","--log_freq=100","--wandb.enable=true","--wandb.project=groot_processed_rkd_phase2","--wandb.entity=RwHlabs"]}' \
  --stage-json '{"name":"phase3_mgd","script":"clvla/lerobot_cl/src/lerobot/scripts/train_groot_mgd_robocasa.py","cuda_visible_devices":"1","mode_arg":"policy.mgd_trainable_mode","mode":"dit_core_only","steps":120000,"save_freq":30000,"output_dir":"/home/seonho/groot_robocasa/processed_rkd_ckpt/phase3","log_path":"/home/seonho/groot_robocasa/processed_rkd_ckpt/phase3/train.log","pretrained_from_prev":true,"pretrained_arg":"pretrained_path","extra_args":["--dataset.root=/home/seonho/groot_robocasa/robocasa_dataset/robocasa_human_v3","--data_split=pretrain","--num_workers=2","--policy.type=groot_processed_mgd","--policy.mgd_backprop_backbone=true","--policy.lora_rank=0","--policy.tune_visual=false","--policy.tune_llm=false","--batch_size=64","--log_freq=100","--wandb.enable=true","--wandb.project=groot_processed_mgd_phase3","--wandb.entity=RwHlabs"]}'

Or pass a YAML config:

python scripts/run_groot_phase_pipeline.py --config scripts/run_groot_phase_pipeline.example.yaml
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageSpec:
    name: str
    script: str
    output_dir: str
    steps: int | str
    save_value: int | str | None = None
    mode_arg: str | None = None
    mode: str | None = None
    extra_args: list[str] = field(default_factory=list)
    cuda_visible_devices: str | None = None
    log_path: str | None = None
    pretrained_from_prev: bool = False
    pretrained_arg: str = "pretrained_path"
    pretrained_value: str | None = None
    python_bin: str = "python"
    env: dict[str, str] = field(default_factory=dict)
    steps_arg: str = "steps"
    save_arg: str = "save_freq"
    output_dir_arg: str = "output_dir"
    pipeline: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StageSpec":
        required = ["name", "script", "output_dir", "steps"]
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f"Missing required keys in stage: {missing}")
        extra_args = raw.get("extra_args", [])
        if not isinstance(extra_args, list) or any(not isinstance(x, str) for x in extra_args):
            raise ValueError("extra_args must be a list[str]")
        env = raw.get("env", {})
        if not isinstance(env, dict):
            raise ValueError("env must be a mapping[str, str]")
        if ("mode_arg" in raw) != ("mode" in raw):
            raise ValueError("mode_arg and mode must be provided together.")
        return cls(
            name=_expand_pathish(str(raw["name"])),
            script=_expand_pathish(str(raw["script"])),
            output_dir=_expand_pathish(str(raw["output_dir"])),
            steps=(str(raw["steps"]) if isinstance(raw["steps"], str) else int(raw["steps"])),
            save_value=(
                None
                if raw.get("save_value") is None and raw.get("save_freq") is None
                else (
                    str(raw.get("save_value", raw.get("save_freq")))
                    if isinstance(raw.get("save_value", raw.get("save_freq")), str)
                    else int(raw.get("save_value", raw.get("save_freq")))
                )
            ),
            mode_arg=(None if raw.get("mode_arg") is None else str(raw["mode_arg"])),
            mode=(None if raw.get("mode") is None else str(raw["mode"])),
            extra_args=[_expand_pathish(arg) for arg in extra_args],
            cuda_visible_devices=(None if raw.get("cuda_visible_devices") is None else str(raw["cuda_visible_devices"])),
            log_path=(None if raw.get("log_path") is None else _expand_pathish(str(raw["log_path"]))),
            pretrained_from_prev=bool(raw.get("pretrained_from_prev", False)),
            pretrained_arg=str(raw.get("pretrained_arg", "pretrained_path")),
            pretrained_value=(None if raw.get("pretrained_value") is None else _expand_pathish(str(raw["pretrained_value"]))),
            python_bin=str(raw.get("python_bin", "python")),
            env={str(k): _expand_pathish(str(v)) for k, v in env.items()},
            steps_arg=str(raw.get("steps_arg", "steps")),
            save_arg=str(raw.get("save_arg", "save_freq")),
            output_dir_arg=str(raw.get("output_dir_arg", "output_dir")),
            pipeline=(None if raw.get("pipeline") is None else str(raw["pipeline"])),
        )


def _expand_pathish(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _template(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for key, replacement in variables.items():
            result = result.replace("{{" + key + "}}", replacement)
        return result
    if isinstance(value, list):
        return [_template(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _template(item, variables) for key, item in value.items()}
    return value


def _merge_override(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "extra_args":
            base_extra = merged.get("extra_args", [])
            if not isinstance(base_extra, list) or not isinstance(value, list):
                raise ValueError("extra_args overrides must be list[str]")
            merged["extra_args"] = list(base_extra) + list(value)
        elif key == "env":
            base_env = merged.get("env", {})
            if not isinstance(base_env, dict) or not isinstance(value, dict):
                raise ValueError("env overrides must be mappings")
            merged["env"] = {**base_env, **value}
        else:
            merged[key] = value
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-stage GR00T training pipeline")
    p.add_argument(
        "--config",
        type=str,
        help="YAML config path for multi-stage pipeline.",
    )
    p.add_argument(
        "--stage-json",
        action="append",
        default=[],
        help="JSON object for one stage (repeatable).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands only.",
    )
    p.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Run only the named YAML variant. Repeat to run multiple variants.",
    )
    return p.parse_args()


def _validate_arg_conflicts(args: argparse.Namespace) -> None:
    if args.config and args.stage_json:
        raise ValueError("Use either --config or --stage-json, not both.")
    if not args.config and not args.stage_json:
        raise ValueError("Either --config or at least one --stage-json is required.")


def _parse_stage_json_specs(stage_json_items: list[str]) -> list[StageSpec]:
    specs: list[StageSpec] = []
    for idx, item in enumerate(stage_json_items, start=1):
        try:
            raw = json.loads(item)
        except json.JSONDecodeError as e:
            raise ValueError(f"--stage-json #{idx} is not valid JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(f"--stage-json #{idx} must be a JSON object.")
        specs.append(StageSpec.from_dict(raw))
    return specs


def _merge_stage_dict(defaults: dict[str, Any], raw_stage: dict[str, Any], common_args: list[str]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(raw_stage)

    default_extra = defaults.get("extra_args", [])
    stage_extra = raw_stage.get("extra_args", [])
    if default_extra and (not isinstance(default_extra, list) or any(not isinstance(x, str) for x in default_extra)):
        raise ValueError("stage_defaults.extra_args must be a list[str]")
    if stage_extra and (not isinstance(stage_extra, list) or any(not isinstance(x, str) for x in stage_extra)):
        raise ValueError("stage.extra_args must be a list[str]")
    merged["extra_args"] = [_expand_pathish(arg) for arg in (list(common_args) + list(default_extra) + list(stage_extra))]

    default_env = defaults.get("env", {})
    stage_env = raw_stage.get("env", {})
    if default_env and not isinstance(default_env, dict):
        raise ValueError("stage_defaults.env must be a mapping")
    if stage_env and not isinstance(stage_env, dict):
        raise ValueError("stage.env must be a mapping")
    merged["env"] = {str(k): _expand_pathish(str(v)) for k, v in default_env.items()}
    merged["env"].update({str(k): _expand_pathish(str(v)) for k, v in stage_env.items()})
    return merged


def _parse_config_specs(config_path: str) -> list[StageSpec]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping.")

    raw_stages = data.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("YAML config must contain non-empty 'stages' list.")

    base_variables = data.get("variables", {})
    if base_variables and not isinstance(base_variables, dict):
        raise ValueError("variables must be a mapping.")
    base_variables = {str(k): _expand_pathish(str(v)) for k, v in base_variables.items()}

    common_args = data.get("common_args", [])
    if common_args and (not isinstance(common_args, list) or any(not isinstance(x, str) for x in common_args)):
        raise ValueError("common_args must be a list[str]")

    stage_defaults = data.get("stage_defaults", {})
    if stage_defaults and not isinstance(stage_defaults, dict):
        raise ValueError("stage_defaults must be a mapping.")

    specs: list[StageSpec] = []
    variants = data.get("variants")
    if variants is None:
        variants = [{"name": "default"}]
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants must be a non-empty list when provided.")

    for variant_idx, raw_variant in enumerate(variants, start=1):
        if not isinstance(raw_variant, dict):
            raise ValueError(f"Variant #{variant_idx} must be a mapping.")
        variant_name = str(raw_variant.get("name", f"variant{variant_idx}"))
        variant_vars = raw_variant.get("variables", {})
        if variant_vars and not isinstance(variant_vars, dict):
            raise ValueError(f"Variant '{variant_name}' variables must be a mapping.")
        variables = {
            **base_variables,
            "variant": variant_name,
            **{str(k): _expand_pathish(str(v)) for k, v in variant_vars.items()},
        }

        variant_common_args = raw_variant.get("common_args", [])
        if variant_common_args and (
            not isinstance(variant_common_args, list) or any(not isinstance(x, str) for x in variant_common_args)
        ):
            raise ValueError(f"Variant '{variant_name}' common_args must be a list[str]")

        stage_overrides = raw_variant.get("stage_overrides", {})
        if stage_overrides and not isinstance(stage_overrides, dict):
            raise ValueError(f"Variant '{variant_name}' stage_overrides must be a mapping.")

        templated_defaults = _template(deepcopy(stage_defaults), variables)
        templated_common_args = _template(list(common_args) + list(variant_common_args), variables)

        for idx, raw_stage in enumerate(raw_stages, start=1):
            if not isinstance(raw_stage, dict):
                raise ValueError(f"Stage #{idx} in YAML must be a mapping.")
            stage_name = str(raw_stage.get("name", f"stage{idx}"))
            stage = deepcopy(raw_stage)
            override = stage_overrides.get(stage_name, {})
            if override:
                if not isinstance(override, dict):
                    raise ValueError(f"Variant '{variant_name}' override for '{stage_name}' must be a mapping.")
                stage = _merge_override(stage, override)
            stage = _template(stage, variables)
            stage.setdefault("pipeline", variant_name)
            merged = _merge_stage_dict(templated_defaults, stage, templated_common_args)
            specs.append(StageSpec.from_dict(merged))
    return specs


def parse_stage_specs(args: argparse.Namespace) -> list[StageSpec]:
    _validate_arg_conflicts(args)
    if args.config:
        specs = _parse_config_specs(args.config)
    else:
        if args.variant:
            raise ValueError("--variant can only be used with --config.")
        specs = _parse_stage_json_specs(args.stage_json)
    if args.variant:
        variants = set(args.variant)
        specs = [stage for stage in specs if stage.pipeline in variants]
        if not specs:
            raise ValueError(f"No stages matched requested variants: {sorted(variants)}")
    return specs


def build_stage_cmd(stage: StageSpec, prev_ckpt: str | None) -> list[str]:
    cmd = [stage.python_bin, stage.script]
    cmd.extend(stage.extra_args)
    if stage.mode_arg is not None and stage.mode is not None:
        cmd.append(f"--{stage.mode_arg}={stage.mode}")
    cmd.append(f"--{stage.steps_arg}={stage.steps}")
    if stage.save_value is not None:
        cmd.append(f"--{stage.save_arg}={stage.save_value}")
    cmd.append(f"--{stage.output_dir_arg}={stage.output_dir}")
    if stage.pretrained_from_prev:
        if not prev_ckpt:
            raise RuntimeError(
                f"Stage '{stage.name}' requested pretrained_from_prev=true, "
                "but no previous checkpoint exists."
            )
        cmd.append(f"--{stage.pretrained_arg}={prev_ckpt}")
    elif stage.pretrained_value:
        cmd.append(f"--{stage.pretrained_arg}={stage.pretrained_value}")
    return cmd


def resolve_last_ckpt(output_dir: str) -> str:
    last_link = Path(output_dir) / "checkpoints" / "last"
    if not last_link.exists():
        raise FileNotFoundError(f"Missing last checkpoint link: {last_link}")
    return str(last_link.resolve())


def stream_run(cmd: list[str], env: dict[str, str], log_path: str | None) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    log_file = None
    try:
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "w", encoding="utf-8")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            if log_file:
                log_file.write(line)
        return proc.wait()
    finally:
        if log_file:
            log_file.close()


def main() -> None:
    args = parse_args()
    specs = parse_stage_specs(args)

    prev_ckpt: str | None = None
    prev_pipeline: str | None = None
    for idx, stage in enumerate(specs, start=1):
        if stage.pipeline != prev_pipeline:
            prev_ckpt = None
            prev_pipeline = stage.pipeline
        cmd = build_stage_cmd(stage, prev_ckpt)
        env = os.environ.copy()
        if stage.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = stage.cuda_visible_devices
        env.update(stage.env)

        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        print(f"\n[stage {idx}/{len(specs)}] {stage.name}")
        if stage.pipeline is not None:
            print(f"variant={stage.pipeline}")
        if stage.cuda_visible_devices is not None:
            print(f"CUDA_VISIBLE_DEVICES={stage.cuda_visible_devices}")
        print(cmd_str)

        if args.dry_run:
            prev_ckpt = f"{stage.output_dir}/checkpoints/last"
            continue

        Path(stage.output_dir).mkdir(parents=True, exist_ok=True)
        rc = stream_run(cmd, env, stage.log_path)
        if rc != 0:
            raise RuntimeError(f"Stage '{stage.name}' failed with exit code {rc}")

        prev_ckpt = resolve_last_ckpt(stage.output_dir)
        print(f"[stage done] {stage.name} -> last checkpoint: {prev_ckpt}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
