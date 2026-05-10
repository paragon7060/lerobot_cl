#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors


def collect_repo_ids(root: Path, max_tasks: int) -> list[str]:
    repo_ids: list[str] = []
    for top_dir in ("robocasa_pretrain_human_atomic", "robocasa_target_human_atomic"):
        top_path = root / top_dir
        if not top_path.exists():
            continue
        for p in sorted(top_path.iterdir()):
            if (
                p.is_dir()
                and p.name.startswith("task_")
                and (p / "meta" / "info.json").exists()
                and (p / "meta" / "episodes").exists()
            ):
                repo_ids.append(f"{top_dir}/{p.name}")
                if len(repo_ids) >= max_tasks:
                    return repo_ids
    return repo_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/home/seonho/groot_robocasa/robocasa_dataset/slicing_robocasa_human_v3")
    ap.add_argument("--groot-pretrained-path", default="/home/seonho/ws3/outputs/groot_inst/checkpoints/050000/pretrained_model")
    ap.add_argument("--mgd-trainable-mode", default="processed_only")
    ap.add_argument("--mgd-enabled", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--mgd-loss-weight", type=float, default=0.05)
    ap.add_argument("--max-tasks", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--token-mask-ratio", type=float, default=0.1)
    ap.add_argument("--short-run-steps", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    repo_ids = collect_repo_ids(root, args.max_tasks)
    if not repo_ids:
        raise RuntimeError("No task repos found")

    first_meta = LeRobotDatasetMetadata(repo_id=repo_ids[0], root=root / repo_ids[0])

    cfg = make_policy_config(
        "groot_processed_mgd",
        device="cuda",
        base_model_path="nvidia/GR00T-N1.5-3B",
        groot_pretrained_path=args.groot_pretrained_path,
        mgd_trainable_mode=args.mgd_trainable_mode,
        mgd_enabled=args.mgd_enabled,
        mgd_backprop_backbone=True,
        mgd_target_projection="frozen_random",
        mgd_target_pooling="flatten",
        mgd_token_mask_ratio=args.token_mask_ratio,
        mgd_loss_weight=args.mgd_loss_weight,
        mgd_fm_loss_weight=1.0,
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        lora_rank=0,
        chunk_size=16,
        n_action_steps=16,
    )

    delta_timestamps = resolve_delta_timestamps(cfg, first_meta)
    dataset = MultiLeRobotDataset(
        repo_ids=repo_ids,
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    pre, _ = make_pre_post_processors(cfg, dataset_stats=dataset.stats)
    policy = make_policy(cfg, ds_meta=first_meta)
    policy.train()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    raw_batch = next(iter(loader))
    batch = pre(raw_batch)

    # 4) Check processed feature path.
    with torch.no_grad():
        groot_inputs = policy._build_groot_inputs(batch, include_action=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=policy.config.use_bf16):
            backbone_inputs, _ = policy._groot_model.prepare_input(groot_inputs)
            bb_raw = policy._groot_model.backbone(backbone_inputs)
            raw_feat = bb_raw["backbone_features"].detach().float()
            bb_proc = policy._groot_model.action_head.process_backbone_output(bb_raw)
            proc_feat_direct = bb_proc["backbone_features"].detach().float()
            fw_out = policy._groot_model.forward(groot_inputs, return_intermediate=True)
        proc_feat_fw = fw_out["backbone_features"].detach().float()
        proc_mask_fw = fw_out.get("backbone_attention_mask")

    same_processed_train_mode = torch.allclose(proc_feat_direct, proc_feat_fw, atol=1e-5, rtol=1e-4)
    changed_from_raw = not torch.allclose(raw_feat, proc_feat_direct, atol=1e-5, rtol=1e-4)
    # Dropout in train mode can make two forward paths differ. Re-check in eval mode.
    policy._groot_model.action_head.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=policy.config.use_bf16):
            backbone_inputs, _ = policy._groot_model.prepare_input(groot_inputs)
            bb_raw_eval = policy._groot_model.backbone(backbone_inputs)
            bb_proc_eval = policy._groot_model.action_head.process_backbone_output(bb_raw_eval)
            proc_eval_direct = bb_proc_eval["backbone_features"].detach().float()
            fw_eval = policy._groot_model.forward(groot_inputs, return_intermediate=True)
            proc_eval_fw = fw_eval["backbone_features"].detach().float()
    same_processed_eval_mode = torch.allclose(proc_eval_direct, proc_eval_fw, atol=1e-5, rtol=1e-4)
    policy._groot_model.action_head.train()
    print(f"[SHAPE] processed_backbone_features={tuple(proc_feat_fw.shape)}")
    print(f"[SHAPE] backbone_attention_mask={tuple(proc_mask_fw.shape) if proc_mask_fw is not None else None}")
    print(f"[CHECK] processed_equals_process_backbone_output_train={same_processed_train_mode}")
    print(f"[CHECK] processed_equals_process_backbone_output_eval={same_processed_eval_mode}")
    print(f"[CHECK] processed_differs_from_raw_backbone={changed_from_raw}")

    # 5) Capture mask flow & target/pred shapes.
    capture = {}
    orig_token_mask_forward = policy.token_mask.forward
    orig_seq_forward = policy.sequence_mgd_head.forward
    orig_target_forward = policy.action_target_projector.forward

    def token_mask_wrap(token_features, valid_token_mask):
        masked_tokens, keep_mask, stats = orig_token_mask_forward(token_features, valid_token_mask)
        capture["valid_mask"] = valid_token_mask.detach().clone()
        capture["keep_mask"] = keep_mask.detach().clone()
        capture["token_stats"] = {k: v.detach().clone() for k, v in stats.items()}
        return masked_tokens, keep_mask, stats

    def seq_head_wrap(token_features, valid_token_mask, kept_token_mask):
        capture["seq_valid_mask_arg"] = valid_token_mask.detach().clone()
        capture["seq_keep_mask_arg"] = kept_token_mask.detach().clone()
        out = orig_seq_forward(token_features, valid_token_mask, kept_token_mask)
        capture["z_a_hat_shape"] = tuple(out.shape)
        return out

    def target_wrap(action_enc_out):
        out = orig_target_forward(action_enc_out)
        capture["z_a_target_shape"] = tuple(out.shape)
        return out

    policy.token_mask.forward = token_mask_wrap
    policy.sequence_mgd_head.forward = seq_head_wrap
    policy.action_target_projector.forward = target_wrap

    # 1-step backward runtime smoke.
    optim = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-4)
    optim.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    loss, loss_dict = policy(batch)
    loss.backward()
    optim.step()
    step_time = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # 9) loss_dict required keys.
    required_keys = ["loss", "flow_matching_loss", "mgd_loss"]
    if args.mgd_enabled:
        required_keys += [
            "mgd_cos_sim",
            "valid_token_count",
            "kept_token_count",
            "actual_token_mask_ratio",
        ]
    missing = [k for k in required_keys if k not in loss_dict]

    # 6) trainable parameter grouping.
    all_named = list(policy.named_parameters())

    def group_count(prefix: str):
        total = sum(p.numel() for n, p in all_named if n.startswith(prefix))
        train = sum(p.numel() for n, p in all_named if n.startswith(prefix) and p.requires_grad)
        return total, train

    vlln_total, vlln_train = group_count("_groot_model.action_head.vlln")
    vlsa_total, vlsa_train = group_count("_groot_model.action_head.vl_self_attention")
    dit_total, dit_train = group_count("_groot_model.action_head.model")
    seq_total, seq_train = group_count("sequence_mgd_head")
    ae_total, ae_train = group_count("_groot_model.action_head.action_encoder")
    proj_total, proj_train = group_count("action_target_projector")
    raw_total, raw_train = group_count("_groot_model.backbone")
    whole_total = sum(p.numel() for _, p in all_named)
    whole_train = sum(p.numel() for _, p in all_named if p.requires_grad)
    if args.mgd_trainable_mode == "processed_only":
        selected_prefixes = (
            "_groot_model.action_head.vlln",
            "_groot_model.action_head.vl_self_attention",
            "sequence_mgd_head",
        )
    elif args.mgd_trainable_mode == "dit_core_only":
        selected_prefixes = ("_groot_model.action_head.model",)
    else:
        selected_prefixes = tuple()
    other_train = sum(
        p.numel() for n, p in all_named if p.requires_grad and not n.startswith(selected_prefixes)
    )

    # 8) gradient checks.
    def grad_nonzero(prefix: str):
        nz = 0
        total = 0
        for n, p in all_named:
            if n.startswith(prefix):
                total += 1
                if p.grad is not None and p.grad.detach().abs().sum().item() > 0:
                    nz += 1
        return nz, total

    def grad_not_none(prefix: str):
        bad = 0
        total = 0
        for n, p in all_named:
            if n.startswith(prefix):
                total += 1
                if p.grad is not None:
                    bad += 1
        return bad, total

    vlln_nz = grad_nonzero("_groot_model.action_head.vlln")
    vlsa_nz = grad_nonzero("_groot_model.action_head.vl_self_attention")
    seq_nz = grad_nonzero("sequence_mgd_head")
    raw_vlm_bad = grad_not_none("_groot_model.backbone")
    dit_bad = grad_not_none("_groot_model.action_head.model")
    ae_bad = grad_not_none("_groot_model.action_head.action_encoder")
    proj_bad = grad_not_none("action_target_projector")

    # 5) TokenMask behavior checks.
    if args.mgd_enabled:
        valid_only_masking = bool(((~capture["valid_mask"]) & capture["keep_mask"]).sum().item() == 0)
        keep_mask_passed = torch.equal(capture["keep_mask"], capture["seq_keep_mask_arg"])
        valid_mask_passed = torch.equal(capture["valid_mask"], capture["seq_valid_mask_arg"])
    else:
        valid_only_masking = None
        keep_mask_passed = None
        valid_mask_passed = None

    print(f"[SHAPE] z_A_hat={capture.get('z_a_hat_shape')}")
    print(f"[SHAPE] z_A_target={capture.get('z_a_target_shape')}")
    print(f"[CHECK] token_mask_valid_only={valid_only_masking}")
    print(f"[CHECK] keep_mask_passed_to_attention_pooling={keep_mask_passed}")
    print(f"[CHECK] valid_mask_passed_to_attention_pooling={valid_mask_passed}")
    print(f"[CHECK] loss_dict_missing_keys={missing}")

    print(f"[PARAM] vlln total/train={vlln_total}/{vlln_train}")
    print(f"[PARAM] vl_self_attention total/train={vlsa_total}/{vlsa_train}")
    print(f"[PARAM] dit_core total/train={dit_total}/{dit_train}")
    print(f"[PARAM] sequence_mgd_head total/train={seq_total}/{seq_train}")
    print(f"[PARAM] action_encoder total/train={ae_total}/{ae_train}")
    print(f"[PARAM] target_projector total/train={proj_total}/{proj_train}")
    print(f"[PARAM] raw_backbone total/train={raw_total}/{raw_train}")
    print(f"[PARAM] other_trainable={other_train}")
    print(f"[PARAM] whole total/train={whole_total}/{whole_train}")

    print(f"[GRAD] vlln nonzero/total={vlln_nz}")
    print(f"[GRAD] vl_self_attention nonzero/total={vlsa_nz}")
    print(f"[GRAD] sequence_mgd_head nonzero/total={seq_nz}")
    print(f"[GRAD] raw_vlm grad_not_none/total={raw_vlm_bad}")
    print(f"[GRAD] dit grad_not_none/total={dit_bad}")
    print(f"[GRAD] action_encoder grad_not_none/total={ae_bad}")
    print(f"[GRAD] target_projector grad_not_none/total={proj_bad}")
    print(f"[RUNTIME] one_step_loss={float(loss.item()):.6f} step_time_sec={step_time:.4f} peak_mem_gb={peak_mem_gb:.3f}")

    # 100-step short run.
    if args.short_run_steps > 0:
        policy.train()
        data_iter = iter(loader)
        mgd_vals = []
        cos_vals = []
        fm_vals = []
        step_times = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.short_run_steps):
            try:
                raw = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                raw = next(data_iter)
            b = pre(raw)
            optim.zero_grad(set_to_none=True)
            t1 = time.perf_counter()
            l, d = policy(b)
            l.backward()
            optim.step()
            step_times.append(time.perf_counter() - t1)
            mgd_vals.append(float(d["mgd_loss"]))
            cos_vals.append(float(d["mgd_cos_sim"]))
            fm_vals.append(float(d["flow_matching_loss"]))
        peak_run_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(
            "[SHORT] steps={} mgd_loss_start={:.6f} mgd_loss_end={:.6f} "
            "mgd_cos_start={:.6f} mgd_cos_end={:.6f} fm_start={:.6f} fm_end={:.6f} fm_max={:.6f} "
            "peak_mem_gb={:.3f} step_time_mean_sec={:.4f}".format(
                args.short_run_steps,
                mgd_vals[0],
                mgd_vals[-1],
                cos_vals[0],
                cos_vals[-1],
                fm_vals[0],
                fm_vals[-1],
                max(fm_vals),
                peak_run_gb,
                sum(step_times) / max(len(step_times), 1),
            )
        )


if __name__ == "__main__":
    main()
