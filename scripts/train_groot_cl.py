#!/usr/bin/env python
"""GR00T Contrastive Learning 학습 스크립트 (accelerate + draccus CLI + wandb 지원).

단일 GPU 실행:
  python scripts/train_groot_cl.py \
      --repo_id=paragon7060/INSIGHTfixposV3 \
      --root=/mntvol1/INSIGHTBench/data/paragon7060/INSIGHTfixposV3 \
      --output_dir=./outputs/groot_cl \
      --job_name=groot_cl_run1 \
      --phase1_steps=500 \
      --phase2a_steps=5000 \
      --batch_size=4 \
      --wandb.enable=true \
      --wandb.project=groot_cl \
      --wandb.entity=RwHlabs \
      --wandb.disable_artifact=true

Multi-GPU 실행:
  accelerate launch --num_processes 2 scripts/train_groot_cl.py [same args]

주의: negative_action은 batch_to_transition()을 통과하면 사라지므로
      전처리 파이프라인(pre) 밖에서 NegativeActionNormalizeStep을 직접 호출한다.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from lerobot.configs.default import WandBConfig
from lerobot.datasets.contrastive_dataset import ContrastiveLeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.groot_cl.configuration_groot_cl import GrootCLConfig
from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
from lerobot.policies.groot_cl.processor_groot_cl import NegativeActionNormalizeStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 설정 dataclass — draccus가 CLI 인자로 파싱
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GrootCLTrainConfig:
    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    repo_id: str = "paragon7060/INSIGHTfixposV3"
    root: str | None = None
    neg_pairs_path: str = "/home/bluepot/cl_ws/negative_pairs.json"

    # ── 모델 ─────────────────────────────────────────────────────────────────
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    # 데이터셋별 사전학습된 GrootPolicy 체크포인트 경로.
    # 설정 시 _groot_model.* weights를 로드하고 contrastive heads는 랜덤 초기화.
    groot_pretrained_path: str | None = None

    # ── 출력 / 이름 ───────────────────────────────────────────────────────────
    output_dir: Path = Path("outputs/groot_cl")
    job_name: str = "groot_cl"

    # ── 학습 단계 ─────────────────────────────────────────────────────────────
    phase1_steps: int = 500
    phase2a_steps: int = 5000

    # ── 최적화 ────────────────────────────────────────────────────────────────
    phase1_lr: float = 1e-4
    phase2a_lr: float = 2e-5
    # Phase 2a에서 flow_matching_loss 스케일에 맞게 contrastive loss를 낮춰야 함
    phase2a_loss_weight: float = 0.05

    # ── DataLoader ────────────────────────────────────────────────────────────
    batch_size: int = 4
    num_workers: int = 4
    seed: int = 42

    # ── 로깅 / 저장 ───────────────────────────────────────────────────────────
    log_interval: int = 50
    save_interval: int = 500

    # ── Contrastive 하이퍼파라미터 ────────────────────────────────────────────
    contrastive_latent_dim: int = 256
    contrastive_cnn_hidden_dim: int = 128
    contrastive_proj_hidden_dim: int = 512
    contrastive_triplet_margin: float = 0.5

    # ── GR00T 아키텍처 ────────────────────────────────────────────────────────
    # chunk_size와 n_action_steps는 반드시 같아야 함 (configuration_groot.py 검증).
    # 데이터셋의 action horizon과 일치하도록 설정.
    chunk_size: int = 50
    n_action_steps: int = 50

    # ── GR00T 학습 대상 ───────────────────────────────────────────────────────
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb: WandBConfig = field(default_factory=WandBConfig)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

@draccus.wrap()
def main(cfg: GrootCLTrainConfig) -> None:
    # ── Accelerator ───────────────────────────────────────────────────────────
    # mixed_precision="no": modeling_groot_cl.py forward()에 torch.autocast(bf16)가
    # 이미 존재하므로 accelerate 측 autocast와 중복 적용을 막는다.
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output dir: %s", cfg.output_dir)
        logger.info("Accelerator: num_processes=%d, device=%s", accelerator.num_processes, device)

    # ── WandB 초기화 (main process 전용) ─────────────────────────────────────
    use_wandb = cfg.wandb.enable and accelerator.is_main_process
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.job_name,
            notes=cfg.wandb.notes,
            id=cfg.wandb.run_id,
            mode=cfg.wandb.mode,
            dir=str(cfg.output_dir),
            config={
                "repo_id": cfg.repo_id,
                "base_model_path": cfg.base_model_path,
                "groot_pretrained_path": cfg.groot_pretrained_path,
                "phase1_steps": cfg.phase1_steps,
                "phase2a_steps": cfg.phase2a_steps,
                "phase1_lr": cfg.phase1_lr,
                "phase2a_lr": cfg.phase2a_lr,
                "phase2a_loss_weight": cfg.phase2a_loss_weight,
                "batch_size": cfg.batch_size,
                "num_processes": accelerator.num_processes,
                "effective_batch_size": cfg.batch_size * accelerator.num_processes,
                "contrastive_latent_dim": cfg.contrastive_latent_dim,
                "contrastive_triplet_margin": cfg.contrastive_triplet_margin,
                "tune_llm": cfg.tune_llm,
                "tune_visual": cfg.tune_visual,
                "tune_projector": cfg.tune_projector,
                "tune_diffusion_model": cfg.tune_diffusion_model,
                "seed": cfg.seed,
            },
            save_code=False,
        )
        logger.info("WandB initialized: project=%s, run=%s", cfg.wandb.project, wandb.run.name)

    # ── GrootCLConfig ─────────────────────────────────────────────────────────
    policy_config = GrootCLConfig(
        base_model_path=cfg.base_model_path,
        contrastive_phase="phase1",
        contrastive_latent_dim=cfg.contrastive_latent_dim,
        contrastive_cnn_hidden_dim=cfg.contrastive_cnn_hidden_dim,
        contrastive_proj_hidden_dim=cfg.contrastive_proj_hidden_dim,
        contrastive_triplet_margin=cfg.contrastive_triplet_margin,
        contrastive_loss_weight=1.0,
        contrastive_backprop_backbone=True,
        contrastive_fallback_to_in_batch=False,
        groot_pretrained_path=cfg.groot_pretrained_path,
        tune_llm=cfg.tune_llm,
        tune_visual=cfg.tune_visual,
        tune_projector=cfg.tune_projector,
        tune_diffusion_model=cfg.tune_diffusion_model,
        use_bf16=True,
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps,
        batch_size=cfg.batch_size,
    )

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    ds_meta = LeRobotDatasetMetadata(repo_id=cfg.repo_id, root=cfg.root)
    delta_timestamps = resolve_delta_timestamps(policy_config, ds_meta)

    dataset = ContrastiveLeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root,
        negative_pairs_path=cfg.neg_pairs_path,
        delta_timestamps=delta_timestamps,
    )

    if accelerator.is_main_process:
        logger.info("Dataset: %d frames, %d episodes", dataset.num_frames, dataset.num_episodes)

    def collate_fn(batch: list[dict]) -> dict:
        """negative_action=None 샘플이 있으면 배치 전체를 None으로 처리.
        → forward()에서 loss_cont = 0.0으로 안전하게 스킵된다.
        """
        neg_actions = [item.pop("negative_action", None) for item in batch]
        result = default_collate(batch)
        if all(isinstance(n, torch.Tensor) for n in neg_actions):
            result["negative_action"] = torch.stack(neg_actions)
        else:
            result["negative_action"] = None
        return result

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ── 전처리 파이프라인 ──────────────────────────────────────────────────────
    pre, _ = make_pre_post_processors(policy_config, dataset_stats=ds_meta.stats)
    neg_normalizer = NegativeActionNormalizeStep(stats=ds_meta.stats, normalize_min_max=True)

    # ── 모델 초기화 ───────────────────────────────────────────────────────────
    policy: GrootCLPolicy = make_policy(policy_config, ds_meta=ds_meta)

    # policy, dataloader를 1회 prepare (DDP wrap + DistributedSampler 삽입)
    policy, dataloader = accelerator.prepare(policy, dataloader)

    if accelerator.is_main_process:
        logger.info("Policy: %s", policy.__class__.__name__)

    # ── 헬퍼 함수 ─────────────────────────────────────────────────────────────

    def to_device(batch: dict) -> dict:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def preprocess(raw_batch: dict) -> dict:
        """negative_action을 파이프라인 밖에서 별도 정규화 후 병합."""
        raw_neg = raw_batch.pop("negative_action", None)
        processed = pre(raw_batch)
        if isinstance(raw_neg, torch.Tensor):
            result = neg_normalizer({"negative_action": raw_neg})
            processed["negative_action"] = result.get("negative_action")
        return processed

    def infinite_dataloader():
        while True:
            yield from dataloader

    def save_checkpoint(tag: str) -> None:
        if accelerator.is_main_process:
            path = cfg.output_dir / tag
            path.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(policy).save_pretrained(path)
            logger.info("체크포인트 저장: %s", path)

    # ── 학습 루프 ─────────────────────────────────────────────────────────────

    def train_loop(
        phase: str,
        total_steps: int,
        lr: float,
        loss_weight: float,
        backprop_backbone: bool,
        global_step_offset: int = 0,
    ) -> int:
        """단일 phase 학습. 완료 후 마지막 global_step 반환."""
        if accelerator.is_main_process:
            logger.info("=" * 60)
            logger.info(
                "Phase: %s | steps=%d | lr=%.2e | loss_weight=%.3f | backprop_backbone=%s",
                phase, total_steps, lr, loss_weight, backprop_backbone,
            )
            logger.info("=" * 60)

        raw_policy = accelerator.unwrap_model(policy)
        raw_policy.set_contrastive_phase(phase)
        raw_policy.config.contrastive_loss_weight = loss_weight
        raw_policy.config.contrastive_backprop_backbone = backprop_backbone

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, raw_policy.parameters()),
            lr=lr,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr * 0.1
        )
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        policy.train()
        data_stream = infinite_dataloader()

        for step in range(1, total_steps + 1):
            raw_batch = next(data_stream)
            batch = preprocess(raw_batch)
            batch = to_device(batch)

            loss, loss_dict = policy(batch)

            optimizer.zero_grad()
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            global_step = global_step_offset + step

            if accelerator.is_main_process and step % cfg.log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                log_str = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                logger.info(
                    "[%s] step=%d/%d (global=%d) | lr=%.2e | grad_norm=%.3f | %s",
                    phase, step, total_steps, global_step, lr_now,
                    grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    log_str,
                )

                if use_wandb:
                    wandb_log = {
                        f"{phase}/loss": loss_dict.get("loss", loss.item()),
                        f"{phase}/flow_matching_loss": loss_dict.get("flow_matching_loss", 0.0),
                        f"{phase}/contrastive_loss": loss_dict.get("contrastive_loss", 0.0),
                        f"{phase}/lr": lr_now,
                        f"{phase}/grad_norm": (
                            grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else float(grad_norm)
                        ),
                    }
                    wandb.log(wandb_log, step=global_step)

            if step % cfg.save_interval == 0 or step == total_steps:
                accelerator.wait_for_everyone()
                save_checkpoint(f"{phase}/step_{step:06d}")

                if use_wandb and not cfg.wandb.disable_artifact:
                    ckpt_path = cfg.output_dir / f"{phase}/step_{step:06d}"
                    artifact = wandb.Artifact(
                        name=f"{cfg.job_name}-{phase}-step{step:06d}",
                        type="model",
                        description=f"{phase} checkpoint at step {step}",
                    )
                    artifact.add_dir(str(ckpt_path))
                    wandb.log_artifact(artifact)

        if accelerator.is_main_process:
            logger.info("[%s] 완료", phase)

        return global_step_offset + total_steps

    # ── Phase 1: contrastive heads warm-up (_groot_model frozen) ─────────────
    global_step = train_loop(
        phase="phase1",
        total_steps=cfg.phase1_steps,
        lr=cfg.phase1_lr,
        loss_weight=1.0,
        backprop_backbone=False,
        global_step_offset=0,
    )

    # ── Phase 2a: joint fine-tuning (contrastive gradient → VLM backbone) ────
    train_loop(
        phase="phase2a",
        total_steps=cfg.phase2a_steps,
        lr=cfg.phase2a_lr,
        loss_weight=cfg.phase2a_loss_weight,
        backprop_backbone=True,
        global_step_offset=global_step,
    )

    if accelerator.is_main_process:
        logger.info("전체 학습 완료. 최종 체크포인트: %s", cfg.output_dir)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
