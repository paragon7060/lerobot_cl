#!/usr/bin/env python
"""Phase 1 완료 후 action encoder feature 품질 검증 스크립트.

GT action을 noise_timestep=0으로 MultiEmbodimentActionEncoder에 통과시켜
clean action feature를 추출하고, wrist joint (index 6) 기준으로
feature space 구조를 시각화 및 정량 평가한다.

사용법:
  python scripts/eval_action_features.py \
      --checkpoint_dir=./outputs/groot_cl_v2_phase1/checkpoints/last \
      --dataset.repo_id=paragon7060/INSIGHTfixposV3 \
      --dataset.root=/home/seonho/workspace/data/paragon7060/INSIGHTfixposV3 \
      --dataset.video_backend=pyav \
      --policy.type=groot_cl_v2 \
      --num_samples=512 \
      --output_dir=./outputs/eval_action_features \
      --wrist_index=6
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Phase 1 체크포인트 디렉토리 (model.safetensors 포함)")
    parser.add_argument("--repo_id", type=str, default="paragon7060/INSIGHTfixposV3")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--video_backend", type=str, default="pyav")
    parser.add_argument("--num_samples", type=int, default=512,
                        help="평가할 샘플 수")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_action_features")
    parser.add_argument("--wrist_index", type=int, default=6,
                        help="Wrist joint index in action vector")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def extract_action_features(policy, batch, device):
    """GT action → MultiEmbodimentActionEncoder(timestep=0) → pooled feature."""
    # Get processed inputs
    allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
    groot_inputs = {
        k: v.to(device)
        for k, v in batch.items()
        if (k in allowed_base or k.startswith("eagle_"))
        and not (k.startswith("next.") or k == "info")
    }

    # Prepare inputs through groot's prepare_input
    _, action_inputs = policy._groot_model.prepare_input(groot_inputs)

    gt_action = action_inputs.action          # (B, T, 32) normalized
    embodiment_id = action_inputs.embodiment_id  # (B,)
    B = gt_action.shape[0]

    # timestep=0: clean action features (no noise conditioning)
    t_zero = torch.zeros(B, dtype=torch.long, device=device)

    action_features = policy._groot_model.action_head.action_encoder(
        gt_action, t_zero, embodiment_id
    )  # (B, T, 1536)

    # Global average pooling over action horizon
    pooled = action_features.mean(dim=1)  # (B, 1536)
    return pooled, gt_action


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.groot_cl.processor_groot import make_groot_pre_post_processors
    from lerobot.policies.groot_cl_v2.configuration_groot_cl_v2 import GrootCLv2Config
    from lerobot.policies.groot_cl_v2.modeling_groot_cl_v2 import GrootCLv2Policy

    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.root)

    # ── Load policy from checkpoint ───────────────────────────────────────────
    logger.info("체크포인트 로드: %s", args.checkpoint_dir)
    policy = GrootCLv2Policy.from_pretrained(
        pretrained_name_or_path=args.checkpoint_dir,
    )
    policy = policy.to(device)
    policy.eval()
    logger.info("Policy 로드 완료.")

    # ── Setup preprocessor ────────────────────────────────────────────────────
    pre, _ = make_groot_pre_post_processors(policy.config, dataset_stats=ds_meta.stats)

    # ── Build dataset ─────────────────────────────────────────────────────────
    from lerobot.datasets.factory import resolve_delta_timestamps
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    logger.info("Dataset: %d frames", dataset.num_frames)

    # Subsample
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ── Extract features ──────────────────────────────────────────────────────
    all_features = []
    all_wrist_angles = []

    logger.info("Action feature 추출 중...")
    for raw_batch in dataloader:
        batch = pre(raw_batch)
        features, gt_action = extract_action_features(policy, batch, device)
        # gt_action: (B, T, 32) normalized. wrist = index 6, first timestep
        wrist = gt_action[:, 0, args.wrist_index].cpu().numpy()
        all_features.append(features.cpu().numpy())
        all_wrist_angles.append(wrist)

    features_np = np.concatenate(all_features, axis=0)   # (N, 1536)
    wrist_np = np.concatenate(all_wrist_angles, axis=0)  # (N,)
    N = len(features_np)
    logger.info("추출 완료: %d samples, feature dim=%d", N, features_np.shape[1])

    # Save raw features
    np.save(output_dir / "action_features.npy", features_np)
    np.save(output_dir / "wrist_angles.npy", wrist_np)
    logger.info("Raw features 저장 완료: %s", output_dir)

    # ── Metric 1: Wrist-similarity correlation ────────────────────────────────
    # Compute pairwise: action feature cosine similarity vs wrist angle difference
    logger.info("Metric 1: action feature 유사도 vs wrist angle 차이 상관계수 계산...")
    sample_idx = np.random.choice(N, min(256, N), replace=False)
    feat_sub = features_np[sample_idx]  # (M, 1536)
    wrist_sub = wrist_np[sample_idx]    # (M,)

    # Normalize features for cosine sim
    feat_norm = feat_sub / (np.linalg.norm(feat_sub, axis=1, keepdims=True) + 1e-8)
    cosine_sim = feat_norm @ feat_norm.T  # (M, M)

    wrist_diff = np.abs(wrist_sub[:, None] - wrist_sub[None, :])  # (M, M)

    # Upper triangle (exclude diagonal)
    mask = np.triu(np.ones_like(cosine_sim, dtype=bool), k=1)
    sim_flat = cosine_sim[mask]
    diff_flat = wrist_diff[mask]

    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(sim_flat, -diff_flat)  # higher sim ↔ lower diff
    spearman_r, spearman_p = spearmanr(sim_flat, -diff_flat)
    logger.info(
        "Pearson r=%.4f (p=%.4f), Spearman r=%.4f (p=%.4f)",
        pearson_r, pearson_p, spearman_r, spearman_p,
    )

    # ── Metric 2: kNN wrist-bin classification accuracy ──────────────────────
    logger.info("Metric 2: k-NN 기반 wrist 구간 분류 정확도...")
    # Bin wrist angles into 5 quantile bins
    bins = np.quantile(wrist_np, np.linspace(0, 1, 6))
    wrist_labels = np.digitize(wrist_np, bins[1:-1])  # (N,) 0~4

    # k-NN leave-one-out on subsample
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    scores = cross_val_score(knn, features_np, wrist_labels, cv=5, scoring="accuracy")
    logger.info(
        "5-fold kNN accuracy (k=5, cosine): %.4f ± %.4f",
        scores.mean(), scores.std(),
    )

    # ── Visualization: PCA + t-SNE ────────────────────────────────────────────
    logger.info("시각화 생성 중...")
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # PCA to 50 dims first for efficiency
        pca = PCA(n_components=min(50, N, features_np.shape[1]))
        feat_pca = pca.fit_transform(features_np)

        # t-SNE to 2D
        tsne = TSNE(n_components=2, perplexity=min(30, N // 4), random_state=42, n_jobs=-1)
        feat_2d = tsne.fit_transform(feat_pca)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: t-SNE colored by wrist angle
        sc = axes[0].scatter(
            feat_2d[:, 0], feat_2d[:, 1],
            c=wrist_np, cmap="coolwarm", s=8, alpha=0.7,
        )
        plt.colorbar(sc, ax=axes[0], label="Wrist angle (normalized)")
        axes[0].set_title("Action Features — t-SNE (colored by wrist angle)")
        axes[0].set_xlabel("t-SNE dim 1")
        axes[0].set_ylabel("t-SNE dim 2")

        # Plot 2: wrist angle distribution
        axes[1].hist(wrist_np, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        axes[1].set_title("Wrist angle distribution")
        axes[1].set_xlabel("Wrist angle (normalized)")
        axes[1].set_ylabel("Count")

        plt.suptitle(
            f"Action Feature Quality (Phase 1)\n"
            f"Pearson r={pearson_r:.3f}, kNN acc={scores.mean():.3f}",
            fontsize=13,
        )
        plt.tight_layout()
        fig_path = output_dir / "action_feature_tsne.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("시각화 저장: %s", fig_path)

    except ImportError as e:
        logger.warning("시각화 스킵 (패키지 없음): %s", e)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "num_samples": N,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "knn_accuracy_mean": float(scores.mean()),
        "knn_accuracy_std": float(scores.std()),
    }
    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("평가 결과 요약:")
    logger.info("  Pearson r (sim vs -|wrist_diff|): %.4f", pearson_r)
    logger.info("  Spearman r:                       %.4f", spearman_r)
    logger.info("  kNN wrist-bin accuracy:           %.4f ± %.4f", scores.mean(), scores.std())
    logger.info("  결과 저장: %s", output_dir)
    logger.info("=" * 60)
    logger.info(
        "해석: Pearson r > 0.3 이면 action feature가 wrist angle을 구분하고 있음. "
        "kNN accuracy > 0.5 (5클래스 기준 0.2 이상) 이면 양호."
    )


if __name__ == "__main__":
    main()
