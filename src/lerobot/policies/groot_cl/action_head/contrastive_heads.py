from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ContrastiveHeadConfig:
    latent_dim: int = 256
    vlm_input_dim: int = 1536
    action_input_dim: int = 32
    cnn_hidden_dim: int = 128
    proj_hidden_dim: int = 512
    triplet_margin: float = 0.5


class VLMContrastiveHead(nn.Module):
    def __init__(self, config: ContrastiveHeadConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.vlm_input_dim, config.proj_hidden_dim),
            nn.LayerNorm(config.proj_hidden_dim),
            nn.GELU(),
            nn.Linear(config.proj_hidden_dim, config.latent_dim),
        )

    def forward(self, backbone_features: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1).float()
            pooled = (backbone_features * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = backbone_features.mean(1)
        return F.normalize(self.proj(pooled), dim=-1)


class ActionContrastiveHead(nn.Module):
    def __init__(self, config: ContrastiveHeadConfig):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(config.action_input_dim, config.cnn_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.cnn_hidden_dim),
            nn.GELU(),
            nn.Conv1d(config.cnn_hidden_dim, config.cnn_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.cnn_hidden_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(config.cnn_hidden_dim, config.latent_dim)

    def forward(self, actions: Tensor) -> Tensor:
        x = actions.transpose(1, 2)
        x = self.cnn(x)
        x = x.mean(dim=-1)
        return F.normalize(self.proj(x), dim=-1)


def triplet_contrastive_loss(
    vlm_z: Tensor,
    pos_action_z: Tensor,
    neg_action_z: Tensor,
    margin: float = 0.5,
) -> Tensor:
    if vlm_z.shape[0] < 1:
        return vlm_z.new_tensor(0.0)
    d_pos = 1.0 - (vlm_z * pos_action_z).sum(dim=-1)
    d_neg = 1.0 - (vlm_z * neg_action_z).sum(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


def info_nce_fallback(vlm_z: Tensor, action_z: Tensor, temperature: float = 0.07) -> Tensor:
    B = vlm_z.shape[0]
    if B < 2:
        return vlm_z.new_tensor(0.0)
    logits = torch.matmul(vlm_z, action_z.T) / temperature
    labels = torch.arange(B, device=vlm_z.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
