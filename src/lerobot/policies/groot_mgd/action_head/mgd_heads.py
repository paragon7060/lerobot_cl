from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_pool(features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return features.mean(dim=1)
    mask_f = mask.unsqueeze(-1).to(dtype=features.dtype)
    return (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)


class ChannelMask(nn.Module):
    def __init__(self, mask_ratio: float):
        super().__init__()
        self.mask_ratio = float(mask_ratio)

    def set_mask_ratio(self, mask_ratio: float) -> None:
        self.mask_ratio = float(mask_ratio)

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        if self.mask_ratio <= 0.0:
            return z_v
        keep = torch.rand_like(z_v) >= self.mask_ratio
        return z_v * keep.to(dtype=z_v.dtype)


class ActionTargetProjector(nn.Module):
    def __init__(
        self,
        action_horizon: int,
        action_latent_dim: int,
        target_dim: int = 512,
        pooling: str = "flatten",
        projection: str = "frozen_random",
        pretrained_projector_path: str | None = None,
        trainable: bool = False,
    ):
        super().__init__()
        if pooling not in {"flatten", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling!r}")
        if projection not in {"frozen_random", "pretrained_ae"}:
            raise ValueError(f"Unsupported projection: {projection!r}")

        self.pooling = pooling
        self.projection = projection
        self.action_horizon = int(action_horizon)
        self.action_latent_dim = int(action_latent_dim)
        self.target_dim = int(target_dim)

        in_dim = self.action_horizon * self.action_latent_dim if pooling == "flatten" else self.action_latent_dim
        self.projector = nn.Linear(in_dim, self.target_dim, bias=True)

        if projection == "pretrained_ae":
            if not pretrained_projector_path:
                raise ValueError("pretrained_projector_path is required when projection='pretrained_ae'.")
            self._load_pretrained_projector(pretrained_projector_path)

        if not trainable:
            self.projector.requires_grad_(False)

    def _pool(self, action_enc_out: torch.Tensor) -> torch.Tensor:
        if self.pooling == "flatten":
            return action_enc_out.flatten(start_dim=1)
        return action_enc_out.mean(dim=1)

    def forward(self, action_enc_out: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(action_enc_out)
        return self.projector(pooled)

    def _load_pretrained_projector(self, ckpt_path: str) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Projector checkpoint not found: {path}")

        if path.suffix == ".safetensors":
            from safetensors.torch import load_file as _load_file

            state = _load_file(str(path))
        else:
            obj = torch.load(str(path), map_location="cpu")
            if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
                state = obj["state_dict"]
            elif isinstance(obj, dict):
                state = obj
            else:
                raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

        # 1) direct load by common key names
        direct_candidates = [
            ("weight", "bias"),
            ("projector.weight", "projector.bias"),
            ("encoder.weight", "encoder.bias"),
            ("encoder.fc.weight", "encoder.fc.bias"),
            ("encoder.net.0.weight", "encoder.net.0.bias"),
        ]
        for w_key, b_key in direct_candidates:
            if w_key in state and state[w_key].shape == self.projector.weight.shape:
                with torch.no_grad():
                    self.projector.weight.copy_(state[w_key].to(dtype=self.projector.weight.dtype))
                    if b_key in state and state[b_key].shape == self.projector.bias.shape:
                        self.projector.bias.copy_(state[b_key].to(dtype=self.projector.bias.dtype))
                return

        # 2) fallback: find any matching 2D tensor shape
        weight_key = None
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and tuple(v.shape) == tuple(self.projector.weight.shape):
                weight_key = k
                break
        if weight_key is None:
            raise ValueError(
                f"Could not find projector weight with shape {tuple(self.projector.weight.shape)} in {path}."
            )

        bias_key = None
        target_bias_shape = tuple(self.projector.bias.shape)
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and tuple(v.shape) == target_bias_shape:
                bias_key = k
                break

        with torch.no_grad():
            self.projector.weight.copy_(state[weight_key].to(dtype=self.projector.weight.dtype))
            if bias_key is not None:
                self.projector.bias.copy_(state[bias_key].to(dtype=self.projector.bias.dtype))


class ActionAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 512, hidden_dim: int | None = None):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else max(latent_dim * 2, min(input_dim, 4096))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.GELU(),
            nn.Linear(h, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.GELU(),
            nn.Linear(h, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @staticmethod
    def reconstruction_loss(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mse_weight: float = 1.0,
        cos_weight: float = 0.0,
    ) -> torch.Tensor:
        loss = mse_weight * F.mse_loss(x_hat, x)
        if cos_weight > 0:
            cos = F.cosine_similarity(x_hat, x, dim=-1).mean()
            loss = loss + cos_weight * (1.0 - cos)
        return loss


class MGDReconstructionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(z_v)))


def mgd_reconstruction_loss(z_a_hat: torch.Tensor, z_a_target: torch.Tensor, kind: str = "cosine") -> torch.Tensor:
    if kind == "cosine":
        return (1.0 - F.cosine_similarity(z_a_hat, z_a_target, dim=-1)).mean()
    if kind == "mse":
        return F.mse_loss(z_a_hat, z_a_target)
    raise ValueError(f"Unsupported loss kind: {kind!r}")
