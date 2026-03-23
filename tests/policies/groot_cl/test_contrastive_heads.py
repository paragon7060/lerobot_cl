import pytest
import torch

from lerobot.policies.groot_cl.action_head.contrastive_heads import (
    ActionContrastiveHead,
    ContrastiveHeadConfig,
    VLMContrastiveHead,
    info_nce_fallback,
    triplet_contrastive_loss,
)


@pytest.fixture
def cfg():
    return ContrastiveHeadConfig(
        latent_dim=64,
        vlm_input_dim=128,
        action_input_dim=8,
        cnn_hidden_dim=32,
        proj_hidden_dim=128,
        triplet_margin=0.5,
    )


class TestVLMContrastiveHead:
    def test_output_shape(self, cfg):
        head = VLMContrastiveHead(cfg)
        B, T_seq = 4, 20
        features = torch.randn(B, T_seq, cfg.vlm_input_dim)
        out = head(features)
        assert out.shape == (B, cfg.latent_dim)

    def test_l2_normalized(self, cfg):
        head = VLMContrastiveHead(cfg)
        features = torch.randn(2, 10, cfg.vlm_input_dim)
        out = head(features)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_attn_mask_applied(self, cfg):
        head = VLMContrastiveHead(cfg)
        B, T_seq = 2, 10
        features = torch.randn(B, T_seq, cfg.vlm_input_dim)
        mask_full = torch.ones(B, T_seq)
        mask_half = torch.zeros(B, T_seq)
        mask_half[:, :5] = 1.0

        out_full = head(features, mask_full)
        out_half = head(features, mask_half)
        assert not torch.allclose(out_full, out_half)

    def test_no_attn_mask(self, cfg):
        head = VLMContrastiveHead(cfg)
        features = torch.randn(3, 15, cfg.vlm_input_dim)
        out = head(features, attn_mask=None)
        assert out.shape == (3, cfg.latent_dim)


class TestActionContrastiveHead:
    def test_output_shape(self, cfg):
        head = ActionContrastiveHead(cfg)
        B, T, D = 4, 16, cfg.action_input_dim
        actions = torch.randn(B, T, D)
        out = head(actions)
        assert out.shape == (B, cfg.latent_dim)

    def test_l2_normalized(self, cfg):
        head = ActionContrastiveHead(cfg)
        actions = torch.randn(2, 16, cfg.action_input_dim)
        out = head(actions)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_different_trajectories_different_embeddings(self, cfg):
        head = ActionContrastiveHead(cfg)
        a1 = torch.zeros(1, 16, cfg.action_input_dim)
        a2 = torch.ones(1, 16, cfg.action_input_dim)
        z1 = head(a1)
        z2 = head(a2)
        assert not torch.allclose(z1, z2)


class TestTripletContrastiveLoss:
    def test_zero_loss_when_neg_farther(self):
        B, D = 4, 64
        vlm_z = torch.randn(B, D)
        vlm_z = vlm_z / vlm_z.norm(dim=-1, keepdim=True)
        pos_z = vlm_z.clone()
        neg_z = -vlm_z
        loss = triplet_contrastive_loss(vlm_z, pos_z, neg_z, margin=0.5)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_loss_when_neg_closer(self):
        B, D = 4, 64
        vlm_z = torch.randn(B, D)
        vlm_z = vlm_z / vlm_z.norm(dim=-1, keepdim=True)
        neg_z = vlm_z.clone()
        pos_z = -vlm_z
        loss = triplet_contrastive_loss(vlm_z, pos_z, neg_z, margin=0.5)
        assert loss.item() > 0.0

    def test_empty_batch(self):
        vlm_z = torch.zeros(0, 64)
        pos_z = torch.zeros(0, 64)
        neg_z = torch.zeros(0, 64)
        loss = triplet_contrastive_loss(vlm_z, pos_z, neg_z)
        assert loss.item() == pytest.approx(0.0)

    def test_gradient_flows(self):
        B, D = 4, 64
        vlm_z = torch.randn(B, D, requires_grad=True)
        vlm_z_norm = vlm_z / vlm_z.norm(dim=-1, keepdim=True)
        pos_z = torch.randn(B, D)
        pos_z = pos_z / pos_z.norm(dim=-1, keepdim=True)
        neg_z = torch.randn(B, D)
        neg_z = neg_z / neg_z.norm(dim=-1, keepdim=True)
        loss = triplet_contrastive_loss(vlm_z_norm, pos_z, neg_z)
        loss.backward()
        assert vlm_z.grad is not None


class TestInfoNceFallback:
    def test_output_is_scalar(self):
        B, D = 4, 64
        vlm_z = torch.randn(B, D)
        vlm_z = vlm_z / vlm_z.norm(dim=-1, keepdim=True)
        action_z = torch.randn(B, D)
        action_z = action_z / action_z.norm(dim=-1, keepdim=True)
        loss = info_nce_fallback(vlm_z, action_z)
        assert loss.ndim == 0

    def test_loss_positive(self):
        B, D = 8, 64
        vlm_z = torch.randn(B, D)
        vlm_z = vlm_z / vlm_z.norm(dim=-1, keepdim=True)
        action_z = torch.randn(B, D)
        action_z = action_z / action_z.norm(dim=-1, keepdim=True)
        loss = info_nce_fallback(vlm_z, action_z)
        assert loss.item() > 0.0
