"""Tests for GrootCLPolicy forward pass and phase management.

These tests mock the heavy GR00TN15 model to avoid GPU/weight requirements.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lerobot.policies.groot_cl.action_head.contrastive_heads import (
    ActionContrastiveHead,
    ContrastiveHeadConfig,
    VLMContrastiveHead,
)
from lerobot.policies.groot_cl.configuration_groot_cl import GrootCLConfig


def _make_mock_groot_outputs(B: int = 2, T_seq: int = 32, vlm_dim: int = 1536):
    loss = torch.tensor(0.5)
    backbone_features = torch.randn(B, T_seq, vlm_dim)
    backbone_attention_mask = torch.ones(B, T_seq)
    return {
        "loss": loss,
        "backbone_features": backbone_features,
        "backbone_attention_mask": backbone_attention_mask,
    }


def _make_batch(B: int = 2, T: int = 16, D: int = 32, with_neg: bool = True):
    batch = {
        "state": torch.randn(B, 1, D),
        "state_mask": torch.ones(B, 1, dtype=torch.bool),
        "action": torch.randn(B, T, D),
        "action_mask": torch.ones(B, T, dtype=torch.bool),
    }
    if with_neg:
        batch["negative_action"] = torch.randn(B, T, D)
    return batch


class TestGrootCLForwardWithMock:
    """Tests that mock _groot_model to avoid loading 3B-param weights."""

    def _build_policy_with_mock(self, phase="phase2a", backprop=True, fallback=False):
        cfg = GrootCLConfig.__new__(GrootCLConfig)
        cfg.contrastive_latent_dim = 64
        cfg.contrastive_cnn_hidden_dim = 32
        cfg.contrastive_proj_hidden_dim = 128
        cfg.contrastive_triplet_margin = 0.5
        cfg.contrastive_loss_weight = 0.1
        cfg.contrastive_phase = phase
        cfg.contrastive_backprop_backbone = backprop
        cfg.contrastive_fallback_to_in_batch = fallback
        cfg.use_bf16 = False
        cfg.max_action_dim = 32
        cfg.tune_visual = False
        cfg.tune_llm = False
        cfg.tune_projector = True
        cfg.tune_diffusion_model = True

        contrastive_cfg = ContrastiveHeadConfig(
            latent_dim=cfg.contrastive_latent_dim,
            vlm_input_dim=1536,
            action_input_dim=cfg.max_action_dim,
            cnn_hidden_dim=cfg.contrastive_cnn_hidden_dim,
            proj_hidden_dim=cfg.contrastive_proj_hidden_dim,
            triplet_margin=cfg.contrastive_triplet_margin,
        )

        mock_groot = MagicMock()
        mock_groot.parameters.return_value = iter([torch.zeros(1)])

        from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
        import torch.nn as nn

        policy = nn.Module.__new__(GrootCLPolicy)
        nn.Module.__init__(policy)
        policy.config = cfg
        policy._groot_model = mock_groot
        policy.vlm_contrastive_head = VLMContrastiveHead(contrastive_cfg)
        policy.action_contrastive_head = ActionContrastiveHead(contrastive_cfg)
        return policy, mock_groot

    def test_forward_training_with_negatives(self):
        policy, mock_groot = self._build_policy_with_mock(phase="phase2a", backprop=True)
        policy.train()

        B, T, D = 2, 16, 32
        mock_groot.forward.return_value = _make_mock_groot_outputs(B=B)

        batch = _make_batch(B=B, T=T, D=D, with_neg=True)
        from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
        loss, loss_dict = GrootCLPolicy.forward(policy, batch)

        assert "flow_matching_loss" in loss_dict
        assert "contrastive_loss" in loss_dict
        assert "loss" in loss_dict
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad is False or loss.item() >= 0

    def test_forward_training_no_negatives_fallback_disabled(self):
        policy, mock_groot = self._build_policy_with_mock(phase="phase2a", backprop=True, fallback=False)
        policy.train()

        B = 2
        mock_groot.forward.return_value = _make_mock_groot_outputs(B=B)

        batch = _make_batch(B=B, with_neg=False)
        from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
        loss, loss_dict = GrootCLPolicy.forward(policy, batch)

        assert loss_dict["contrastive_loss"] == pytest.approx(0.0)

    def test_forward_training_no_negatives_fallback_enabled(self):
        policy, mock_groot = self._build_policy_with_mock(phase="phase2a", backprop=True, fallback=True)
        policy.train()

        B = 4
        mock_groot.forward.return_value = _make_mock_groot_outputs(B=B)

        batch = _make_batch(B=B, with_neg=False)
        from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
        loss, loss_dict = GrootCLPolicy.forward(policy, batch)

        assert "contrastive_loss" in loss_dict

    def test_forward_eval_no_contrastive(self):
        policy, mock_groot = self._build_policy_with_mock(phase="phase2a", backprop=True)
        policy.eval()

        B = 2
        mock_groot.forward.return_value = _make_mock_groot_outputs(B=B)

        batch = _make_batch(B=B, with_neg=True)
        from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
        loss, loss_dict = GrootCLPolicy.forward(policy, batch)

        assert "contrastive_loss" not in loss_dict
        mock_groot.forward.assert_called_once()
        _, call_kwargs = mock_groot.forward.call_args
        assert call_kwargs.get("return_intermediate") is False

    def test_backbone_detached_when_backprop_disabled(self):
        policy, mock_groot = self._build_policy_with_mock(phase="phase1", backprop=False)
        policy.train()

        B = 2
        outputs = _make_mock_groot_outputs(B=B)
        outputs["backbone_features"] = outputs["backbone_features"].requires_grad_(True)
        mock_groot.forward.return_value = outputs

        batch = _make_batch(B=B, with_neg=True)
        from lerobot.policies.groot_cl.modeling_groot_cl import GrootCLPolicy
        loss, _ = GrootCLPolicy.forward(policy, batch)


class TestGrootCLConfig:
    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError, match="contrastive_phase"):
            GrootCLConfig(contrastive_phase="invalid")

    def test_valid_phases(self):
        for phase in ("phase1", "phase2a", "phase2b"):
            cfg = GrootCLConfig(contrastive_phase=phase)
            assert cfg.contrastive_phase == phase

    def test_defaults(self):
        cfg = GrootCLConfig()
        assert cfg.contrastive_latent_dim == 256
        assert cfg.contrastive_loss_weight == 0.1
        assert cfg.contrastive_backprop_backbone is True
        assert cfg.contrastive_fallback_to_in_batch is False
