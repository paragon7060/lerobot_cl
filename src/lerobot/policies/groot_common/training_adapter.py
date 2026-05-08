"""Forward / unwrap / sanity helpers shared by GR00T-family training scripts.

These helpers let `train_groot_robocasa_official.py` drive any of
`groot_robocasa`, `groot_cl`, `groot_mgd`, `groot_cl_v2` through the official
batch path without having to special-case each policy's `forward()` signature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from accelerate import Accelerator

    from .robocasa_preset import RobocasaPreset


# Policy `name` strings supported by this training path. Used by the script's
# `--policy.type` validator and `assert_groot_compatible`.
GROOT_POLICY_TYPES: tuple[str, ...] = (
    "groot_robocasa",
    "groot_cl",
    "groot_mgd",
    "groot_cl_v2",
)


def assert_groot_compatible(policy: Any) -> None:
    """Verify the policy exposes the GR00T contract this adapter relies on."""
    if not hasattr(policy, "_groot_model"):
        raise TypeError(
            f"Policy {type(policy).__name__} does not expose `_groot_model`; "
            "official-style training requires a GR00T-family policy."
        )
    name = getattr(policy, "name", None)
    if name not in GROOT_POLICY_TYPES:
        raise TypeError(
            f"Policy name {name!r} is not in the supported set {GROOT_POLICY_TYPES}. "
            "Add it to GROOT_POLICY_TYPES if it is GR00T-compatible."
        )


def forward_with_groot_batch(policy: Any, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
    """Run one official-batch forward pass.

    Priority:
      1. `policy.forward_official_batch(batch)` — opt-in hook for experimental
         policies (`groot_cl`, `groot_mgd`, `groot_cl_v2`) that need to keep
         their custom losses (contrastive, MGD, etc.) when training on the
         official batch layout.
      2. Fallback: `policy._groot_model(batch)["loss"]`. Mirrors what
         Isaac-GR00T's `DualBrainTrainer.compute_loss` does.
    """
    if hasattr(policy, "forward_official_batch"):
        out = policy.forward_official_batch(batch)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError(
                "forward_official_batch must return (loss, metrics_dict); "
                f"got {type(out).__name__}"
            )
        return out

    outputs = policy._groot_model(batch)
    if "loss" not in outputs:
        raise KeyError(
            "GR00T model forward did not return a 'loss' key. "
            f"Available keys: {list(outputs.keys())}"
        )
    loss = outputs["loss"]
    return loss, {"loss": loss.detach()}


def unwrap_groot_model(policy_or_wrapped: Any, accelerator: "Accelerator | None" = None):
    """Return the underlying GR00TN15 model regardless of DDP / Accelerator wrap.

    Always go through `accelerator.unwrap_model` once accelerator wrapping has
    been applied, otherwise checkpoint saving silently writes the DDP wrapper's
    state dict and breaks reload.
    """
    target = policy_or_wrapped
    if accelerator is not None:
        target = accelerator.unwrap_model(target)
    if not hasattr(target, "_groot_model"):
        raise AttributeError(
            f"Unwrapped object {type(target).__name__} has no `_groot_model`."
        )
    return target._groot_model


def _count_trainable(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def assert_official_config_match(policy: Any, preset: "RobocasaPreset") -> None:
    """Print the policy's effective dims and fail fast on preset mismatch.

    Catches the silent footgun where `--policy.type=groot_cl` keeps the parent
    `chunk_size=50` default and trains against an action-horizon-16 head.
    """
    groot_model = policy._groot_model
    action_head = groot_model.action_head
    action_horizon = int(action_head.action_horizon)
    action_dim = int(getattr(action_head.config, "action_dim", preset.max_action_dim))
    state_dim = int(getattr(action_head.config, "max_state_dim", preset.max_state_dim))

    trainable, total = _count_trainable(policy)
    print(
        "[groot_common] policy=%s | action_horizon=%d | action_dim=%d | state_dim=%d | "
        "trainable=%d / %d (%.2f%%)"
        % (
            getattr(policy, "name", type(policy).__name__),
            action_horizon,
            action_dim,
            state_dim,
            trainable,
            total,
            100.0 * trainable / max(total, 1),
        )
    )

    mismatches: list[str] = []
    if action_horizon != preset.chunk_size:
        mismatches.append(
            f"action_horizon={action_horizon} but preset.chunk_size={preset.chunk_size}"
        )
    if action_horizon != preset.n_action_steps:
        mismatches.append(
            f"action_horizon={action_horizon} but preset.n_action_steps={preset.n_action_steps}"
        )
    if action_dim != preset.max_action_dim:
        mismatches.append(
            f"action head action_dim={action_dim} but preset.max_action_dim={preset.max_action_dim}"
        )
    if state_dim != preset.max_state_dim:
        mismatches.append(
            f"action head max_state_dim={state_dim} but preset.max_state_dim={preset.max_state_dim}"
        )

    if mismatches:
        raise RuntimeError(
            "Official preset / policy config mismatch — refusing to start training:\n  "
            + "\n  ".join(mismatches)
        )
