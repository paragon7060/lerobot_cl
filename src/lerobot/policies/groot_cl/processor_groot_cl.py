from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.policies.groot_cl.configuration_groot_cl import GrootCLConfig
from lerobot.policies.groot_cl.processor_groot import make_groot_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@dataclass
@ProcessorStepRegistry.register(name="groot_cl_negative_action_normalize_v1")
class NegativeActionNormalizeStep(ProcessorStep):
    stats: dict[str, dict[str, Any]] | None = None
    normalize_min_max: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        negative_action = transition.get("negative_action")
        if not isinstance(negative_action, torch.Tensor):
            return transition

        if not self.normalize_min_max or self.stats is None or ACTION not in self.stats:
            return transition

        stats_k = self.stats[ACTION]
        d = negative_action.shape[-1]

        min_v = torch.as_tensor(
            stats_k.get("min", torch.zeros(d)), dtype=negative_action.dtype, device=negative_action.device
        ).flatten()
        max_v = torch.as_tensor(
            stats_k.get("max", torch.ones(d)), dtype=negative_action.dtype, device=negative_action.device
        ).flatten()

        if min_v.numel() != d:
            min_v = torch.nn.functional.pad(min_v[:d], (0, max(0, d - min_v.numel())))
        if max_v.numel() != d:
            max_v = torch.nn.functional.pad(max_v[:d], (0, max(0, d - max_v.numel())))

        denom = max_v - min_v
        mask = denom != 0
        safe_denom = torch.where(mask, denom, torch.ones_like(denom))
        mapped = 2 * (negative_action - min_v) / safe_denom - 1
        transition["negative_action"] = torch.where(mask, mapped, torch.zeros_like(mapped))
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        if not self.stats:
            return {}
        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                flat[f"{key}.{stat_name}"] = torch.as_tensor(value).cpu()
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            return
        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                reconstructed.setdefault(key, {})[stat_name] = tensor
        if reconstructed:
            self.stats = reconstructed


def make_groot_cl_pre_post_processors(
    config: GrootCLConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    pre, post = make_groot_pre_post_processors(config, dataset_stats)

    pre.steps.append(
        NegativeActionNormalizeStep(
            stats=dataset_stats or {},
            normalize_min_max=True,
        )
    )

    return pre, post
