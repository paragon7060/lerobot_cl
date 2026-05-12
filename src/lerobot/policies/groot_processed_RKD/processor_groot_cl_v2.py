"""Processor entrypoint for `policy.type=groot_processed_rkd`.

Factory dynamic loader expects this module path from
`configuration_groot_cl_v2.GrootCLv2Config` and looks up
`make_groot_processed_rkd_pre_post_processors`.
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.groot_processed_RKD.configuration_groot_cl_v2 import GrootCLv2Config


def make_groot_processed_rkd_pre_post_processors(
    config: GrootCLv2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Reuse GR00T pre/post processor stack for processed-RKD policy."""
    from lerobot.policies.groot_cl.processor_groot import make_groot_pre_post_processors

    return make_groot_pre_post_processors(config=config, dataset_stats=dataset_stats)

