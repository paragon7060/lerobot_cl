#!/usr/bin/env python

from ..groot.modeling_groot import GrootPolicy
from .configuration_groot import GrootRobocasaConfig


class GrootRobocasaPolicy(GrootPolicy):
    """Robocasa-specialized GR00T policy with default checkpoint/schema."""

    name = "groot_robocasa"
    config_class = GrootRobocasaConfig
