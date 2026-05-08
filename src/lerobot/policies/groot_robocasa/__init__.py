#!/usr/bin/env python

from .configuration_groot import GrootRobocasaConfig


def make_groot_pre_post_processors(*args, **kwargs):
    from .processor_groot import make_groot_pre_post_processors as _impl

    return _impl(*args, **kwargs)


def __getattr__(name):
    if name == "GrootRobocasaPolicy":
        from .modeling_groot import GrootRobocasaPolicy

        return GrootRobocasaPolicy
    raise AttributeError(name)


__all__ = ["GrootRobocasaConfig", "GrootRobocasaPolicy", "make_groot_pre_post_processors"]
