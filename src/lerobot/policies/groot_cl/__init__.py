from .configuration_groot import GrootConfig
from .configuration_groot_cl import GrootCLConfig


def make_groot_pre_post_processors(*args, **kwargs):
    from .processor_groot import make_groot_pre_post_processors as _impl

    return _impl(*args, **kwargs)


def make_groot_cl_pre_post_processors(*args, **kwargs):
    from .processor_groot_cl import make_groot_cl_pre_post_processors as _impl

    return _impl(*args, **kwargs)


def __getattr__(name):
    if name == "GrootPolicy":
        from .modeling_groot import GrootPolicy

        return GrootPolicy
    if name == "GrootCLPolicy":
        from .modeling_groot_cl import GrootCLPolicy

        return GrootCLPolicy
    raise AttributeError(name)


__all__ = [
    "GrootConfig",
    "GrootCLConfig",
    "GrootPolicy",
    "GrootCLPolicy",
    "make_groot_pre_post_processors",
    "make_groot_cl_pre_post_processors",
]
