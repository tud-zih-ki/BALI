import importlib
import logging


_FRAMEWORK_SPECS = [
    ("hf_accelerate", ".hf_accelerate", "HFAccelerate"),
    ("vllm", ".vllm", "VLLM"),
    ("vllm_async", ".vllm_async", "VLLM_Async"),
    ("llmlingua", ".llm_lingua", "LLMLingua"),
    ("openllm", ".openLLM", "OpenLLM"),
    ("deepspeed", ".deepspeed_mii", "Deepspeed"),
]

frameworks_available: dict[str, type] = {}

for _identifier, _rel_module, _cls_name in _FRAMEWORK_SPECS:
    try:
        _module = importlib.import_module(_rel_module, package=__name__)
        _cls = getattr(_module, _cls_name)
        frameworks_available[_identifier] = _cls
        globals()[_cls_name] = _cls  # expose at package level
    except Exception as _exc:  # pragma: no cover
        logging.warning(
            "Acceleration framework '%s' unavailable and will be skipped (%s)",
            _identifier,
            _exc,
        )

# What gets imported via "from acceleration_frameworks import *"
__all__ = [cls.__name__ for cls in frameworks_available.values()]
