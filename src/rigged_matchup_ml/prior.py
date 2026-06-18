from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


PriorProvider = Callable[[dict[str, Any]], float]


def load_prior_provider(dotted_path: str | None) -> PriorProvider:
    if not dotted_path:
        return lambda _record: 0.5
    module_name, separator, attribute = dotted_path.partition(":")
    if not separator:
        raise ValueError("matrix_prior_provider must use 'module:function' syntax")
    provider = getattr(importlib.import_module(module_name), attribute)
    if not callable(provider):
        raise TypeError(f"{dotted_path} is not callable")
    return provider
