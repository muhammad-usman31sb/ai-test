from __future__ import annotations

from typing import Dict

from src.engines.base import EngineAdapter
from src.engines.stub_impl import StubEngineAdapter


SUPPORTED_ENGINES = ("vllm", "sglang", "llama_cpp")


def build_engine_registry(use_stub: bool = True) -> Dict[str, EngineAdapter]:
    if use_stub:
        return {name: StubEngineAdapter(name) for name in SUPPORTED_ENGINES}

    raise NotImplementedError(
        "Real engine clients are not wired yet. Set --use-stub to run locally/Colab first."
    )
