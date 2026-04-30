from __future__ import annotations

import random

from src.common.models import InferenceMetrics
from src.engines.base import EngineAdapter


class StubEngineAdapter(EngineAdapter):
    def __init__(self, engine_name: str, seed: int = 7) -> None:
        self._name = engine_name
        self._rng = random.Random(seed + hash(engine_name) % 1000)

    @property
    def name(self) -> str:
        return self._name

    def run_inference(
        self,
        model: str,
        prompt: str,
        max_output_tokens: int,
        batch_size: int,
    ) -> InferenceMetrics:
        base_ttft = {
            "vllm": 45,
            "sglang": 52,
            "llama_cpp": 68,
        }.get(self._name, 60)

        jitter = self._rng.uniform(-8, 8)
        ttft_ms = max(8.0, base_ttft + jitter + batch_size * 2)

        base_tpot = {
            "vllm": 8,
            "sglang": 9.5,
            "llama_cpp": 13,
        }.get(self._name, 10)
        tpot_ms = max(1.0, base_tpot + self._rng.uniform(-1.5, 1.5) + batch_size * 0.6)

        output_tokens = max_output_tokens
        token_time_ms = max(1.0, (output_tokens - 1) * tpot_ms)
        total_time_ms = ttft_ms + token_time_ms
        throughput = output_tokens / max(total_time_ms / 1000.0, 1e-6)

        base_mem = {
            "vllm": 2400,
            "sglang": 2200,
            "llama_cpp": 1800,
        }.get(self._name, 2000)
        peak_memory_mb = max(500.0, base_mem + batch_size * 140 + self._rng.uniform(-120, 120))

        return InferenceMetrics(
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            throughput_tok_per_sec=throughput,
            peak_memory_mb=peak_memory_mb,
            total_time_ms=total_time_ms,
            output_tokens=output_tokens,
        )
