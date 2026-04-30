from __future__ import annotations

import time
from typing import Any

from src.common.models import InferenceMetrics
from src.engines.base import EngineAdapter


class SGLangEngineAdapter(EngineAdapter):
    """
    Real SGLang adapter using sglang.Engine (offline runtime).

    SGLang's Engine class supports synchronous generation with output
    metadata. TTFT is approximated via a single-token prefill run,
    consistent with the vLLM adapter methodology.
    """

    def __init__(self) -> None:
        self._engine_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "sglang"

    def _get_engine(self, model: str) -> Any:
        if model not in self._engine_cache:
            import sglang as sgl  # type: ignore[import]

            engine = sgl.Engine(model_path=model, trust_remote_code=True)
            self._engine_cache[model] = engine
        return self._engine_cache[model]

    def run_inference(
        self,
        model: str,
        prompt: str,
        max_output_tokens: int,
        batch_size: int,
    ) -> InferenceMetrics:
        import torch  # type: ignore[import]

        engine = self._get_engine(model)
        sampling_params = {"max_new_tokens": max_output_tokens}

        # --- TTFT proxy: single-token prefill ---
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        engine.generate(prompt=[prompt], sampling_params={"max_new_tokens": 1})
        ttft_ms = (time.perf_counter() - t0) * 1000

        # --- Full generation ---
        torch.cuda.reset_peak_memory_stats()
        prompts = [prompt] * batch_size
        t_gen = time.perf_counter()
        outputs = engine.generate(prompt=prompts, sampling_params=sampling_params)
        total_time_ms = (time.perf_counter() - t_gen) * 1000

        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

        # outputs is a list of dicts with key "text"
        output_tokens = sum(
            len(o.get("meta_info", {}).get("completion_tokens", []) or o["text"].split())
            for o in outputs
        )
        if output_tokens == 0:
            output_tokens = sum(len(o["text"].split()) for o in outputs)

        avg_output_tokens = output_tokens / batch_size
        decode_time_ms = max(total_time_ms - ttft_ms, 1.0)
        tpot_ms = decode_time_ms / max(avg_output_tokens - 1, 1)
        throughput = output_tokens / max(total_time_ms / 1000.0, 1e-6)

        return InferenceMetrics(
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            throughput_tok_per_sec=throughput,
            peak_memory_mb=peak_memory_mb,
            total_time_ms=total_time_ms,
            output_tokens=output_tokens,
        )
