from __future__ import annotations

import time
from typing import Any

from src.common.models import InferenceMetrics
from src.engines.base import EngineAdapter


class VLLMEngineAdapter(EngineAdapter):
    """
    Real vLLM adapter using the offline LLM class.

    TTFT is measured by timing a single-token prefill-only generation,
    which is a standard proxy for prefill latency. TPOT is derived from
    the remaining decode time spread over output tokens.
    """

    def __init__(self, model: str | None = None) -> None:
        self._default_model = model
        self._llm_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "vllm"

    def _get_llm(self, model: str) -> Any:
        if model not in self._llm_cache:
            from vllm import LLM  # type: ignore[import]

            self._llm_cache[model] = LLM(model=model, trust_remote_code=True)
        return self._llm_cache[model]

    def run_inference(
        self,
        model: str,
        prompt: str,
        max_output_tokens: int,
        batch_size: int,
    ) -> InferenceMetrics:
        import torch  # type: ignore[import]
        from vllm import SamplingParams  # type: ignore[import]

        llm = self._get_llm(model)
        prompts = [prompt] * batch_size

        # --- TTFT proxy: time a single-token prefill on one prompt ---
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        llm.generate([prompt], SamplingParams(max_tokens=1))
        ttft_ms = (time.perf_counter() - t0) * 1000

        # --- Full generation ---
        torch.cuda.reset_peak_memory_stats()
        t_gen_start = time.perf_counter()
        outputs = llm.generate(prompts, SamplingParams(max_tokens=max_output_tokens))
        total_time_ms = (time.perf_counter() - t_gen_start) * 1000

        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

        output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
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
