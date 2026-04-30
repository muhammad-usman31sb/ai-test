from __future__ import annotations

import time
from typing import Any

from src.common.models import InferenceMetrics
from src.engines.base import EngineAdapter

# Default GGUF model map: HuggingFace model id -> (repo_id, filename)
_GGUF_MAP = {
    "Qwen/Qwen2.5-0.5B-Instruct": (
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ),
    "meta-llama/Llama-3.2-1B-Instruct": (
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    ),
}


def _resolve_model_path(model: str) -> str:
    """Download GGUF from HF Hub if needed; return local path."""
    from huggingface_hub import hf_hub_download  # type: ignore[import]

    if model in _GGUF_MAP:
        repo_id, filename = _GGUF_MAP[model]
    else:
        raise ValueError(
            f"No GGUF mapping for model '{model}'. "
            "Add an entry to _GGUF_MAP in llama_cpp_impl.py."
        )

    return hf_hub_download(repo_id=repo_id, filename=filename)


class LlamaCppEngineAdapter(EngineAdapter):
    """
    Real llama.cpp adapter using llama-cpp-python.

    TTFT is measured precisely by recording the time to the first
    streamed token chunk. TPOT is computed from per-token timestamps.
    """

    def __init__(self) -> None:
        self._llm_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "llama_cpp"

    def _get_llm(self, model: str) -> Any:
        if model not in self._llm_cache:
            from llama_cpp import Llama  # type: ignore[import]

            path = _resolve_model_path(model)
            self._llm_cache[model] = Llama(
                model_path=path,
                n_gpu_layers=-1,  # offload all layers to GPU if available
                n_ctx=4096,
                verbose=False,
            )
        return self._llm_cache[model]

    def run_inference(
        self,
        model: str,
        prompt: str,
        max_output_tokens: int,
        batch_size: int,
    ) -> InferenceMetrics:
        import psutil  # type: ignore[import]

        llm = self._get_llm(model)

        # llama_cpp does not support native batching; simulate via sequential runs
        # and report aggregate totals to stay consistent with other engines.
        total_output_tokens = 0
        total_time_ms = 0.0
        ttft_sum_ms = 0.0
        tpot_sum_ms = 0.0
        peak_memory_mb = 0.0

        process = psutil.Process()

        for _ in range(batch_size):
            t_start = time.perf_counter()
            ttft_ms: float | None = None
            token_count = 0

            for chunk in llm(
                prompt,
                max_tokens=max_output_tokens,
                stream=True,
            ):
                now = time.perf_counter()
                if ttft_ms is None:
                    ttft_ms = (now - t_start) * 1000
                token_count += 1

            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000
            ttft_ms = ttft_ms or elapsed_ms

            decode_time_ms = max(elapsed_ms - ttft_ms, 1.0)
            tpot_ms = decode_time_ms / max(token_count - 1, 1)

            total_output_tokens += token_count
            total_time_ms += elapsed_ms
            ttft_sum_ms += ttft_ms
            tpot_sum_ms += tpot_ms

            mem_mb = process.memory_info().rss / 1024**2
            peak_memory_mb = max(peak_memory_mb, mem_mb)

        throughput = total_output_tokens / max(total_time_ms / 1000.0, 1e-6)

        return InferenceMetrics(
            ttft_ms=ttft_sum_ms / batch_size,
            tpot_ms=tpot_sum_ms / batch_size,
            throughput_tok_per_sec=throughput,
            peak_memory_mb=peak_memory_mb,
            total_time_ms=total_time_ms,
            output_tokens=total_output_tokens,
        )
