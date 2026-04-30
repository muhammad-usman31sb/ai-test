from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List

from src.benchmark.config import DEFAULT_BATCH_SIZES, INPUT_LENGTHS, OUTPUT_LENGTHS
from src.common.io_utils import append_csv_rows
from src.common.models import BenchmarkCase, BenchmarkResult, utc_now_iso
from src.common.token_utils import build_prompt_for_token_target
from src.engines.base import EngineAdapter


def build_cases(
    engines: Iterable[str],
    models: Iterable[str],
    batch_sizes: Iterable[int] | None = None,
    repeats: int = 1,
) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []
    selected_batch_sizes = list(batch_sizes) if batch_sizes is not None else DEFAULT_BATCH_SIZES

    for engine in engines:
        for model in models:
            for input_bucket, input_tokens in INPUT_LENGTHS.items():
                for output_bucket, output_tokens in OUTPUT_LENGTHS.items():
                    for batch_size in selected_batch_sizes:
                        for repeat_idx in range(repeats):
                            cases.append(
                                BenchmarkCase(
                                    engine=engine,
                                    model=model,
                                    input_bucket=input_bucket,
                                    output_bucket=output_bucket,
                                    input_tokens_target=input_tokens,
                                    max_output_tokens=output_tokens,
                                    batch_size=batch_size,
                                    repeat_idx=repeat_idx,
                                )
                            )
    return cases


def run_benchmark_cases(
    engine_registry: dict[str, EngineAdapter],
    cases: Iterable[BenchmarkCase],
    output_csv_path: str,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []

    for case in cases:
        adapter = engine_registry[case.engine]
        prompt = build_prompt_for_token_target(case.input_tokens_target)

        try:
            metrics = adapter.run_inference(
                model=case.model,
                prompt=prompt,
                max_output_tokens=case.max_output_tokens,
                batch_size=case.batch_size,
            )
            result = BenchmarkResult(
                timestamp_utc=utc_now_iso(),
                engine=case.engine,
                model=case.model,
                input_bucket=case.input_bucket,
                output_bucket=case.output_bucket,
                input_tokens_target=case.input_tokens_target,
                max_output_tokens=case.max_output_tokens,
                batch_size=case.batch_size,
                output_tokens=metrics.output_tokens,
                ttft_ms=metrics.ttft_ms,
                tpot_ms=metrics.tpot_ms,
                throughput_tok_per_sec=metrics.throughput_tok_per_sec,
                peak_memory_mb=metrics.peak_memory_mb,
                total_time_ms=metrics.total_time_ms,
                status="ok",
                error_message="",
            )
        except Exception as exc:  # noqa: BLE001
            result = BenchmarkResult(
                timestamp_utc=utc_now_iso(),
                engine=case.engine,
                model=case.model,
                input_bucket=case.input_bucket,
                output_bucket=case.output_bucket,
                input_tokens_target=case.input_tokens_target,
                max_output_tokens=case.max_output_tokens,
                batch_size=case.batch_size,
                output_tokens=0,
                ttft_ms=0.0,
                tpot_ms=0.0,
                throughput_tok_per_sec=0.0,
                peak_memory_mb=0.0,
                total_time_ms=0.0,
                status="error",
                error_message=str(exc),
            )

        results.append(result)
        append_csv_rows(output_csv_path, [asdict(result)])

    return results
