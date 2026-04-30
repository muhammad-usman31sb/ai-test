from __future__ import annotations

import csv
from dataclasses import asdict
from typing import Dict, List

from src.common.models import RouterRequest, RouterResponse
from src.common.token_utils import bucket_from_input_tokens, bucket_from_output_tokens


_METRIC_KEYS = ["ttft_ms", "tpot_ms", "throughput_tok_per_sec", "peak_memory_mb"]


def _load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_best(rows: List[Dict[str, str]], priority: str) -> Dict[str, str]:
    if priority == "throughput":
        return max(rows, key=lambda r: float(r["throughput_tok_per_sec"]))

    # Default behavior optimizes latency by minimizing weighted TTFT + TPOT.
    return min(rows, key=lambda r: float(r["ttft_ms"]) + float(r["tpot_ms"]))


def route_request(req: RouterRequest, benchmark_csv_path: str) -> RouterResponse:
    input_bucket = bucket_from_input_tokens(req.input_tokens)
    output_bucket = bucket_from_output_tokens(req.max_output_tokens)

    rows = _load_rows(benchmark_csv_path)

    filtered = [
        r
        for r in rows
        if r.get("status") == "ok"
        and r.get("model") == req.model
        and r.get("input_bucket") == input_bucket
        and r.get("output_bucket") == output_bucket
        and int(r.get("batch_size", 0)) == req.batch_size
    ]

    if req.memory_limit_mb is not None:
        filtered = [r for r in filtered if float(r["peak_memory_mb"]) <= req.memory_limit_mb]

    if not filtered:
        raise ValueError("No benchmark rows match request constraints")

    best = _pick_best(filtered, req.priority)

    engine = best["engine"]
    metrics = {k: float(best[k]) for k in _METRIC_KEYS}

    if req.priority == "throughput":
        alt = sorted(filtered, key=lambda r: float(r["throughput_tok_per_sec"]), reverse=True)
        second = alt[1] if len(alt) > 1 else alt[0]
        denom = max(float(second["throughput_tok_per_sec"]), 1e-6)
        gain = (float(best["throughput_tok_per_sec"]) - float(second["throughput_tok_per_sec"])) / denom * 100
        reason = f"{engine} selected for throughput (+{gain:.1f}% vs next best)."
    else:
        alt = sorted(filtered, key=lambda r: float(r["ttft_ms"]) + float(r["tpot_ms"]))
        second = alt[1] if len(alt) > 1 else alt[0]
        best_score = float(best["ttft_ms"]) + float(best["tpot_ms"])
        second_score = float(second["ttft_ms"]) + float(second["tpot_ms"])
        denom = max(second_score, 1e-6)
        gain = (second_score - best_score) / denom * 100
        reason = f"{engine} selected for latency ({gain:.1f}% lower TTFT+TPOT score)."

    return RouterResponse(engine=engine, predicted_metrics=metrics, reason=reason)


def response_to_dict(resp: RouterResponse) -> Dict[str, object]:
    return asdict(resp)
