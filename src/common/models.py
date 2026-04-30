from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class BenchmarkCase:
    engine: str
    model: str
    input_bucket: str
    output_bucket: str
    input_tokens_target: int
    max_output_tokens: int
    batch_size: int
    repeat_idx: int


@dataclass
class BenchmarkResult:
    timestamp_utc: str
    engine: str
    model: str
    input_bucket: str
    output_bucket: str
    input_tokens_target: int
    max_output_tokens: int
    batch_size: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    throughput_tok_per_sec: float
    peak_memory_mb: float
    total_time_ms: float
    status: str
    error_message: str

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceMetrics:
    ttft_ms: float
    tpot_ms: float
    throughput_tok_per_sec: float
    peak_memory_mb: float
    total_time_ms: float
    output_tokens: int


@dataclass
class RouterRequest:
    model: str
    input_tokens: int
    max_output_tokens: int
    batch_size: int
    priority: str
    memory_limit_mb: Optional[float] = None


@dataclass
class RouterResponse:
    engine: str
    predicted_metrics: Dict[str, float]
    reason: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
