from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.common.models import RouterRequest
from src.router.decision import response_to_dict, route_request

app = FastAPI(title="LLM Engine Router", version="0.1.0")

BENCHMARK_CSV_PATH = "data/benchmark_results.csv"


class RouterInput(BaseModel):
    model: str = Field(..., examples=["Qwen/Qwen2.5-0.5B-Instruct"])
    input_tokens: int = Field(..., ge=1)
    max_output_tokens: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    priority: str = Field("latency", pattern="^(latency|throughput)$")
    memory_limit_mb: float | None = Field(default=None, ge=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/route")
def route(payload: RouterInput) -> dict:
    req = RouterRequest(
        model=payload.model,
        input_tokens=payload.input_tokens,
        max_output_tokens=payload.max_output_tokens,
        batch_size=payload.batch_size,
        priority=payload.priority,
        memory_limit_mb=payload.memory_limit_mb,
    )
    resp = route_request(req=req, benchmark_csv_path=BENCHMARK_CSV_PATH)
    return response_to_dict(resp)
