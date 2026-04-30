from __future__ import annotations

import argparse
import json

from src.common.models import RouterRequest
from src.router.decision import response_to_dict, route_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route LLM request to best engine")
    parser.add_argument("--benchmark-csv", default="data/benchmark_results.csv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--priority", choices=["latency", "throughput"], default="latency")
    parser.add_argument("--memory-limit-mb", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    req = RouterRequest(
        model=args.model,
        input_tokens=args.input_tokens,
        max_output_tokens=args.max_output_tokens,
        batch_size=args.batch_size,
        priority=args.priority,
        memory_limit_mb=args.memory_limit_mb,
    )

    resp = route_request(req=req, benchmark_csv_path=args.benchmark_csv)
    print(json.dumps(response_to_dict(resp), indent=2))


if __name__ == "__main__":
    main()
