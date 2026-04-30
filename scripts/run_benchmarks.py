from __future__ import annotations

import argparse
import os

from src.benchmark.config import DEFAULT_MODELS
from src.benchmark.runner import build_cases, run_benchmark_cases
from src.engines.factory import SUPPORTED_ENGINES, build_engine_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM engine benchmark matrix")
    parser.add_argument(
        "--output",
        default="data/benchmark_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model list",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=list(SUPPORTED_ENGINES),
        help="Engine list",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8],
        help="Batch sizes",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per case",
    )
    parser.add_argument(
        "--use-stub",
        action="store_true",
        default=True,
        help="Use stubbed engine adapters",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    registry = build_engine_registry(use_stub=args.use_stub)
    cases = build_cases(
        engines=args.engines,
        models=args.models,
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
    )

    results = run_benchmark_cases(
        engine_registry=registry,
        cases=cases,
        output_csv_path=args.output,
    )

    success = sum(1 for r in results if r.status == "ok")
    print(f"Completed {len(results)} cases, success={success}, failed={len(results) - success}")
    print(f"Wrote CSV: {args.output}")


if __name__ == "__main__":
    main()
