# LLM Inference Engine Router - Colab-Friendly Starter

This project is a practical starter implementation for the take-home assignment.
It lets you:

- Run benchmark experiments across engine/model/input/output/batch configurations.
- Save all runs to `data/benchmark_results.csv`.
- Route an incoming request to the best engine based on benchmark data.
- Use either CLI or FastAPI for the router interface.

## 1) Assignment Coverage

- Part 1 (Benchmark): implemented via `scripts/run_benchmarks.py`.
- Part 2 (Router): implemented via `scripts/router_cli.py` and `scripts/router_api.py`.
- Deliverables:
  - CSV output: `data/benchmark_results.csv`
  - Full code: this repository
  - README: this file
  - Optional chart: generate in notebook or Python script from CSV

## 2) Colab Setup (No Local GPU Needed)

In Google Colab:

1. Runtime -> Change runtime type -> GPU (if available).
2. Clone/upload this project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run benchmark (stub engines for starter validation):

```bash
python -m scripts.run_benchmarks --use-stub
```

5. Run router CLI:

```bash
python -m scripts.router_cli \
  --benchmark-csv data/benchmark_results.csv \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-tokens 500 \
  --max-output-tokens 500 \
  --batch-size 4 \
  --priority latency
```

6. Run router API:

```bash
uvicorn scripts.router_api:app --host 0.0.0.0 --port 8000
```

## 3) Project Structure

- `src/benchmark/`: benchmark matrix and execution harness
- `src/engines/`: engine abstraction and starter adapter factory
- `src/router/`: decision logic for routing
- `src/common/`: shared models and utility code
- `scripts/`: executable entrypoints
- `data/`: benchmark CSV outputs

## 4) What Is Stubbed vs Real

Current code uses `StubEngineAdapter` for quick end-to-end development.
This is intentional so you can run everything immediately on free Colab.

To complete the final assignment with real inference engines:

- Replace `build_engine_registry(use_stub=False)` in `src/engines/factory.py`.
- Add real clients for:
  - vLLM
  - SGLang
  - llama.cpp
- Ensure each adapter returns:
  - TTFT
  - TPOT
  - throughput
  - peak memory

## 5) Router Decisioning Logic

The router currently does a benchmark-lookup strategy:

- Convert request token counts into assignment buckets:
  - input: short/medium/long
  - output: brief/standard/long
- Filter benchmark rows by:
  - model
  - input/output bucket
  - batch size
  - optional memory limit
- Choose best engine:
  - `priority=latency`: minimize `ttft_ms + tpot_ms`
  - `priority=throughput`: maximize `throughput_tok_per_sec`
- Return selected engine, predicted metrics, and reason string.

This is simple, deterministic, and easy to explain in your write-up.

## 6) Suggested Next Steps to Finalize Submission

1. Integrate real engines in `src/engines`.
2. Run at least one full matrix for two models on Colab GPU.
3. Save `data/benchmark_results.csv` from real runs.
4. Add one visualization chart from the CSV.
5. Document runtime limits and assumptions in analysis section.

## 7) Example JSON I/O for Router

Input:

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "input_tokens": 500,
  "max_output_tokens": 500,
  "batch_size": 4,
  "priority": "latency"
}
```

Output:

```json
{
  "engine": "vllm",
  "predicted_metrics": {
    "ttft_ms": 45.0,
    "tpot_ms": 12.0,
    "throughput_tok_per_sec": 280.0,
    "peak_memory_mb": 2048.0
  },
  "reason": "vllm selected for latency (12.4% lower TTFT+TPOT score)."
}
```
