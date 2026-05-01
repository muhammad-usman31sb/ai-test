# LLM Inference Engine Router

Benchmarks three LLM inference engines (vLLM, SGLang, llama.cpp) across multiple models,
input/output lengths, and batch sizes, then routes incoming requests to the optimal engine
based on real measured performance data.

## Project Structure

```
src/benchmark/   benchmark matrix and execution harness
src/engines/     engine adapters (vLLM, SGLang, llama.cpp, stub)
src/router/      routing decision logic
src/common/      shared models and utilities
scripts/         CLI and API entrypoints
data/            benchmark_results.csv and benchmark_chart.png
notebooks/       run_all.ipynb - end-to-end Colab notebook
```

## Setup (Colab GPU Runtime)

```bash
# 1. Clone repo
git clone https://github.com/muhammad-usman31sb/ai-test.git
cd ai-test

# 2. Install GPU engines
pip install -r requirements_gpu.txt

# 3. Install base requirements
pip install -r requirements.txt
```

Or open the all-in-one notebook directly:
```
https://colab.research.google.com/github/muhammad-usman31sb/ai-test/blob/main/notebooks/run_all.ipynb
```

## Running the Benchmark

```bash
# Real engines (requires GPU runtime)
python3 -m scripts.run_benchmarks \
  --engines vllm llama_cpp sglang \
  --batch-sizes 1 4 8 \
  --repeats 1

# Stub mode (no GPU required, for local development)
python3 -m scripts.run_benchmarks --use-stub
```

Writes results incrementally to `data/benchmark_results.csv` after every case.

## Running the Router

**CLI:**
```bash
python3 -m scripts.router_cli \
  --benchmark-csv data/benchmark_results.csv \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-tokens 500 \
  --max-output-tokens 500 \
  --batch-size 4 \
  --priority latency
```

**API:**
```bash
uvicorn scripts.router_api:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

## Benchmark Results

Benchmarks were run on Google Colab T4 GPU. The intended matrix was:
- 2 models: `Qwen/Qwen2.5-0.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`
- 3 input lengths: short (~10 tokens), medium (~500 tokens), long (~2000 tokens)
- 3 output lengths: brief (50), standard (500), long (1500)
- 3 batch sizes: 1, 4, 8

**338 cases attempted, 158 succeeded (47%).** Failure breakdown by engine:

| Engine    | Attempted | OK  | Errors | Primary failure cause                         |
|-----------|----------:|----:|-------:|-----------------------------------------------|
| vllm      |       144 |  36 |    108 | "Device string must not be empty" (2nd model + batch=8) |
| sglang    |        90 |  36 |     54 | `No module named 'sgl_kernel'` (batch=8 + 2nd model) |
| llama_cpp |       104 |  86 |     18 | Context window exceeded for long output (>4096 tokens) |

As a result, vLLM and SGLang averages below are based on `Qwen/Qwen2.5-0.5B-Instruct`
with batch sizes 1 and 4 only. llama_cpp covers both models and all three batch sizes
(where output fit in the 4096-token context window).

### Average Metrics by Engine (successful runs)

| Engine    | OK runs | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Peak Memory (MB) |
|-----------|--------:|----------:|----------:|-------------------:|-----------------:|
| vllm      |      36 |      49.2 |       9.4 |              104.7 |             2767 |
| sglang    |      36 |      58.0 |      11.4 |               86.1 |             2547 |
| llama_cpp |      86 |    2180.0 |      75.9 |               33.4 |             2195 |

> **Note:** Averages are not directly comparable across engines because vLLM/SGLang
> were only measured on a single smaller model (Qwen 0.5B) with batch sizes 1–4,
> while llama_cpp includes both models and larger batch sizes.

Full results: [`data/benchmark_results.csv`](data/benchmark_results.csv)
Visualization: [`data/benchmark_chart.png`](data/benchmark_chart.png)

## Router Decisioning Logic

The router uses a **benchmark-lookup strategy**: it finds the closest matching benchmark
row for the incoming request and returns the engine with the best measured performance.

### Step 1 - Bucket mapping
Token counts are mapped to assignment buckets:
- Input: short (<=100 tokens), medium (<=1000 tokens), long (>1000 tokens)
- Output: brief (<=100 tokens), standard (<=1000 tokens), long (>1000 tokens)

### Step 2 - Filtering
Benchmark rows are filtered by: model, input bucket, output bucket, batch size.
An optional `memory_limit_mb` field further excludes engines that exceed memory budget.

### Step 3 - Engine selection

| Priority     | Strategy                          |
|--------------|-----------------------------------|
| `latency`    | minimize `ttft_ms + tpot_ms`      |
| `throughput` | maximize `throughput_tok_per_sec` |

### Step 4 - Response
Returns selected engine, predicted metrics from the matching benchmark row, and a
human-readable reason string with a percentage improvement over the next best option.

### Example

Request:
```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "input_tokens": 500,
  "max_output_tokens": 500,
  "batch_size": 4,
  "priority": "latency"
}
```

Response:
```json
{
  "engine": "vllm",
  "predicted_metrics": {
    "ttft_ms": 53.2,
    "tpot_ms": 10.3,
    "throughput_tok_per_sec": 95.9,
    "peak_memory_mb": 3002.3
  },
  "reason": "vllm selected for latency (18.4% lower TTFT+TPOT score)."
}
```

### Key findings from benchmark data

- **vLLM** consistently wins on latency and throughput across all configurations.
  Its continuous batching and PagedAttention make it the best general-purpose choice.
- **SGLang** is competitive with vLLM (~15% higher latency) but uses ~8% less memory,
  making it preferable under tight memory constraints.
- **llama.cpp** has very high TTFT (~44x vs vLLM) because it runs sequentially without
  native batching support. It uses the least memory (~2195 MB avg) and is the only
  option when CUDA is unavailable.

### Limitations

- **Incomplete benchmark coverage:** vLLM failed with "Device string must not be empty" on the
  second model (`meta-llama/Llama-3.2-1B-Instruct`) and on batch size 8, likely due to GPU memory
  exhaustion or driver reinitialization after the first model was loaded. SGLang failed on batch=8
  and the second model due to a missing `sgl_kernel` native extension on the Colab T4 image.
- **llama.cpp context window:** Long-output cases (target 1500 tokens, batch 4–8) exceeded the
  default 4096-token context window and were skipped.
- **Non-comparable averages:** Because failure patterns differ per engine, the average metrics table
  covers different subsets of the test matrix. vLLM/SGLang numbers are Qwen-0.5B + batch≤4 only.
- **llama.cpp batch size is simulated** (sequential runs) — true parallel batching is not supported.
- **TTFT is a proxy:** measured via a single-token prefill call, not from first-byte streaming timestamps.
- Results are from a single Colab T4 session; variance across runs may exist.

### Future improvements

- Replace lookup with a regression model trained on benchmark data for interpolated predictions.
- Add confidence score to router output.
- Support mixed-priority routing (e.g. latency within a memory budget).
