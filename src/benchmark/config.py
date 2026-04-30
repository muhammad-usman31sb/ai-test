from __future__ import annotations

INPUT_LENGTHS = {
    "short": 10,
    "medium": 500,
    "long": 2000,
}

OUTPUT_LENGTHS = {
    "brief": 50,
    "standard": 500,
    "long": 1500,
}

DEFAULT_BATCH_SIZES = [1, 4, 8]
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]
