from __future__ import annotations


def build_prompt_for_token_target(target_tokens: int) -> str:
    # This synthetic prompt is intentionally repetitive for predictable token length growth.
    seed = "machine learning benchmark prompt "
    words_needed = max(target_tokens, 1)
    return (seed * words_needed).strip()


def bucket_from_input_tokens(token_count: int) -> str:
    if token_count <= 100:
        return "short"
    if token_count <= 1000:
        return "medium"
    return "long"


def bucket_from_output_tokens(token_count: int) -> str:
    if token_count <= 100:
        return "brief"
    if token_count <= 1000:
        return "standard"
    return "long"
