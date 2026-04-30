from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.models import InferenceMetrics


class EngineAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run_inference(
        self,
        model: str,
        prompt: str,
        max_output_tokens: int,
        batch_size: int,
    ) -> InferenceMetrics:
        raise NotImplementedError
