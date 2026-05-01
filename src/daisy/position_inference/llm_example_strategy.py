"""Compatibility alias for the example-aware position strategy."""

from pathlib import Path
from typing import Any

from src.config import PositionInfererConfig
from src.daisy.position_inference.base import register_position_strategy
from src.daisy.position_inference.llm_strategy import LLMPositionStrategy
from src.llm.llm_configurations import LLM


@register_position_strategy("LLM_EXAMPLE")
class LLMExamplePositionStrategy(LLMPositionStrategy):
    """Backward-compatible alias that keeps the LLM_EXAMPLE registry entry."""

    def __init__(
        self,
        llm: LLM,
        config: PositionInfererConfig,
        cache_dir: Path | None = None,
        **kwargs: Any,
    ):
        super().__init__(llm=llm, config=config, cache_dir=cache_dir, **kwargs)
        self.name = "LLM_EXAMPLE"

