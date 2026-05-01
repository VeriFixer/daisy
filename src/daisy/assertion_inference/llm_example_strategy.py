"""Compatibility alias for the example-aware assertion strategy."""

from pathlib import Path
from typing import Any

from src.config import AssertionInfererConfig
from src.daisy.assertion_inference.base import register_assertion_strategy
from src.daisy.assertion_inference.llm_strategy import LLMAssertionStrategy
from src.llm.llm_configurations import LLM


@register_assertion_strategy("LLM_EXAMPLE")
class LLMExampleAssertionStrategy(LLMAssertionStrategy):
    """Backward-compatible alias that keeps the LLM_EXAMPLE registry entry."""

    def __init__(
        self,
        llm: LLM,
        config: AssertionInfererConfig,
        cache_dir: Path | None = None,
        **kwargs: Any,
    ):
        super().__init__(llm=llm, config=config, cache_dir=cache_dir, **kwargs)
        self.name = "LLM_EXAMPLE"
