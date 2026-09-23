"""Compatibility parser for string model specifications."""

from __future__ import annotations

from ..config import RerankerConfig
from ..errors import ConfigurationError
from .base import ModelBackend
from .jev import JevBackend


def backend_from_spec(spec: str, config: RerankerConfig) -> ModelBackend:
    if spec.startswith("typesafe:"):
        model_id = spec.partition(":")[2]
        if not model_id:
            raise ConfigurationError("typesafe model identifier cannot be empty")
        return JevBackend(
            model=model_id,
            timeout_s=config.timeout_s,
            retry=config.retry,
        )
    raise ConfigurationError(f"unknown model namespace: {spec!r}")
