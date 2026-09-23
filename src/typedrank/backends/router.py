"""Explicit, inspectable selection of a model backend for one ranking stage."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

from ..context import RerankContext
from ..errors import CapabilityError
from .base import BackendCapabilities, ModelBackend, ModelRequest, ModelResponse

RoutingPolicy = Literal[
    "local_first",
    "remote_first",
    "availability",
    "language_aware",
    "cost_aware",
    "latency_aware",
]


@dataclass(frozen=True, slots=True)
class BackendRoute:
    backend: ModelBackend
    reason: str


@dataclass(slots=True)
class BackendRouter:
    """Choose WHERE a model stage runs; the strategy still chooses HOW to rank.

    Cost and latency routing use caller-supplied measurements. With no comparable
    measurements, the declared primary wins; no speed or price is fabricated.
    """

    primary: ModelBackend
    fallback: ModelBackend | None = None
    policy: RoutingPolicy = "availability"
    measured_latency_ms: Mapping[str, float] = field(default_factory=dict)
    measured_cost_usd: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy not in {
            "local_first",
            "remote_first",
            "availability",
            "language_aware",
            "cost_aware",
            "latency_aware",
        }:
            raise ValueError("unknown backend routing policy")
        for measurements in (self.measured_latency_ms, self.measured_cost_usd):
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                for value in measurements.values()
            ):
                raise ValueError("routing measurements must be finite and non-negative")

    @property
    def backend_id(self) -> str:
        return "router"

    @property
    def model(self) -> str:
        return self.primary.model

    @property
    def capabilities(self) -> BackendCapabilities:
        return self.primary.capabilities

    @property
    def cache_identity(self) -> str:
        return f"router:{self.primary.cache_identity}"

    @property
    def cache_stable(self) -> bool:
        return False

    def estimate_request_tokens(self, request: ModelRequest) -> int:
        estimator = getattr(self.primary, "estimate_request_tokens", None)
        if callable(estimator):
            return cast(int, estimator(request))
        return max(1, (len(request.query) + sum(len(c.text) for c in request.candidates)) // 4)

    def estimate_context_tokens(self, request: ModelRequest) -> int:
        estimator = getattr(self.primary, "estimate_context_tokens", None)
        return (
            cast(int, estimator(request))
            if callable(estimator)
            else self.estimate_request_tokens(request)
        )

    def _eligible(self, backend: ModelBackend, mode: str, context: RerankContext) -> bool:
        caps = backend.capabilities
        if mode == "pointwise" and not caps.pointwise:
            return False
        if mode == "listwise" and not caps.listwise:
            return False
        if (
            context.network_policy == "deny" or context.quality_mode == "offline"
        ) and caps.execution_location != "local":
            return False
        available = getattr(backend, "is_available", None)
        if callable(available) and not available():
            return False
        return True

    def routes(self, mode: str, context: RerankContext) -> tuple[BackendRoute, ...]:
        choices = [
            backend
            for backend in (self.primary, self.fallback)
            if backend is not None and self._eligible(backend, mode, context)
        ]
        if not choices:
            raise CapabilityError("no backend satisfies this stage and network policy")
        reason = f"{self.policy}: configured primary"
        if self.policy == "local_first":
            choices.sort(key=lambda backend: backend.capabilities.execution_location != "local")
            reason = "local_first: local execution preferred"
        elif self.policy == "remote_first":
            choices.sort(key=lambda backend: backend.capabilities.execution_location != "remote")
            reason = "remote_first: remote execution preferred"
        elif self.policy == "language_aware":
            if context.language and context.language.lower() not in {"en", "eng", "english"}:
                choices.sort(key=lambda backend: not backend.capabilities.multilingual)
                reason = f"language_aware: {context.language} requires multilingual support"
            else:
                reason = "language_aware: no non-English language hint; configured primary"
        elif self.policy in {"cost_aware", "latency_aware"}:
            measurements = (
                self.measured_cost_usd if self.policy == "cost_aware" else self.measured_latency_ms
            )
            backend_ids = tuple(item.backend_id for item in choices)

            def measured(backend: ModelBackend) -> float | None:
                if backend.cache_identity in measurements:
                    return measurements[backend.cache_identity]
                if backend_ids.count(backend.backend_id) == 1:
                    return measurements.get(backend.backend_id)
                return None

            scores = {id(backend): measured(backend) for backend in choices}
            if all(value is not None for value in scores.values()):
                choices.sort(key=lambda backend: cast(float, scores[id(backend)]))
                reason = f"{self.policy}: lowest configured measurement"
            else:
                reason = f"{self.policy}: measurements incomplete; configured primary"
        else:
            reason = "availability: first available backend"
        return tuple(BackendRoute(backend, reason) for backend in choices)

    async def score(self, request: ModelRequest) -> ModelResponse:
        raise CapabilityError("BackendRouter must be used through Reranker for stage-level routing")

    async def aclose(self) -> None:
        closed: set[int] = set()
        for backend in (self.primary, self.fallback):
            if backend is not None and id(backend) not in closed:
                closed.add(id(backend))
                await backend.aclose()
