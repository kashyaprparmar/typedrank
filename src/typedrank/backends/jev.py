"""Async TypeSafe AI System One / Jev backend."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from typing import Any

from ..config import RetryConfig
from ..errors import (
    AuthenticationError,
    BackendError,
    CapabilityError,
    ContextLimitError,
    ErrorDetails,
    OutputValidationError,
    RateLimitError,
)
from .base import BackendCapabilities, ModelRequest, ModelResponse
from .systemone import (
    HTTPResponse,
    HTTPTransport,
    HttpxTransport,
    ResponsePricing,
    _token_value,
    _unique_object,
    decode_http_response,
    decode_relevance,
    relevance_payload,
)

JevHttpResponse = HTTPResponse
JevTransport = HTTPTransport


@dataclass(slots=True)
class JevBackend:
    """Jev Noul relevance adapter with exact ID and score validation."""

    api_key: str | None = field(default=None, repr=False)
    model: str = "jev-1.13.0"
    endpoint: str = "https://api.typesafe.ai/v1/systemone"
    timeout_s: float = 30.0
    retry: RetryConfig = field(default_factory=RetryConfig)
    reasoning_level: str | None = None
    input_price_per_million_usd: Decimal | None = None
    output_price_per_million_usd: Decimal | None = None
    transport: JevTransport | None = field(default=None, repr=False)
    _owns_transport: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or not math.isfinite(self.timeout_s):
            raise ValueError("timeout_s must be finite and positive")
        if self.reasoning_level is not None:
            raise CapabilityError("Jev does not expose a reasoning-level control")
        for price in (self.input_price_per_million_usd, self.output_price_per_million_usd):
            if price is not None and (not price.is_finite() or price < 0):
                raise ValueError("token prices must be finite and non-negative")
        self._owns_transport = self.transport is None

    @property
    def backend_id(self) -> str:
        return "typesafe"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            pointwise=True,
            listwise=True,
            explanations=False,
            reasoning_levels=(),
            max_context_tokens=None,
            usage_reporting=True,
            execution_location="remote",
        )

    @property
    def cache_identity(self) -> str:
        return f"typesafe:{self.endpoint}:{self.model}:noul-v1"

    @property
    def cache_stable(self) -> bool:
        return re.fullmatch(r"jev-\d+\.\d+\.\d+", self.model) is not None

    def _get_api_key(self) -> str:
        key = self.api_key or os.getenv("TYPESAFE_API_KEY")
        if not key:
            raise AuthenticationError("TypeSafe API key is not configured")
        return key

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        return relevance_payload(request, model=self.model, profile="jev")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    def estimated_cost(self, tokens: int) -> Decimal | None:
        if self.input_price_per_million_usd is None or self.output_price_per_million_usd is None:
            return None
        price = max(self.input_price_per_million_usd, self.output_price_per_million_usd)
        return price * Decimal(tokens) / Decimal(1_000_000)

    def estimate_request_tokens(self, request: ModelRequest) -> int:
        """Conservative character-based allowance over the actual serialized payload."""
        payload = self._payload(request)
        rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return max(1, math.ceil(len(rendered) / 3) + 16 * len(request.candidates))

    async def score(self, request: ModelRequest) -> ModelResponse:
        if not request.candidates:
            return ModelResponse(scores=(), resolved_model=self.model)
        expected = tuple(candidate.candidate_id for candidate in request.candidates)
        if len(set(expected)) != len(expected):
            raise OutputValidationError("backend request contains duplicate candidate IDs")
        payload = self._payload(request)
        headers = self._headers()
        transport = self.transport
        if transport is None:
            transport = HttpxTransport()
            self.transport = transport
        started = time.perf_counter()
        attempts = 0
        response: JevHttpResponse | None = None
        while attempts < self.retry.max_attempts:
            attempts += 1
            try:
                response = await transport.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout_s=self.timeout_s,
                )
                self._raise_for_status(response)
                break
            except asyncio.CancelledError:
                raise
            except (TimeoutError, RateLimitError, BackendError) as exc:
                retryable = isinstance(exc, TimeoutError) or exc.details.retryable
                if not retryable or attempts >= self.retry.max_attempts:
                    if isinstance(exc, TimeoutError):
                        raise BackendError(
                            "TypeSafe request timed out",
                            details=ErrorDetails(
                                stage="jev", retryable=True, safe_context={"attempts": attempts}
                            ),
                        ) from exc
                    exc.details = replace(
                        exc.details,
                        safe_context={**(exc.details.safe_context or {}), "attempts": attempts},
                    )
                    raise
                delay = min(
                    self.retry.max_backoff_s,
                    self.retry.initial_backoff_s * (2 ** (attempts - 1)),
                )
                if isinstance(exc, RateLimitError) and response is not None:
                    retry_after = self._retry_after_seconds(response.headers)
                    if retry_after is not None:
                        if retry_after > self.retry.max_backoff_s:
                            exc.details = replace(exc.details, safe_context={"attempts": attempts})
                            raise
                        delay = max(delay, retry_after)
                jitter = delay * self.retry.jitter_ratio * random.random()
                await asyncio.sleep(delay + jitter)
        if response is None:  # pragma: no cover - defensive
            raise BackendError("TypeSafe request produced no response")
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            return self._parse_response(
                response,
                expected=expected,
                allow_partial=request.allow_partial,
                attempts=attempts,
                latency_ms=latency_ms,
            )
        except OutputValidationError as exc:
            exc.details = replace(exc.details, safe_context={"attempts": attempts})
            raise

    @staticmethod
    def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
        raw = next((value for key, value in headers.items() if key.lower() == "retry-after"), None)
        if raw is None:
            return None
        try:
            seconds = float(raw)
        except ValueError:
            try:
                seconds = (
                    parsedate_to_datetime(raw).astimezone(UTC) - datetime.now(UTC)
                ).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return None
        return max(0.0, seconds) if math.isfinite(seconds) else None

    def _raise_for_status(self, response: JevHttpResponse) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        details = ErrorDetails(stage="jev", status_code=status)
        if status in {401, 403}:
            raise AuthenticationError("TypeSafe authentication failed", details=details)
        if status == 429:
            raise RateLimitError(
                "TypeSafe rate limit exceeded",
                details=ErrorDetails(stage="jev", status_code=status, retryable=True),
            )
        if status in {413, 422}:
            raise ContextLimitError("TypeSafe request exceeds provider limits", details=details)
        if status == 529 or status >= 500:
            raise BackendError(
                "TypeSafe service is unavailable",
                details=ErrorDetails(stage="jev", status_code=status, retryable=True),
            )
        raise BackendError("TypeSafe rejected the request", details=details)

    def _parse_response(
        self,
        response: JevHttpResponse,
        *,
        expected: tuple[str, ...],
        allow_partial: bool,
        attempts: int,
        latency_ms: float,
    ) -> ModelResponse:
        return decode_relevance(
            decode_http_response(response),
            expected=expected,
            allow_partial=allow_partial,
            attempts=attempts,
            latency_ms=latency_ms,
            requested_model=self.model,
            pricing=ResponsePricing(
                self.input_price_per_million_usd, self.output_price_per_million_usd
            ),
        )

    _unique_object = staticmethod(_unique_object)
    _token_value = staticmethod(_token_value)

    async def aclose(self) -> None:
        if self._owns_transport and self.transport is not None:
            await self.transport.aclose()
            self.transport = None
