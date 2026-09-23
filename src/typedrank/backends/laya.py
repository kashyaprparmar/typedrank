"""In-process and self-hosted Laya System One backends."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

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
from .jev import JevBackend, JevHttpResponse
from .laya_context import validate_local_context
from .systemone import decode_relevance, relevance_payload


@dataclass(slots=True)
class LayaBackend:
    """Lazy, bounded bridge to Laya's public Router API.

    One Router and one executor belong to this backend instance. The default
    serializes inference, which avoids extra GPU pressure and model copies.
    """

    model: str = "auto"
    device: str | None = None
    preload: bool = False
    max_loaded: int = 2
    lang_guess: str | Callable[[Any], str | None] | None = None
    max_inference_concurrency: int = 1
    max_batch_size: int = 16
    max_context_tokens: int = 512
    cache_revision: str | None = None
    context_policy: Literal["strict", "allow_provider_truncation"] = "strict"
    router: Any | None = field(default=None, repr=False)
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _router_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if any(
            value < 1
            for value in (
                self.max_loaded,
                self.max_inference_concurrency,
                self.max_batch_size,
                self.max_context_tokens,
            )
        ):
            raise ValueError("Laya concurrency, batch, and context limits must be positive")
        if not self.model:
            raise ValueError("Laya model cannot be empty")
        if self.max_inference_concurrency != 1:
            raise ValueError("Laya 0.3.7 supports one inference worker per Router")
        if self.context_policy not in {"strict", "allow_provider_truncation"}:
            raise ValueError("unknown Laya context policy")
        if self.cache_revision is not None and not self.cache_revision.strip():
            raise ValueError("cache_revision cannot be empty")

    @property
    def backend_id(self) -> str:
        return "laya-local"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            pointwise=True,
            listwise=True,
            max_batch_size=self.max_batch_size,
            max_context_tokens=self.max_context_tokens,
            usage_reporting=True,
            execution_location="local",
            multilingual=True,
            batched_decisions=True,
        )

    @property
    def cache_identity(self) -> str:
        return (
            f"laya-local:{self.model}:{self.device}:{self.max_loaded}:"
            f"{self.lang_guess}:{self.max_context_tokens}:{self.cache_revision}:noul-v2"
        )

    @property
    def cache_stable(self) -> bool:
        # An alias alone does not pin mutable Hugging Face weights.
        return self.cache_revision is not None and not callable(self.lang_guess)

    def estimate_request_tokens(self, request: ModelRequest) -> int:
        payload = relevance_payload(request, model=request.checkpoint or self.model, profile="laya")
        return max(1, math.ceil(len(json.dumps(payload, ensure_ascii=False)) / 3))

    def estimate_context_tokens(self, request: ModelRequest) -> int:
        payload = relevance_payload(request, model=request.checkpoint or self.model, profile="laya")
        return max(
            (
                math.ceil(
                    len(
                        json.dumps(
                            {"state": payload["state"], "question": question}, ensure_ascii=False
                        )
                    )
                    / 3
                )
                for question in payload["questions"].values()
            ),
            default=0,
        )

    def _get_router(self) -> Any:
        with self._router_lock:
            if self.router is None:
                try:
                    from laya import Router
                except ImportError as exc:
                    raise CapabilityError(
                        'Laya is not installed; run pip install "typedrank[laya]"'
                    ) from exc
                self.router = Router(
                    device=self.device,
                    max_loaded=self.max_loaded,
                    preload=self.preload,
                    lang_guess=self.lang_guess,
                )
        return self.router

    async def _worker(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise BackendError("Laya backend is closed")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_inference_concurrency,
                thread_name_prefix="typedrank-laya",
            )
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_inference_concurrency)
        async with self._semaphore:
            if self._closed:
                raise BackendError("Laya backend is closed")
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                # Torch cannot be interrupted safely. Keep the concurrency slot
                # occupied until the worker actually finishes.
                try:
                    await asyncio.shield(future)
                except Exception:
                    pass
                raise

    def _route(
        self, query: str, language: str | None, candidate_texts: Sequence[str]
    ) -> Mapping[str, Any]:
        router = self._get_router()
        state = {"query": query, "candidates": list(candidate_texts)}
        if self.model != "auto":
            decision = router.route(state, {}, model=self.model)
        else:
            decision = router.route(state, {}, lang=language, lang_guess=self.lang_guess)
            if language is None and candidate_texts and decision.get("model") == "english":
                for candidate_text in candidate_texts:
                    candidate_route = router.route(candidate_text, {}, lang_guess=self.lang_guess)
                    if candidate_route.get("model") == "multilingual":
                        decision = dict(router.route(state, {}, model="multilingual"))
                        decision["reason"] = "multilingual survivor in candidate pool"
                        break
        if not isinstance(decision, Mapping) or not isinstance(decision.get("model"), str):
            raise OutputValidationError("Laya Router returned an invalid route")
        return decision

    async def prepare_stage(
        self, query: str, language: str | None = None, candidate_texts: Sequence[str] = ()
    ) -> Mapping[str, Any]:
        """Pin one checkpoint before pointwise calls to avoid mixed-model scores."""
        try:
            return cast(
                Mapping[str, Any], await self._worker(self._route, query, language, candidate_texts)
            )
        except (CapabilityError, OutputValidationError):
            raise
        except Exception as exc:
            raise BackendError("Laya route selection failed") from exc

    def _predict(self, state: Any, questions: Any, checkpoint: str | None) -> Mapping[str, Any]:
        router = self._get_router()
        if self.context_policy == "strict":
            if checkpoint is None:
                raise CapabilityError("strict Laya scoring requires a pinned checkpoint")
            validate_local_context(router, state, questions, checkpoint, self.max_context_tokens)
        result = router.predict(state, questions, model=checkpoint)
        if not isinstance(result, Mapping):
            raise OutputValidationError("Laya returned a non-object response")
        return cast(Mapping[str, Any], result)

    async def score(self, request: ModelRequest) -> ModelResponse:
        if self._closed:
            raise BackendError("Laya backend is closed")
        if not request.candidates:
            return ModelResponse(scores=(), resolved_model=request.checkpoint or self.model)
        if len(request.candidates) > self.max_batch_size:
            raise CapabilityError("Laya request exceeds the configured decision batch limit")
        if self.estimate_context_tokens(request) > self.max_context_tokens:
            raise ContextLimitError("Laya request may exceed the checkpoint context limit")
        payload = relevance_payload(request, model=request.checkpoint, profile="laya")
        started = time.perf_counter()
        checkpoint = request.checkpoint or (None if self.model == "auto" else self.model)
        try:
            data = await self._worker(
                self._predict, payload["state"], payload["questions"], checkpoint
            )
        except (CapabilityError, BackendError, OutputValidationError, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise BackendError(
                "Laya inference failed",
                details=ErrorDetails(stage="laya-local", retryable=False),
            ) from exc
        response = decode_relevance(
            data,
            expected=tuple(candidate.candidate_id for candidate in request.candidates),
            allow_partial=request.allow_partial,
            attempts=1,
            latency_ms=(time.perf_counter() - started) * 1000,
            requested_model=checkpoint,
        )
        if self.context_policy == "allow_provider_truncation":
            return replace(
                response, metadata={**response.metadata, "context_validation": "unverified"}
            )
        return response

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        executor = self._executor
        try:
            if executor is not None:
                await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=False)
                self._executor = None
        finally:
            if self.router is not None:
                await asyncio.to_thread(self.router.unload)
                self.router = None


@dataclass(slots=True)
class LayaHTTPBackend(JevBackend):
    """Self-hosted Laya's Jev-compatible POST /v1/systemone endpoint."""

    model: str = "auto"
    endpoint: str = "http://localhost:8000/v1/systemone"
    max_batch_size: int = 16
    max_context_tokens: int = 512
    cache_revision: str | None = None
    context_policy: Literal["strict", "allow_provider_truncation"] = "strict"
    context_validator: Callable[[Mapping[str, Any]], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.reasoning_level is not None:
            raise CapabilityError("Laya HTTP does not expose a reasoning-level control")
        JevBackend.__post_init__(self)
        if self.max_batch_size < 1 or self.max_context_tokens < 1:
            raise ValueError("Laya batch and context limits must be positive")
        if self.cache_revision is not None and not self.cache_revision.strip():
            raise ValueError("cache_revision cannot be empty")
        if self.context_policy not in {"strict", "allow_provider_truncation"}:
            raise ValueError("unknown Laya HTTP context policy")
        if self.model not in {"auto", "english", "multilingual", "typed-decisions"}:
            raise ValueError("unknown Laya HTTP checkpoint")

    @property
    def backend_id(self) -> str:
        return "laya-http"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            pointwise=True,
            listwise=True,
            max_batch_size=self.max_batch_size,
            max_context_tokens=self.max_context_tokens,
            usage_reporting=True,
            execution_location="remote",
            multilingual=True,
            batched_decisions=True,
        )

    @property
    def cache_identity(self) -> str:
        return (
            f"laya-http:{self.endpoint}:{self.model}:{self.max_context_tokens}:"
            f"{self.cache_revision}:noul-v2"
        )

    @property
    def cache_stable(self) -> bool:
        return self.cache_revision is not None and self.model != "auto"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        return relevance_payload(
            request,
            model=request.checkpoint or (None if self.model == "auto" else self.model),
            profile="laya",
        )

    def estimate_context_tokens(self, request: ModelRequest) -> int:
        payload = self._payload(request)
        return max(
            (
                math.ceil(
                    len(
                        json.dumps(
                            {"state": payload["state"], "question": question}, ensure_ascii=False
                        )
                    )
                    / 3
                )
                for question in payload["questions"].values()
            ),
            default=0,
        )

    async def score(self, request: ModelRequest) -> ModelResponse:
        if not request.candidates:
            return await JevBackend.score(self, request)
        if len(request.candidates) > self.max_batch_size:
            raise CapabilityError("Laya HTTP decision batch exceeds its configured limit")
        if self.estimate_context_tokens(request) > self.max_context_tokens:
            raise ContextLimitError("Laya HTTP request may exceed the checkpoint context limit")
        if self.context_policy == "strict":
            if self.context_validator is None:
                raise CapabilityError(
                    "Laya HTTP strict context mode requires a deployment validator"
                )
            self.context_validator(self._payload(request))
        response = await JevBackend.score(self, request)
        requested = request.checkpoint or (None if self.model == "auto" else self.model)
        routing = response.metadata.get("routing")
        if requested is None and (
            not isinstance(routing, Mapping)
            or routing.get("model") not in {"english", "multilingual", "typed-decisions"}
        ):
            raise OutputValidationError("Laya HTTP auto mode did not identify its checkpoint")
        if requested is not None and (
            not isinstance(routing, Mapping) or routing.get("model") != requested
        ):
            raise OutputValidationError("Laya HTTP did not confirm the requested checkpoint")
        if self.context_policy == "allow_provider_truncation":
            return replace(
                response, metadata={**response.metadata, "context_validation": "unverified"}
            )
        return response

    def _raise_for_status(self, response: JevHttpResponse) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        details = ErrorDetails(stage="laya-http", status_code=status)
        if status in {401, 403}:
            raise AuthenticationError("Laya HTTP authentication failed", details=details)
        if status == 429:
            raise RateLimitError(
                "Laya HTTP rate limit exceeded",
                details=ErrorDetails(stage="laya-http", status_code=status, retryable=True),
            )
        if status == 413:
            raise ContextLimitError("Laya HTTP request is too large", details=details)
        if status >= 500:
            raise BackendError(
                "Laya HTTP service is unavailable",
                details=ErrorDetails(stage="laya-http", status_code=status, retryable=True),
            )
        raise BackendError("Laya HTTP rejected the request", details=details)

    def _timeout_message(self) -> str:
        return "Laya HTTP request timed out"

    def _error_stage(self) -> str:
        return "laya-http"
