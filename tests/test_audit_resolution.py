from __future__ import annotations

import asyncio
import sys
import threading
import time
from dataclasses import replace
from decimal import Decimal
from types import ModuleType
from typing import Any

import pytest

from typedrank import AutoReranker, RerankContext, Reranker
from typedrank.backends import (
    BackendCapabilities,
    BackendRouter,
    FakeBackend,
    LayaHTTPBackend,
    ModelResponse,
    SentenceTransformerBackend,
)
from typedrank.backends.laya_context import validate_local_context
from typedrank.cache import MemoryCache
from typedrank.config import Budget, CacheConfig, RerankerConfig, RetryConfig
from typedrank.errors import CapabilityError, ContextLimitError, OutputValidationError
from typedrank.types import RequestStatistics, Usage


class LocalFake(FakeBackend):
    @property
    def capabilities(self) -> BackendCapabilities:
        return replace(super().capabilities, execution_location="local")


@pytest.mark.asyncio
async def test_unknown_location_is_rejected_under_network_denial() -> None:
    backend = FakeBackend()
    with pytest.raises(CapabilityError):
        await Reranker(backend).rerank(
            query="q", candidates=["a"], context=RerankContext(network_policy="deny")
        )
    with pytest.raises(CapabilityError):
        await Reranker(BackendRouter(backend)).rerank(
            query="q", candidates=["a"], context=RerankContext(network_policy="deny")
        )
    assert not backend.calls


@pytest.mark.asyncio
async def test_fake_backend_cache_identity_is_per_configuration() -> None:
    cache = MemoryCache()
    config = RerankerConfig(cache=CacheConfig(enabled=True))
    first = FakeBackend(lambda _q, _t: 0.2)
    second = FakeBackend(lambda _q, _t: 0.9)
    await Reranker(first, cache=cache, config=config).rerank(query="q", candidates=["a"])
    response = await Reranker(second, cache=cache, config=config).rerank(
        query="q", candidates=["a"]
    )
    assert response.results[0].score == 0.9
    assert len(second.calls) == 1


@pytest.mark.asyncio
async def test_generic_executor_ignores_legacy_jev_tariff_fields() -> None:
    backend = FakeBackend()
    backend._capabilities = replace(backend.capabilities, execution_location="remote")
    backend.input_price_per_million_usd = Decimal("1000000")
    backend.output_price_per_million_usd = Decimal("1000000")
    response = await Reranker(
        backend, config=RerankerConfig(budget=Budget(max_cost_usd=0.1))
    ).rerank(query="q", candidates=["a"])
    assert response.results[0].score == 0.5
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_auto_router_selects_listwise_only_backend() -> None:
    backend = LocalFake()
    backend._capabilities = replace(backend.capabilities, pointwise=False, listwise=True)
    result = await AutoReranker(BackendRouter(backend)).rerank(query="q", candidates=["a", "b"])
    assert result.execution_plan.strategy == "listwise"
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_auto_uses_pointwise_fallback_when_listwise_primary_cannot_fit() -> None:
    primary = LocalFake()
    primary._capabilities = replace(
        primary.capabilities, pointwise=False, listwise=True, max_context_tokens=1
    )
    fallback = LocalFake(scores=lambda _q, _t: 0.8)
    fallback._capabilities = replace(fallback.capabilities, listwise=False)
    result = await AutoReranker(BackendRouter(primary, fallback)).rerank(
        query="q", candidates=["a long candidate", "another long candidate"]
    )
    assert result.execution_plan.strategy == "pointwise"
    assert not primary.calls
    assert len(fallback.calls) == 2


@pytest.mark.asyncio
async def test_backend_timeout_can_fall_back() -> None:
    primary = LocalFake(delay_s=0.03)
    fallback = LocalFake(scores=lambda _q, _t: 0.8)
    result = await Reranker(
        BackendRouter(primary, fallback), config=RerankerConfig(timeout_s=0.005)
    ).rerank(query="q", candidates=["a"])
    assert result.results[0].score == 0.8
    assert result.statistics.fallbacks == 1


@pytest.mark.asyncio
async def test_context_incompatibility_can_fall_back() -> None:
    primary = LocalFake(fail=ContextLimitError("too long"))
    fallback = LocalFake(scores=lambda _q, _t: 0.8)
    result = await Reranker(BackendRouter(primary, fallback)).rerank(query="q", candidates=["a"])
    assert result.results[0].score == 0.8
    assert result.statistics.fallbacks == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statistics",
    [
        RequestStatistics(attempts=0),
        RequestStatistics(attempts=2),
        RequestStatistics(latency_ms=float("nan")),
        RequestStatistics(usage=Usage(input_tokens=-1)),
        RequestStatistics(usage=Usage(cost_usd=-1)),
    ],
)
async def test_invalid_backend_accounting_is_rejected(statistics: RequestStatistics) -> None:
    class BadAccounting(LocalFake):
        async def score(self, request: Any) -> ModelResponse:
            response = await super().score(request)
            return replace(response, statistics=statistics)

    with pytest.raises(OutputValidationError):
        await Reranker(BadAccounting()).rerank(query="q", candidates=["a"])


@pytest.mark.asyncio
async def test_http_auto_rejects_mixed_resolved_checkpoints() -> None:
    class Response:
        status_code = 200
        text = None

        def __init__(self, ids: list[str], model: str) -> None:
            self.ids = ids
            self.model = model
            self.headers: dict[str, str] = {}

        def json(self) -> dict[str, Any]:
            return {
                "answers": {key: {"type": "noul", "noul": 0.6} for key in self.ids},
                "routing": {"model": self.model, "reason": "test"},
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }

    class Transport:
        def __init__(self) -> None:
            self.calls = 0

        async def post(self, url: str, *, headers: Any, json: Any, timeout_s: float) -> Response:
            del url, headers, timeout_s
            self.calls += 1
            return Response(
                list(json["questions"]), "english" if self.calls == 1 else "multilingual"
            )

        async def aclose(self) -> None:
            pass

    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        max_batch_size=1,
        transport=Transport(),
        retry=RetryConfig(max_attempts=1),
    )
    with pytest.raises(OutputValidationError, match="multiple checkpoints"):
        await Reranker(backend, strategy="pointwise").rerank(query="q", candidates=["a", "b"])


def test_strict_local_context_uses_loaded_checkpoint_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = ModuleType("laya.common")
    common.render_options = lambda _question: ["yes", "no"]  # type: ignore[attr-defined]
    common.serialize_state = lambda state: str(state)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya.common", common)

    class Tokenizer:
        mask_token = "[MASK]"

        def __call__(self, value: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            del add_special_tokens
            return {"input_ids": list(range(len(value.split())))}

    class Agent:
        def __init__(self) -> None:
            self.tok = Tokenizer()
            self.cfg = {"max_len": 40, "head_max_len": 20}

        def _to_internal(self, question: dict[str, Any]) -> dict[str, Any]:
            return {"t": "noul", "ins": question["instructions"]}

    class Router:
        def load(self, checkpoint: str) -> Agent:
            assert checkpoint == "english"
            return Agent()

    router = Router()
    question = {"id": {"instructions": "short rubric"}}
    validate_local_context(router, {"candidate": "short text"}, question, "english", 40)
    with pytest.raises(ContextLimitError, match="candidate state"):
        validate_local_context(router, {"candidate": "word " * 50}, question, "english", 40)
    with pytest.raises(ContextLimitError, match="instructions"):
        validate_local_context(
            router,
            {"candidate": "short"},
            {"id": {"instructions": "word " * 30}},
            "english",
            40,
        )


@pytest.mark.asyncio
async def test_cancelled_embedding_retains_serialization_and_close_drains() -> None:
    class Encoder(SentenceTransformerBackend):
        def __init__(self) -> None:
            super().__init__("fake")
            self.active = 0
            self.maximum = 0
            self.started = threading.Event()
            self.release = threading.Event()

        def _encode(self, texts: tuple[str, ...], task: str) -> list[list[float]]:
            del task
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.started.set()
            self.release.wait(timeout=2)
            time.sleep(0.01)
            self.active -= 1
            return [[1.0] for _ in texts]

    backend = Encoder()
    first = asyncio.create_task(backend.embed(["a"]))
    await asyncio.to_thread(backend.started.wait, 1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(backend.embed(["b"]))
    await asyncio.sleep(0.01)
    assert backend.maximum == 1
    backend.release.set()
    await second
    await backend.aclose()
    assert backend.maximum == 1
    assert backend.active == 0


@pytest.mark.asyncio
async def test_embedding_close_waits_for_inflight_worker() -> None:
    class Encoder(SentenceTransformerBackend):
        def __init__(self) -> None:
            super().__init__("fake")
            self.started = threading.Event()
            self.release = threading.Event()

        def _encode(self, texts: tuple[str, ...], task: str) -> list[list[float]]:
            del task
            self.started.set()
            self.release.wait(timeout=2)
            return [[1.0] for _ in texts]

    backend = Encoder()
    work = asyncio.create_task(backend.embed(["a"]))
    await asyncio.to_thread(backend.started.wait, 1)
    close = asyncio.create_task(backend.aclose())
    await asyncio.sleep(0.01)
    assert not close.done()
    backend.release.set()
    await asyncio.gather(work, close)
    with pytest.raises(CapabilityError, match="closed"):
        await backend.embed(["b"])
