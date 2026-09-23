from __future__ import annotations

from dataclasses import replace

import pytest

from typedrank import AutoReranker, RerankContext, Reranker
from typedrank.backends import BackendCapabilities, BackendRouter, FakeBackend
from typedrank.cache import MemoryCache
from typedrank.config import CacheConfig, RerankerConfig
from typedrank.errors import BackendError, CapabilityError
from typedrank.pipeline import ModelReranker, RerankPipeline
from typedrank.types import ResultStatus


class NamedBackend(FakeBackend):
    def __init__(
        self,
        name: str,
        location: str,
        score: float = 0.5,
        fail: Exception | None = None,
        multilingual: bool = False,
    ) -> None:
        super().__init__(scores=lambda query, text: score, fail=fail)
        self.name = name
        self._capabilities = replace(
            self._capabilities, execution_location=location, multilingual=multilingual
        )

    @property
    def backend_id(self) -> str:
        return self.name

    @property
    def cache_identity(self) -> str:
        return f"{self.name}:test-v1"


class FlakyBackend(NamedBackend):
    def __init__(self) -> None:
        super().__init__("local", "local", score=0.8)
        self.fail_next = True

    async def score(self, request):
        if self.fail_next:
            self.fail_next = False
            raise BackendError("first stage unavailable")
        return await super().score(request)


@pytest.mark.asyncio
async def test_whole_stage_falls_back_without_mixing_scores() -> None:
    local = NamedBackend("local", "local", fail=BackendError("unavailable"))
    remote = NamedBackend("remote", "remote", score=0.9)
    router = BackendRouter(primary=local, fallback=remote)
    result = await Reranker(backend=router).rerank(query="query", candidates=["a", "b", "c"])
    assert all(item.score == 0.9 for item in result.results)
    assert len(local.calls) == len(remote.calls) == 3
    assert result.statistics.fallbacks == 1
    assert result.statistics.selected_backend == "remote"
    assert result.statistics.routing_reason == "fallback after local"
    assert result.execution_plan.stages[0].backend == "remote"
    assert result.status is ResultStatus.FALLBACK


@pytest.mark.asyncio
async def test_network_policy_filters_remote() -> None:
    remote = NamedBackend("remote", "remote")
    local = NamedBackend("local", "local", score=0.7)
    router = BackendRouter(primary=remote, fallback=local)
    result = await Reranker(backend=router).rerank(
        query="query", candidates=["a"], context=RerankContext(network_policy="deny")
    )
    assert result.results[0].score == 0.7
    assert not remote.calls
    with pytest.raises(CapabilityError):
        await Reranker(backend=remote).rerank(
            query="query", candidates=["a"], context=RerankContext(network_policy="deny")
        )


@pytest.mark.asyncio
async def test_auto_offline_can_use_local_backend() -> None:
    remote = NamedBackend("remote", "remote", score=0.1)
    local = NamedBackend("local", "local", score=0.7)
    router = BackendRouter(primary=remote, fallback=local)
    result = await AutoReranker(backend=router).rerank(
        query="query",
        candidates=["a", "b"],
        context=RerankContext(quality_mode="offline"),
    )
    assert result.statistics.selected_backend == "local"
    assert result.results[0].score == 0.7
    assert not remote.calls


def test_policy_order_uses_only_declared_capabilities_and_measurements() -> None:
    remote = NamedBackend("remote", "remote", multilingual=False)
    local = NamedBackend("local", "local", multilingual=True)
    context = RerankContext(language="hi")
    assert (
        BackendRouter(remote, local, "local_first").routes("pointwise", context)[0].backend is local
    )
    assert (
        BackendRouter(local, remote, "remote_first").routes("pointwise", context)[0].backend
        is remote
    )
    assert (
        BackendRouter(remote, local, "language_aware").routes("pointwise", context)[0].backend
        is local
    )
    assert (
        BackendRouter(
            remote,
            local,
            "latency_aware",
            measured_latency_ms={"remote": 8.0, "local": 12.0},
        )
        .routes("pointwise", context)[0]
        .backend
        is remote
    )
    assert (
        BackendRouter(
            remote,
            local,
            "cost_aware",
            measured_cost_usd={"remote": 1.0, "local": 0.5},
        )
        .routes("pointwise", context)[0]
        .backend
        is local
    )
    assert (
        BackendRouter(remote, local, "cost_aware").routes("pointwise", context)[0].backend is remote
    )


@pytest.mark.asyncio
async def test_cache_keeps_backend_identities_separate() -> None:
    cache = MemoryCache()
    config = RerankerConfig(cache=CacheConfig(enabled=True))
    local = NamedBackend("local", "local", score=0.2)
    remote = NamedBackend("remote", "remote", score=0.8)
    one = await Reranker(backend=local, cache=cache, config=config).rerank(
        query="query", candidates=["candidate"]
    )
    two = await Reranker(backend=remote, cache=cache, config=config).rerank(
        query="query", candidates=["candidate"]
    )
    assert one.results[0].score == 0.2
    assert two.results[0].score == 0.8
    assert two.statistics.cache_hits == 0


def test_router_rejects_unsupported_stage() -> None:
    backend = NamedBackend("local", "local")
    backend._capabilities = BackendCapabilities(pointwise=False, listwise=False)
    with pytest.raises(CapabilityError):
        BackendRouter(primary=backend).routes("pointwise", RerankContext())


@pytest.mark.asyncio
async def test_router_closes_shared_child_once() -> None:
    class CloseTrackingBackend(NamedBackend):
        closes = 0

        async def aclose(self) -> None:
            self.closes += 1

    child = CloseTrackingBackend("local", "local")
    await BackendRouter(primary=child, fallback=child).aclose()
    assert child.closes == 1


def test_router_rejects_invalid_measurement() -> None:
    child = NamedBackend("local", "local")
    with pytest.raises(ValueError):
        BackendRouter(primary=child, policy="latency_aware", measured_latency_ms={"local": -1})


@pytest.mark.asyncio
async def test_auto_cascade_uses_configured_shortlists() -> None:
    class Embeddings:
        backend_id = "fake-embeddings"
        cache_identity = "fake-embeddings:v1"

        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

        async def aclose(self) -> None:
            return None

    result = await AutoReranker(
        backend=NamedBackend("local", "local", score=0.7),
        embedding_backend=Embeddings(),
    ).rerank(
        query="candidate",
        candidates=[f"candidate {index}" for index in range(500)],
        top_k=10,
    )
    assert [stage.name for stage in result.execution_plan.stages] == [
        "BM25Filter",
        "EmbeddingReranker",
        "ModelReranker",
    ]
    assert [stage.output_count for stage in result.execution_plan.stages] == [100, 30, 10]


@pytest.mark.asyncio
async def test_pipeline_records_each_model_stage_route() -> None:
    local = FlakyBackend()
    remote = NamedBackend("remote", "remote", score=0.6)
    pipeline = RerankPipeline(
        [ModelReranker(limit=2, mode="listwise"), ModelReranker(limit=2, mode="listwise")]
    )
    result = await Reranker(
        backend=BackendRouter(primary=local, fallback=remote),
        strategy=pipeline,
    ).rerank(query="query", candidates=["a", "b"], top_k=2)
    assert [stage.backend for stage in result.execution_plan.stages] == ["remote", "local"]
    assert result.statistics.selected_backend == "local"
    assert result.statistics.fallbacks == 1
