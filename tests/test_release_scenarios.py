"""Backend parity and configured cross-provider routes without live model calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from typedrank import Reranker
from typedrank.backends import BackendRouter, FakeBackend, JevBackend, LayaBackend, LayaHTTPBackend
from typedrank.config import RetryConfig
from typedrank.metrics import EmbeddingSimilarity
from typedrank.pipeline import EmbeddingReranker, ModelReranker, RerankPipeline
from typedrank.types import ResultStatus

SCORES = {"c00000000": 0.2, "c00000001": 0.9, "c00000002": 0.5}


@dataclass(frozen=True)
class Record:
    title: str


class HTTPResponse:
    def __init__(self, data: dict[str, Any], status_code: int = 200) -> None:
        self.data = data
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return self.data


class Transport:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def post(self, url: str, *, headers: Any, json: Any, timeout_s: float) -> HTTPResponse:
        del url, headers, timeout_s
        self.calls += 1
        if self.fail:
            return HTTPResponse({}, 503)
        return HTTPResponse(
            {
                "answers": {
                    key: {"type": "noul", "noul": SCORES[key]} for key in json["questions"]
                },
                "usage": {"input_tokens": 3, "output_tokens": 1},
                "model": "english",
                "routing": {"model": "english", "reason": "test"},
            }
        )

    async def aclose(self) -> None:
        return None


class LocalRouter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def route(self, state: Any, questions: Any, **kwargs: Any) -> dict[str, str]:
        del state, questions, kwargs
        return {"model": "english"}

    def predict(self, state: Any, questions: Any, model: str | None = None) -> dict[str, Any]:
        del state, model
        self.calls += 1
        if self.fail:
            raise RuntimeError("mock local inference failure")
        return {
            "answers": {key: {"type": "noul", "noul": SCORES[key]} for key in questions},
            "usage": {"input_tokens": 3, "output_tokens": 1},
            "model": "english",
        }

    def unload(self) -> None:
        return None


def backend(kind: str, *, fail: bool = False) -> Any:
    if kind == "fake":
        return FakeBackend(SCORES)
    if kind == "jev":
        return JevBackend(
            api_key="mock-key", transport=Transport(fail=fail), retry=RetryConfig(max_attempts=1)
        )
    if kind == "laya-local":
        return LayaBackend(
            context_policy="allow_provider_truncation", router=LocalRouter(fail=fail)
        )
    if kind == "laya-http":
        return LayaHTTPBackend(
            context_policy="allow_provider_truncation",
            transport=Transport(fail=fail),
            retry=RetryConfig(max_attempts=1),
        )
    raise AssertionError(kind)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["fake", "jev", "laya-local", "laya-http"])
@pytest.mark.parametrize("strategy", ["pointwise", "listwise"])
async def test_equivalent_typed_scores_preserve_ranking_and_objects(
    kind: str, strategy: str
) -> None:
    records = [Record("first"), Record("second"), Record("third")]
    adapter = backend(kind)
    try:
        result = await Reranker(backend=adapter, strategy=strategy).rerank(
            query="choose one", candidates=records, text_fn=lambda record: record.title, top_k=3
        )
        assert [item.item for item in result.results] == [records[1], records[2], records[0]]
        assert all(item.item is records[item.input_index] for item in result.results)
        assert [item.score for item in result.results] == [0.9, 0.5, 0.2]
        expected_calls = 3 if strategy == "pointwise" and kind in {"fake", "jev"} else 1
        assert result.statistics.model_calls == expected_calls
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary", "fallback"),
    [
        ("laya-local", None),
        ("laya-local", "jev"),
        ("jev", "laya-local"),
        ("laya-http", "laya-local"),
    ],
)
async def test_configured_backend_routes(primary: str, fallback: str | None) -> None:
    first = backend(primary, fail=fallback is not None)
    second = backend(fallback) if fallback is not None else None
    router = BackendRouter(primary=first, fallback=second)
    try:
        result = await Reranker(backend=router).rerank(
            query="choose one", candidates=["first", "second", "third"], top_k=2
        )
        assert [item.item for item in result.results] == ["second", "third"]
        selected = fallback or primary
        assert result.statistics.selected_backend == ("typesafe" if selected == "jev" else selected)
        assert result.statistics.fallbacks == (1 if fallback else 0)
        assert result.status is (ResultStatus.FALLBACK if fallback else ResultStatus.OK)
        first_calls = first.router.calls if primary == "laya-local" else first.transport.calls
        assert first_calls == (3 if primary == "jev" else 1)
    finally:
        await router.aclose()


class Embeddings:
    backend_id = "mock-embeddings"
    cache_identity = "mock-embeddings:v1"

    async def embed(self, texts: Any) -> list[list[float]]:
        return [
            [1.0, 0.0] if text in {"choose one", "first", "second"} else [0.0, 1.0]
            for text in texts
        ]

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["laya-local", "jev"])
async def test_embedding_prefilter_then_model_backend(kind: str) -> None:
    adapter = backend(kind)
    embeddings = Embeddings()
    ranker = Reranker(
        backend=adapter,
        embedding_backend=embeddings,
        strategy=RerankPipeline(
            [EmbeddingReranker(EmbeddingSimilarity(embeddings), limit=2), ModelReranker(limit=1)]
        ),
    )
    try:
        result = await ranker.rerank(
            query="choose one", candidates=["first", "second", "third"], top_k=1
        )
        assert result.results[0].item == "second"
        assert [stage.name for stage in result.execution_plan.stages] == [
            "EmbeddingReranker",
            "ModelReranker",
        ]
        assert result.execution_plan.stages[0].output_count == 2
        assert result.statistics.selected_backend == (
            "laya-local" if kind == "laya-local" else "typesafe"
        )
    finally:
        await ranker.aclose()
        await adapter.aclose()
