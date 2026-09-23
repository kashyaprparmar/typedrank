from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import Any

import pytest

from typedrank import RerankContext, Reranker
from typedrank.backends import BackendCandidate, LayaBackend, ModelRequest
from typedrank.config import Budget, CacheConfig, RerankerConfig
from typedrank.errors import BackendError, CapabilityError, ContextLimitError, OutputValidationError
from typedrank.prompts import DEFAULT_PROMPT


class RouterDouble:
    def __init__(self, *, delay: float = 0.0, malformed: bool = False) -> None:
        self.delay = delay
        self.malformed = malformed
        self.models: list[str | None] = []
        self.route_states: list[Any] = []
        self.active = 0
        self.maximum_active = 0
        self._lock = threading.Lock()
        self.unloaded = False

    def route(
        self,
        state: Any,
        questions: Any,
        model: str | None = None,
        lang: str | None = None,
        lang_guess: str | None = None,
    ) -> dict[str, str]:
        del questions, lang_guess
        self.route_states.append(state)
        checkpoint = model or (
            "multilingual"
            if lang == "hi" or (isinstance(state, str) and "हिंदी" in state)
            else "english"
        )
        return {"model": checkpoint, "reason": f"language={lang}"}

    def predict(
        self, state: Any, questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        del state
        with self._lock:
            self.models.append(model)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            answers = {key: {"type": "noul", "noul": 0.75} for key in questions}
            if self.malformed:
                answers.pop(next(iter(answers)))
            return {
                "model": "laya-rl-agent",
                "answers": answers,
                "usage": {"input_tokens": 4, "output_tokens": 0},
                "routing": {"model": model, "reason": "explicit"},
            }
        finally:
            with self._lock:
                self.active -= 1

    def unload(self) -> None:
        self.unloaded = True


def _request() -> ModelRequest:
    return ModelRequest("hello", (BackendCandidate("one", "text"),), DEFAULT_PROMPT)


@pytest.mark.asyncio
async def test_local_stage_pins_multilingual_checkpoint_and_provenance() -> None:
    fake = RouterDouble()
    backend = LayaBackend(context_policy="allow_provider_truncation", router=fake)
    response = await Reranker(backend=backend).rerank(
        query="hello",
        candidates=["one", "two"],
        context=RerankContext(language="hi"),
    )
    assert fake.models == ["multilingual"]
    assert fake.route_states[0]["candidates"] == ["one", "two"]
    assert response.statistics.selected_backend == "laya-local"
    assert response.statistics.resolved_model == "multilingual"
    assert response.statistics.backend_metadata["routing"]["model"] == "multilingual"
    assert response.statistics.estimated_cost_usd is None
    assert response.statistics.model_calls == 1
    assert response.execution_plan.stages[0].backend == "laya-local"
    await backend.aclose()
    assert fake.unloaded


@pytest.mark.asyncio
async def test_explicit_checkpoint_and_zero_provider_budget() -> None:
    fake = RouterDouble()
    backend = LayaBackend(
        model="typed-decisions", context_policy="allow_provider_truncation", router=fake
    )
    result = await Reranker(backend=backend).rerank(
        query="choose",
        candidates=["one"],
        context=RerankContext(budget=Budget(max_cost_usd=0, strict_cost=True)),
    )
    assert result.results[0].score == 0.75
    assert fake.models == ["typed-decisions"]
    assert result.statistics.estimated_cost_usd is None
    await backend.aclose()


@pytest.mark.asyncio
async def test_auto_route_checks_multilingual_survivor_separately() -> None:
    fake = RouterDouble()
    backend = LayaBackend(context_policy="allow_provider_truncation", router=fake)
    response = await Reranker(backend=backend).rerank(
        query="best database", candidates=["ordinary English text", "हिंदी दस्तावेज़"]
    )
    assert response.statistics.resolved_model == "multilingual"
    assert response.statistics.backend_metadata["stage_routing"]["reason"] == (
        "multilingual survivor in candidate pool"
    )
    await backend.aclose()


@pytest.mark.asyncio
async def test_local_rejects_missing_id() -> None:
    backend = LayaBackend(
        context_policy="allow_provider_truncation", router=RouterDouble(malformed=True)
    )
    with pytest.raises(OutputValidationError):
        await backend.score(_request())
    await backend.aclose()


@pytest.mark.asyncio
async def test_local_worker_does_not_block_loop_and_serializes_by_default() -> None:
    fake = RouterDouble(delay=0.06)
    backend = LayaBackend(context_policy="allow_provider_truncation", router=fake)
    ticked = False

    async def ticker() -> None:
        nonlocal ticked
        await asyncio.sleep(0.01)
        ticked = True

    await asyncio.gather(backend.score(_request()), backend.score(_request()), ticker())
    assert ticked
    assert fake.maximum_active == 1
    await backend.aclose()


@pytest.mark.asyncio
async def test_cancelled_local_call_waits_for_worker_cleanup() -> None:
    fake = RouterDouble(delay=0.05)
    backend = LayaBackend(context_policy="allow_provider_truncation", router=fake)
    started = time.perf_counter()
    task = asyncio.create_task(backend.score(_request()))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.perf_counter() - started >= 0.04
    assert fake.active == 0
    await backend.aclose()


@pytest.mark.asyncio
async def test_local_context_limit_rejects_before_inference() -> None:
    fake = RouterDouble()
    backend = LayaBackend(
        context_policy="allow_provider_truncation", router=fake, max_context_tokens=32
    )
    with pytest.raises(ContextLimitError):
        await backend.score(_request())
    assert not fake.models
    await backend.aclose()


@pytest.mark.asyncio
async def test_missing_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = LayaBackend()
    monkeypatch.setitem(sys.modules, "laya", None)
    with pytest.raises(CapabilityError, match=r"typedrank\[laya\]"):
        await backend.score(_request())
    await backend.aclose()


@pytest.mark.asyncio
async def test_closed_local_backend_rejects_work() -> None:
    backend = LayaBackend(context_policy="allow_provider_truncation", router=RouterDouble())
    await backend.aclose()
    with pytest.raises(BackendError, match="closed"):
        await backend.score(_request())


def test_local_model_validation() -> None:
    with pytest.raises(ValueError):
        LayaBackend(max_inference_concurrency=0)
    with pytest.raises(ValueError, match="one inference worker"):
        LayaBackend(max_inference_concurrency=2)
    assert LayaBackend().capabilities.execution_location == "local"
    assert (
        LayaBackend(model="english").cache_identity
        != LayaBackend(model="multilingual").cache_identity
    )
    assert not LayaBackend().cache_stable
    assert LayaBackend(cache_revision="weights-v1").cache_stable


@pytest.mark.asyncio
async def test_cancelled_close_still_unloads_router() -> None:
    fake = RouterDouble(delay=0.08)
    backend = LayaBackend(context_policy="allow_provider_truncation", router=fake)
    task = asyncio.create_task(backend.score(_request()))
    await asyncio.sleep(0.01)
    close = asyncio.create_task(backend.aclose())
    await asyncio.sleep(0.01)
    close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close
    await task
    await backend.aclose()
    assert fake.unloaded
    assert backend.router is None


@pytest.mark.asyncio
async def test_local_cache_requires_declared_weight_revision() -> None:
    fake = RouterDouble()
    backend = LayaBackend(
        context_policy="allow_provider_truncation", router=fake, cache_revision="weights-v1"
    )
    ranker = Reranker(
        backend=backend,
        config=RerankerConfig(cache=CacheConfig(enabled=True)),
    )
    first = await ranker.rerank(query="hello", candidates=["one"])
    second = await ranker.rerank(query="hello", candidates=["one"])
    assert first.statistics.model_calls == 1
    assert second.statistics.cache_hits == 1
    assert len(fake.models) == 1
    await backend.aclose()


@pytest.mark.asyncio
async def test_concurrent_reranks_share_one_local_router() -> None:
    fake = RouterDouble(delay=0.02)
    backend = LayaBackend(context_policy="allow_provider_truncation", router=fake)
    ranker = Reranker(backend=backend)
    one, two = await asyncio.gather(
        ranker.rerank(query="first", candidates=["a", "b"]),
        ranker.rerank(query="second", candidates=["c", "d"]),
    )
    assert one.statistics.model_calls == two.statistics.model_calls == 1
    assert fake.maximum_active == 1
    assert len(fake.models) == 2
    await backend.aclose()
