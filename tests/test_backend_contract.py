"""A common contract run against every shipped System One model adapter."""

from __future__ import annotations

from typing import Any

import pytest

from typedrank.backends import (
    BackendCandidate,
    JevBackend,
    LayaBackend,
    LayaHTTPBackend,
    ModelBackend,
    ModelRequest,
)
from typedrank.backends.systemone import (
    SystemOneQuestion,
    SystemOneRequest,
    validate_typed_answer,
)
from typedrank.config import RetryConfig
from typedrank.errors import OutputValidationError
from typedrank.prompts import DEFAULT_PROMPT


class Response:
    status_code = 200

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return {
            "answers": self.answers,
            "usage": {"input_tokens": 1, "output_tokens": 0},
            "routing": {"model": "english", "reason": "test"},
        }


class Transport:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers

    async def post(self, url: str, *, headers: Any, json: Any, timeout_s: float) -> Response:
        del url, headers, json, timeout_s
        return Response(self.answers)

    async def aclose(self) -> None:
        return None


class Router:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers

    def predict(self, state: Any, questions: Any, model: str | None = None) -> dict[str, Any]:
        del state, questions, model
        return {"answers": self.answers, "usage": {"input_tokens": 1, "output_tokens": 0}}

    def unload(self) -> None:
        return None


def backend(kind: str, answers: dict[str, Any]) -> ModelBackend:
    if kind == "jev":
        return JevBackend(
            api_key="test", transport=Transport(answers), retry=RetryConfig(max_attempts=1)
        )
    if kind == "laya-http":
        return LayaHTTPBackend(
            context_policy="allow_provider_truncation",
            transport=Transport(answers),
            retry=RetryConfig(max_attempts=1),
        )
    return LayaBackend(context_policy="allow_provider_truncation", router=Router(answers))


def request(ids: tuple[str, ...] = ("a", "b")) -> ModelRequest:
    return ModelRequest(
        "query",
        tuple(BackendCandidate(key, f"text {key}") for key in ids),
        DEFAULT_PROMPT,
        mode="listwise",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["jev", "laya-local", "laya-http"])
async def test_system_one_backend_contract(kind: str) -> None:
    adapter = backend(
        kind, {"a": {"type": "noul", "noul": 0.2}, "b": {"type": "noul", "noul": 0.8}}
    )
    assert isinstance(adapter, ModelBackend)
    result = await adapter.score(request())
    assert tuple(score.candidate_id for score in result.scores) == ("a", "b")
    assert tuple(score.score for score in result.scores) == (0.2, 0.8)
    assert all(score.raw_kind == "noul_probability" for score in result.scores)
    await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["jev", "laya-local", "laya-http"])
async def test_system_one_contract_rejects_missing_ids(kind: str) -> None:
    adapter = backend(kind, {"a": {"type": "noul", "noul": 0.2}})
    with pytest.raises(OutputValidationError):
        await adapter.score(request())
    await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["jev", "laya-local", "laya-http"])
async def test_system_one_contract_rejects_duplicate_request_ids(kind: str) -> None:
    adapter = backend(kind, {"a": {"type": "noul", "noul": 0.2}})
    with pytest.raises(OutputValidationError):
        await adapter.score(request(("a", "a")))
    await adapter.aclose()


def test_internal_system_one_representation_supports_all_typed_primitives() -> None:
    wire = SystemOneRequest(
        state={"query": "refund"},
        questions={
            "department": SystemOneQuestion(
                "choice", "Choose a department", {"billing": "payments", "sales": "orders"}
            ),
            "urgency": SystemOneQuestion("score", "Rate urgency", ["low", "medium", "high"]),
            "relevant": SystemOneQuestion("noul", "Is this relevant?"),
        },
    ).to_payload()
    assert set(wire["questions"]) == {"department", "urgency", "relevant"}
    assert (
        validate_typed_answer(
            {"type": "choice", "choice": "billing"},
            kind="choice",
            criteria={"billing": "payments"},
        )
        == "billing"
    )
    assert (
        validate_typed_answer(
            {"type": "score", "score": 1.5}, kind="score", criteria=["low", "medium", "high"]
        )
        == 1.5
    )
    assert validate_typed_answer({"type": "noul", "noul": 0.5}, kind="noul") == 0.5
    with pytest.raises(OutputValidationError):
        validate_typed_answer(
            {"type": "choice", "choice": "unknown"},
            kind="choice",
            criteria={"billing": "payments"},
        )
