from __future__ import annotations

from typing import Any

import pytest

from typedrank import Reranker
from typedrank.backends import BackendCandidate, LayaHTTPBackend, ModelRequest
from typedrank.config import RetryConfig
from typedrank.errors import (
    AuthenticationError,
    BackendError,
    CapabilityError,
    OutputValidationError,
)
from typedrank.prompts import DEFAULT_PROMPT


class Response:
    def __init__(self, status_code: int = 200, data: Any = None, text: str | None = None) -> None:
        self.status_code = status_code
        self._data = data
        self.text = text
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return self._data


class Transport:
    def __init__(self, response: Response | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self.closed = False

    async def post(self, url: str, *, headers: Any, json: Any, timeout_s: float) -> Response:
        del timeout_s
        self.calls.append((url, dict(headers), dict(json)))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def request() -> ModelRequest:
    return ModelRequest(
        "query",
        (BackendCandidate("id-1", "first"), BackendCandidate("id-2", "second")),
        DEFAULT_PROMPT,
        mode="listwise",
    )


def answer(ids: tuple[str, ...] = ("id-1", "id-2")) -> dict[str, Any]:
    return {
        "model": "laya-rl-agent",
        "answers": {key: {"type": "noul", "noul": 0.4} for key in ids},
        "usage": {"input_tokens": 9, "output_tokens": 0},
        "routing": {"model": "multilingual", "reason": "script"},
    }


@pytest.mark.asyncio
async def test_http_wire_auth_optional_and_routing_metadata() -> None:
    transport = Transport(Response(data=answer()))
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        endpoint="http://localhost:8000/v1/systemone",
        transport=transport,
    )
    result = await backend.score(request())
    assert [score.candidate_id for score in result.scores] == ["id-1", "id-2"]
    assert result.metadata["routing"]["model"] == "multilingual"
    assert result.statistics.usage.cost_usd is None
    assert "Authorization" not in transport.calls[0][1]
    assert "model" not in transport.calls[0][2]
    assert [item["content"] for item in transport.calls[0][2]["state"]["candidates"]] == [
        "first",
        "second",
    ]
    assert all(q["type"] == "noul" for q in transport.calls[0][2]["questions"].values())
    secured = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        api_key="secret",
        model="multilingual",
        transport=transport,
    )
    await secured.score(request())
    assert transport.calls[1][1]["Authorization"] == "Bearer secret"
    assert transport.calls[1][2]["model"] == "multilingual"


@pytest.mark.asyncio
async def test_http_strict_requires_deployment_context_validator() -> None:
    transport = Transport(Response(data=answer()))
    backend = LayaHTTPBackend(transport=transport)
    with pytest.raises(CapabilityError, match="deployment validator"):
        await backend.score(request())
    assert not transport.calls


@pytest.mark.asyncio
async def test_http_strict_calls_deployment_validator() -> None:
    seen: list[dict[str, Any]] = []
    backend = LayaHTTPBackend(
        transport=Transport(Response(data=answer())),
        context_validator=lambda payload: seen.append(dict(payload)),
    )
    result = await backend.score(request())
    assert seen and "state" in seen[0]
    assert "context_validation" not in result.metadata


def test_laya_http_reasoning_diagnostic_is_provider_specific() -> None:
    with pytest.raises(CapabilityError, match="Laya HTTP"):
        LayaHTTPBackend(reasoning_level="high")


@pytest.mark.asyncio
async def test_http_truncation_opt_in_marks_result_approximate() -> None:
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        transport=Transport(Response(data=answer(("c00000000", "c00000001")))),
    )
    result = await Reranker(backend, strategy="listwise").rerank(
        query="query", candidates=["first", "second"]
    )
    assert result.execution_plan.approximate
    assert any("not validated" in warning for warning in result.warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [("id-1",), ("id-1", "id-2", "surprise")])
async def test_http_rejects_bad_coverage(ids: tuple[str, ...]) -> None:
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation", transport=Transport(Response(data=answer(ids)))
    )
    with pytest.raises(OutputValidationError):
        await backend.score(request())


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, "0.5", True])
async def test_http_rejects_bad_scores(value: Any) -> None:
    data = answer()
    data["answers"]["id-1"]["noul"] = value
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation", transport=Transport(Response(data=data))
    )
    with pytest.raises(OutputValidationError):
        await backend.score(request())


@pytest.mark.asyncio
async def test_http_rejects_duplicate_json_keys() -> None:
    body = '{"answers":{"id-1":{"type":"noul","noul":0.2},"id-1":{"type":"noul","noul":0.9}}}'
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation", transport=Transport(Response(text=body))
    )
    with pytest.raises(OutputValidationError, match="duplicate"):
        await backend.score(request())


@pytest.mark.asyncio
async def test_http_explicit_checkpoint_must_be_confirmed() -> None:
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        model="english",
        transport=Transport(Response(data=answer())),
    )
    with pytest.raises(OutputValidationError, match="checkpoint"):
        await backend.score(request())


@pytest.mark.asyncio
async def test_http_auto_requires_resolved_checkpoint() -> None:
    data = answer()
    data.pop("routing")
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation", transport=Transport(Response(data=data))
    )
    with pytest.raises(OutputValidationError, match="identify its checkpoint"):
        await backend.score(request())


def test_http_cache_identity_includes_endpoint_and_model() -> None:
    one = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        endpoint="http://one/v1/systemone",
        model="english",
    )
    two = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        endpoint="http://two/v1/systemone",
        model="english",
    )
    three = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        endpoint="http://one/v1/systemone",
        model="multilingual",
    )
    assert len({one.cache_identity, two.cache_identity, three.cache_identity}) == 3


@pytest.mark.asyncio
async def test_http_auth_timeout_and_server_error() -> None:
    for response, error in [
        (Response(status_code=401), AuthenticationError),
        (Response(status_code=503), BackendError),
        (TimeoutError("timeout"), BackendError),
    ]:
        backend = LayaHTTPBackend(
            context_policy="allow_provider_truncation",
            transport=Transport(response),
            retry=RetryConfig(max_attempts=1),
        )
        with pytest.raises(error):
            await backend.score(request())


@pytest.mark.asyncio
async def test_http_timeout_has_laya_diagnostic() -> None:
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        transport=Transport(TimeoutError("timeout")),
        retry=RetryConfig(max_attempts=1),
    )
    with pytest.raises(BackendError, match="Laya HTTP request timed out") as error:
        await backend.score(request())
    assert error.value.details.stage == "laya-http"


@pytest.mark.asyncio
async def test_http_partial_only_when_requested() -> None:
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        transport=Transport(Response(data=answer(("id-1",)))),
    )
    result = await backend.score(
        ModelRequest("query", request().candidates, DEFAULT_PROMPT, "listwise", allow_partial=True)
    )
    assert result.missing_candidate_ids == ("id-2",)
