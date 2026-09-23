"""Opt-in check against a running Laya System One-compatible service."""

from __future__ import annotations

import os

import pytest

from typedrank.backends import BackendCandidate, LayaHTTPBackend, ModelRequest
from typedrank.prompts import DEFAULT_PROMPT

pytestmark = pytest.mark.laya_live


@pytest.mark.asyncio
async def test_laya_http_service_live() -> None:
    if os.getenv("TYPEDRANK_LAYA_HTTP_LIVE") != "1":
        pytest.skip("set TYPEDRANK_LAYA_HTTP_LIVE=1 to call a running Laya HTTP service")
    endpoint = os.getenv("TYPEDRANK_LAYA_ENDPOINT")
    if not endpoint:
        pytest.skip("TYPEDRANK_LAYA_ENDPOINT is required for the opt-in HTTP check")
    backend = LayaHTTPBackend(
        context_policy="allow_provider_truncation",
        endpoint=endpoint,
        api_key=os.getenv("TYPEDRANK_LAYA_HTTP_API_KEY"),
    )
    try:
        result = await backend.score(
            ModelRequest(
                "Which candidate describes vector search?",
                (BackendCandidate("candidate-a", "A vector index finds similar vectors."),),
                DEFAULT_PROMPT,
            )
        )
        assert len(result.scores) == 1
        assert 0 <= result.scores[0].score <= 1
    finally:
        await backend.aclose()
