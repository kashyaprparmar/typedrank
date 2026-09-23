"""Opt-in integration test; downloads checkpoint weights on first use."""

from __future__ import annotations

import os

import pytest

from typedrank.backends import BackendCandidate, LayaBackend, ModelRequest
from typedrank.prompts import DEFAULT_PROMPT


@pytest.mark.laya_live
@pytest.mark.asyncio
async def test_local_laya_checkpoint_live() -> None:
    if os.getenv("TYPEDRANK_LAYA_LIVE") != "1":
        pytest.skip("set TYPEDRANK_LAYA_LIVE=1 to download and run a Laya checkpoint")
    pytest.importorskip("laya")
    backend = LayaBackend(model="english")
    try:
        result = await backend.score(
            ModelRequest(
                "How to reset a password?",
                (BackendCandidate("relevant", "Open settings and select Reset password."),),
                DEFAULT_PROMPT,
            )
        )
        assert len(result.scores) == 1
        assert 0 <= result.scores[0].score <= 1
    finally:
        await backend.aclose()
