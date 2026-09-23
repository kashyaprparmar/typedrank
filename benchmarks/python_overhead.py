"""Reproducible in-process overhead probe; no network or model downloads.

Run: python benchmarks/python_overhead.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typedrank import AutoReranker, Reranker
from typedrank.backends import FakeBackend


class FakeEmbeddings:
    backend_id = "fake-embeddings"
    cache_identity = "fake-embeddings:v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def aclose(self) -> None:
        return None


async def measure(count: int, auto: bool, embeddings: bool = False, repetitions: int = 20) -> float:
    candidates = [f"document {index} about typed ranking" for index in range(count)]
    samples = []
    for _ in range(repetitions):
        backend = FakeBackend()
        ranker = (
            AutoReranker(backend, embedding_backend=FakeEmbeddings() if embeddings else None)
            if auto
            else Reranker(backend)
        )
        started = time.perf_counter()
        await ranker.rerank(query="typed ranking", candidates=candidates, top_k=10)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


async def main() -> None:
    for count, auto, embeddings in (
        (10, False, False),
        (100, False, False),
        (500, True, False),
        (500, True, True),
    ):
        elapsed = await measure(count, auto, embeddings)
        route = "auto+embeddings" if embeddings else ("auto" if auto else "pointwise")
        print(f"{count} candidates, {route}: {elapsed:.2f} ms median")


if __name__ == "__main__":
    asyncio.run(main())
