"""Prefer local Laya and route to hosted Jev when needed."""

import asyncio

from typedrank import Reranker
from typedrank.backends import BackendRouter, JevBackend, LayaBackend


async def main() -> None:
    backend = BackendRouter(
        primary=LayaBackend(model="auto"),
        fallback=JevBackend(),
        policy="local_first",
    )
    try:
        response = await Reranker(backend=backend).rerank(
            query="best vector database",
            candidates=["A guide to vector search.", "A guide to office calendars."],
            top_k=1,
        )
        print(response.results)
        print("backend:", response.statistics.selected_backend)
        print("fallbacks:", response.statistics.fallbacks)
    finally:
        await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
