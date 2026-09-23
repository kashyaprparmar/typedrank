"""Run in-process Laya inference (install typedrank[laya])."""

import asyncio

from typedrank import Reranker
from typedrank.backends import LayaBackend


async def main() -> None:
    backend = LayaBackend(model="auto", device="cuda")
    try:
        response = await Reranker(backend=backend).rerank(
            query="payment support",
            candidates=[
                "Refund status is available in billing settings.",
                "Office hours are 9 to 5.",
            ],
            top_k=1,
        )
        print(response.results)
    finally:
        await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
