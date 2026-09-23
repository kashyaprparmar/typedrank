"""Call a separately hosted Laya System One-compatible service."""

import asyncio

from typedrank import Reranker
from typedrank.backends import LayaHTTPBackend


async def main() -> None:
    backend = LayaHTTPBackend(
        endpoint="http://localhost:8000/v1/systemone",
        context_policy="allow_provider_truncation",
    )
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
