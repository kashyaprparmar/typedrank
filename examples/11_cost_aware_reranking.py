"""Budget controls; fake usage is deterministic and not a provider price quote."""

import asyncio

from typedrank import Budget, RerankContext, Reranker
from typedrank.backends import FakeModelBackend


async def main() -> None:
    context = RerankContext(budget=Budget(max_model_calls=3, max_tokens=200, max_latency_ms=1000))
    response = await Reranker(FakeModelBackend(), strategy="pointwise").rerank(
        query="vector database",
        candidates=["vector search", "SQL database", "image processing"],
        top_k=2,
        context=context,
    )
    print(response.stats.model_calls, response.stats.input_tokens)
    print(response.stats.cost_confidence)


if __name__ == "__main__":
    asyncio.run(main())
