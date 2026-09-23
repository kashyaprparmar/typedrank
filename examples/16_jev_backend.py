"""Rank documents with the hosted Jev backend (requires TYPESAFE_API_KEY)."""

import asyncio

from typedrank import Reranker
from typedrank.backends import JevBackend


async def main() -> None:
    documents = [
        {"title": "Vector database guide", "text": "Compare vector search systems."},
        {"title": "Office calendar", "text": "Holiday and meeting dates."},
    ]
    backend = JevBackend()
    try:
        response = await Reranker(backend=backend).rerank(
            query="best vector database",
            candidates=documents,
            text_fn=lambda document: f"{document['title']}\n{document['text']}",
            top_k=1,
        )
        for result in response.results:
            print(result.rank, result.score, result.item)
    finally:
        await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
