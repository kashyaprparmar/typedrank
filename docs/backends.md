# Backends

TypedRank separates candidate preparation and ranking strategy from model execution. Choose a backend based on deployment constraints and validate ranking quality on your own data; no backend is universally best.

| Backend | Where inference runs | Requirements and behavior |
| --- | --- | --- |
| Jev | Hosted TypeSafe API | Requires `TYPESAFE_API_KEY`, network access, and the `jev` HTTP extra. It does not require local model management. |
| Laya local | In-process in your application | Self-hosted inference with optional CPU or GPU execution. Its multilingual Router selects among models. Install `typedrank[laya]`; the model runtime and weights add substantial optional dependencies and resource use. |
| Laya HTTP | Separate self-hosted process | Sends requests to a System One-compatible API, allowing inference to run outside the application process. Install `typedrank[http]` for the client transport. |
| Sentence Transformers | In-process embedding computation | Useful for embedding similarity and candidate pre-filtering. It is not equivalent to typed decision scoring from a model backend. Install `typedrank[embeddings]`. |

## Selecting a backend

```python
from typedrank import Reranker
from typedrank.backends import JevBackend

reranker = Reranker(backend=JevBackend())
response = await reranker.rerank(
    query="best vector database",
    candidates=documents,
    top_k=5,
)
```

When you construct and pass a backend instance, close it yourself after use. If you pass a model string instead, the ranker constructs and owns the backend. See [Jev](jev.md), [Laya](laya.md), and [backend routing](backend-routing.md) for configuration details.
