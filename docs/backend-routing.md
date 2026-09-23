# Backend routing

`BackendRouter` selects which backend handles model stages; a ranking strategy still determines how candidates are scored. A router can use a fallback backend when the selected backend fails.

```python
from typedrank import Reranker
from typedrank.backends import BackendRouter, JevBackend, LayaBackend

backend = BackendRouter(
    primary=LayaBackend(model="auto"),
    fallback=JevBackend(),
    policy="local_first",
)
reranker = Reranker(backend=backend)
response = await reranker.rerank(
    query="best vector database",
    candidates=documents,
    top_k=5,
)
print(response.statistics.fallbacks, response.statistics.selected_backend)
await backend.aclose()  # closes the router and its child backends
```

Policies include `availability`, `local_first`, `remote_first`, `language_aware`, `cost_aware`, and `latency_aware`. Cost and latency policies use caller-supplied measurements keyed by backend cache identity (or an unambiguous backend ID); incomplete measurements leave the configured primary first. Network-deny and offline request contexts require an explicit `execution_location="local"` capability; unknown locality is excluded. These declarations are trusted application metadata, not an operating-system network sandbox. Availability routing does not mean automatic health checking: it checks backend capability and any exposed local availability signal.

Routing is explicit. It does not establish that one provider is faster, cheaper, or more accurate. Inspect `response.execution_plan` and `response.statistics` to see the selected route and fallback activity. See [backend comparison](backends.md) for deployment characteristics.
