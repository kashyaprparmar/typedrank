# Jev backend

`JevBackend` adapts TypedRank model requests to the hosted TypeSafe System One API. Inference runs remotely: configure `TYPESAFE_API_KEY` and allow network access. Jev does not require downloading or managing local model weights.

Install the HTTP client extra:

```bash
python -m pip install "typedrank[jev]"
```

```python
from typedrank import Reranker
from typedrank.backends import JevBackend

backend = JevBackend()  # reads TYPESAFE_API_KEY from the environment
reranker = Reranker(backend=backend)
response = await reranker.rerank(
    query="best vector database",
    candidates=documents,
    top_k=5,
)
await backend.aclose()
```

Set the key in your deployment environment; do not put credentials in source control. Live API calls require network connectivity and may incur provider charges. The default model is `jev-1.13.0`; configure `model` explicitly to select another supported Jev model. Backend and execution details are available in the response statistics and execution plan.
