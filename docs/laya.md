# Laya backends

TypedRank offers two Laya deployment modes. Both speak TypedRank's typed model-backend contract and use Laya model routing. The local mode is in-process; the HTTP mode sends requests to a separate System One-compatible server.

## Local inference

Install `typedrank[laya]`. This optional runtime includes heavy machine-learning dependencies and model weights may require substantial disk, memory, and CPU/GPU resources.

```python
from typedrank import Reranker
from typedrank.backends import LayaBackend

backend = LayaBackend(model="auto", device="cuda")  # use device="cpu" for CPU execution
reranker = Reranker(backend=backend)
response = await reranker.rerank(query="billing help", candidates=tickets, top_k=5)
await backend.aclose()
```

`model="auto"` delegates model selection to Laya's multilingual Router using the query and survivor texts. You can pin a supported model instead. `preload=True` requests eager loading on first backend use. Laya 0.3.7 supports one inference worker per Router; use separate processes to scale. Strict context validation loads the selected checkpoint's tokenizer and rejects requests that Laya would truncate. It can raise `CapabilityError` if the inspected tokenizer layout changes. Set `context_policy="allow_provider_truncation"` only when truncation risk is acceptable; those results are marked approximate and context validation unverified. Local inference may use CPU or GPU. Configure Torch thread limits at the application level when needed.

## HTTP deployment

Run Laya's System One-compatible service separately, then install the lightweight HTTP extra in the TypedRank application:

```bash
python -m pip install "typedrank[http]"
```

```python
from typedrank import Reranker
from typedrank.backends import LayaHTTPBackend

backend = LayaHTTPBackend(
    endpoint="http://localhost:8000/v1/systemone",
    context_policy="allow_provider_truncation",
)
reranker = Reranker(backend=backend)
response = await reranker.rerank(query="billing help", candidates=tickets, top_k=5)
await backend.aclose()
```

HTTP mode provides process isolation: the model server and application can be deployed and scaled separately. Pass `api_key` if the server expects bearer authentication. The endpoint must implement the System One request and response contract used by TypedRank. Strict mode is the default and requires a deployment `context_validator` callback that raises on truncation. A stock server does not supply that guarantee; the example explicitly opts into provider truncation, which marks results approximate. With `model="auto"`, a multi-request stage is rejected if responses resolve to different checkpoints.
