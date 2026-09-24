# TypedRank

Universal typed reranking for RAG, search, agents and arbitrary Python objects.

![TypedRank architecture: typed reranking connects Jev, Laya and Python search workloads](https://raw.githubusercontent.com/kashyaprparmar/typedrank/main/docs/assets/typedrank-architecture.png)

## Install

```bash
python -m pip install typedrank
python -m pip install "typedrank[jev]"  # Jev's optional HTTP transport
python -m pip install "typedrank[laya]"  # optional local Laya and Torch runtime
python -m pip install "typedrank[http]"  # self-hosted Laya HTTP transport
python -m pip install "typedrank[embeddings]"  # optional Sentence Transformers
```

The core has no runtime dependencies and does not import Torch. Jev requests use the TypeSafe System One endpoint and `TYPESAFE_API_KEY`; the credential and Jev model IDs remain provider-specific.

## Quick start

```python
from typedrank import Reranker
from typedrank.backends import JevBackend

backend = JevBackend()
ranker = Reranker(backend=backend)
response = await ranker.rerank(
    query="How can I reset my password?",
    candidates=[
        "Open account settings and select Reset password.",
        "Our office is closed on Sunday.",
    ],
    top_k=2,
)
for result in response.results:
    print(result.rank, result.score, result.item)
await backend.aclose()
```

Configure `TYPESAFE_API_KEY` in the process environment before making a live Jev request. Calls can incur provider charges. No live requests run in the test suite by default.

For local Laya inference, install the Laya extra and keep one backend instance per service:

```python
from typedrank import AutoReranker
from typedrank.backends import LayaBackend

backend = LayaBackend(model="auto", device="cuda", preload=True)
async with AutoReranker(backend=backend) as reranker:
    response = await reranker.rerank(query="billing help", candidates=documents, top_k=10)
await backend.aclose()
```

`preload=True` loads checkpoints on the first use of the backend. Laya 0.3.7 uses one inference worker per Router; TypedRank rejects higher concurrency values. Laya is imported lazily, and normal tests use fakes rather than downloading weights. Local results leave monetary cost unknown.

For a self-hosted Laya server, use `LayaHTTPBackend(endpoint="http://localhost:8000/v1/systemone", context_policy="allow_provider_truncation")`. This explicit opt-in marks context validation unverified and results approximate; strict mode requires a deployment validator. Authentication is optional; pass `api_key` when the server requires a bearer token. Pass `model="english"`, `"multilingual"`, or `"typed-decisions"` to pin a checkpoint.

To fall back from a local model to Jev, compose explicit backends:

```python
from typedrank.backends import BackendRouter, JevBackend, LayaBackend

backend = BackendRouter(
    primary=LayaBackend(model="auto"),
    fallback=JevBackend(model="jev-1.13.0"),
)
reranker = AutoReranker(backend=backend)
```

`BackendRouter` chooses where a model stage runs. `AutoReranker` chooses the ranking strategy and shortlist. The route, fallback count, resolved model, and backend metadata are available in `response.execution_plan` and `response.statistics`. Auto shortlist limits are configurable in `RerankerConfig`; evaluate them against your own data.

Without a model backend, `Reranker()` uses deterministic lexical ranking. Custom Python objects remain intact in results; provide `text_fn` or a `CandidateAdapter` to select text explicitly.

```python
response = await Reranker().rerank(
    query="billing issue",
    candidates=tickets,
    text_fn=lambda ticket: f"{ticket.title}\n{ticket.description}",
    top_k=5,
)
```

## Included

TypedRank includes generic candidate preparation, pointwise and listwise strategies, metrics, BM25, embeddings, reciprocal-rank fusion, diversity selection, pipelines, budgets, caching, observability, evaluation, and Jev and Laya backends. The generic model pipeline stage is `ModelReranker`.

See [backend comparison](docs/backends.md), [Jev](docs/jev.md), [Laya](docs/laya.md), [backend routing](docs/backend-routing.md), [migration from Jev Rankkit](docs/migration-from-jev-rankkit.md), [usage guide](docs/usage.md), [integration adapters](docs/integrations.md), [evaluation](docs/evaluation.md), and [benchmarks](docs/benchmarks.md). Examples cover RAG, entities, tools, SQL schemas, memory, custom objects and metrics, hybrid pipelines, and evaluation. Thresholds and synthetic benchmarks are starting points; measure ranking quality on your own held-out data.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
mypy src/typedrank
python -m build
```

TypedRank is distributed under the MIT License. Source notices from the original Jev Rankkit project are preserved in the license.
