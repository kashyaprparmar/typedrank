# Usage guide

TypedRank reranks a candidate pool you already have. The core package needs no API key or model service; install optional extras for the backends you use.

## Install

```bash
python -m pip install typedrank
```

For TypeSafe AI Jev scoring, install the HTTP extra:

```bash
python -m pip install "typedrank[jev]"
```

For local Sentence Transformers embeddings, install `typedrank[embeddings]`. Its model runtime and model weights are optional and can be large.

## Rank candidates locally

With no model configured, the default ranker uses deterministic lexical scoring. This is useful for a zero-credential quick start and as a cheap first stage.

```python
import asyncio

from typedrank import Reranker


async def main() -> None:
    candidates = [
        "Cooking notes and recipes",
        "A guide to vector search databases",
        "SQL indexing reference",
    ]

    async with Reranker() as reranker:
        response = await reranker.rerank(
            query="best database for vector search",
            candidates=candidates,
            top_k=2,
        )

    for result in response.results:
        print(result.rank, result.item, result.score)
    print("Selected route:", response.execution_plan.strategy)


asyncio.run(main())
```

The returned `result.item` is the original input object. Use `top_k=None` to request a full ordering. `rerank_sync(...)` is available for synchronous programs that are not already running an event loop; async applications should use `await reranker.rerank(...)`.

## Rerank your own Python objects

Objects do not need to inherit from a library base class. Tell the ranker which fields should be used as text:

```python
from dataclasses import dataclass

from typedrank import Reranker


@dataclass
class Product:
    sku: str
    name: str
    description: str


products = [
    Product("db-1", "VectorStore", "Managed vector search database"),
    Product("sql-1", "SQL Server", "Relational database with SQL indexes"),
]


async def best_product_for(query: str) -> Product:
    async with Reranker() as reranker:
        response = await reranker.rerank(
            query=query,
            candidates=products,
            text_fn=lambda item: f"{item.name}\n{item.description}",
            top_k=1,
        )
    return response.results[0].item
```

Use `metadata_fn` for metadata metrics and `id_fn` for caller-supplied display IDs. Candidate objects are not serialized wholesale: model requests receive the projected text, while local metrics can also use metadata you explicitly project.

## Using the TypeSafe AI Jev API

Create an API key in your TypeSafe AI account, then provide it to the application as the `TYPESAFE_API_KEY` environment variable. The Jev backend reads this variable when it makes a request, so do not put the key in Python source, README files, or version control. For a deployed service, use your platform's secret manager.

For a temporary PowerShell session, enter the key at the secure prompt:

```powershell
$secureKey = Read-Host "TypeSafe API key" -AsSecureString
$env:TYPESAFE_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
python app.py
Remove-Item Env:TYPESAFE_API_KEY
```

For Bash or Zsh, read it without echoing it and export it only for the current shell:

```bash
read -rsp "TypeSafe API key: " TYPESAFE_API_KEY
printf '\n'
export TYPESAFE_API_KEY
python app.py
unset TYPESAFE_API_KEY
```

Then configure the TypeSafe model in Python. `auto` selects a bounded strategy based on the candidate pool and available budget; pass `"listwise"` or `"pointwise"` when you want to select the strategy explicitly.

```python
import asyncio

from typedrank import Reranker


async def main() -> None:
    candidates = [
        "A review of vector database indexing and retrieval",
        "A recipe collection for home cooking",
        "A comparison of vector search databases",
    ]

    async with Reranker(
        model="typesafe:jev-1.13.0",
        strategy="auto",
    ) as reranker:
        response = await reranker.rerank(
            query="best database for vector search",
            candidates=candidates,
            top_k=2,
        )

    for result in response.results:
        print(result.rank, result.item, result.score)
    print("Strategy:", response.execution_plan.strategy)
    print("Model calls:", response.stats.model_calls)


asyncio.run(main())
```

Jev returns relevance judgments but this adapter does not provide free-form explanations or a reasoning-level setting. Usage and exact cost may be unknown depending on what the provider reports. Model calls can incur charges; use small candidate pools, a `Budget`, and your own quality and cost measurements before increasing traffic. The automatic route and its rationale are available on `response.execution_plan`.

## Local and self-hosted Laya

Install `typedrank[laya]` for in-process Laya. It imports and loads the model only when the backend is first used; model weights may download then. The HTTP adapter needs only `typedrank[http]` and a running Laya server.

```python
from typedrank import AutoReranker, RerankContext
from typedrank.backends import BackendRouter, JevBackend, LayaBackend, LayaHTTPBackend

local = LayaBackend(model="auto", max_inference_concurrency=1)
router = BackendRouter(primary=local, fallback=JevBackend())
response = await AutoReranker(backend=router).rerank(
    query="billing support",
    candidates=tickets,
    text_fn=lambda ticket: ticket.body,
    top_k=10,
    context=RerankContext(language="en"),
)
print(response.statistics.selected_backend, response.statistics.resolved_model)
print(response.statistics.backend_latency_ms, response.statistics.fallbacks)
await router.aclose()  # closes both configured child backends

server = LayaHTTPBackend(endpoint="http://localhost:8000/v1/systemone")
```

`BackendRouter` defaults to the declared primary. Optional `local_first`, `remote_first`, `language_aware`, `cost_aware`, and `latency_aware` policies use declared capabilities or caller-provided measurements. `quality_mode="offline"` permits a local backend and excludes remote calls. Use `network_policy="deny"` to block remote execution independently of the chosen strategy.

Laya uses independent binary `noul` questions, batched up to `max_batch_size`; candidate IDs are preserved. The default context gate is conservative at 512 estimated tokens per decision and can be configured for a known checkpoint. Local Laya and automatic Laya HTTP routes are not cached by default. Declare `cache_revision` only when the model weights or server deployment are fixed; it becomes part of the cache identity but does not itself pin weights. Monetary cost remains unknown for local execution, while a strict provider-charge budget can still treat it as having no provider API charge.

## Rerank for RAG

Retrieve a manageable candidate pool first, then rerank it before building the generation context. This example accepts common document objects with a `page_content` field:

```python
from typing import Protocol

from typedrank import Reranker


class Document(Protocol):
    page_content: str


async def select_context(query: str, retrieved_documents: list[Document]) -> str:
    async with Reranker(
        model="typesafe:jev-1.13.0",
        strategy="auto",
    ) as reranker:
        response = await reranker.rerank_documents(
            query=query,
            candidates=retrieved_documents,
            top_k=5,
        )

    return "\n\n".join(result.item.page_content for result in response.results)
```

Apply authorization and metadata filters before reranking, keep candidate text concise, and evaluate retrieval recall and answer evidence quality on your own data. `AutoReranker` does not replace retrieval for very large collections; use a retriever or cheap filter to reduce the pool first.

## Inspect results and execution

Each response has `results`, `stats`, `execution_plan`, `status`, and `coverage`. A result includes the original `item`, its `score`, `rank`, per-metric values, and metadata. Inspect the execution plan and statistics to see which route ran, how many model calls it made, and whether usage or cost is known. See [evaluation](evaluation.md), [integrations](integrations.md), and the runnable [examples](../examples/).
