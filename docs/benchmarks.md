# Reproducible benchmark smoke test

The [synthetic dataset](../benchmarks/data/synthetic_v1.json) has six original, small candidate pools: general retrieval, technical RAG, entity search, products, tool selection, and SQL schema selection. The script shuffles each fixed pool with seed `20260923` and records the dataset SHA-256. This is an integration and measurement smoke test, **not** a representative quality benchmark.

```bash
uv sync --group dev
uv run python benchmarks/compare.py --output benchmark-local.json
```

The default compares input ordering and candidate-pool BM25. Optional rows cover Sentence Transformers, Jev pointwise/listwise, local Laya, Laya HTTP, and BM25 → embedding → model hybrid pipelines. Rows that were not run are `unmeasured` with a reason; attempted runs that fail are `failed`. No numeric field is invented for an unavailable run.

To opt into a local embedding model, install `typedrank[embeddings]` and pass `--embedding-model MODEL_ID` (ideally also `--embedding-revision REVISION`). Model weights may download on first use. To permit billable Jev requests, install `typedrank[jev]`, set `TYPESAFE_API_KEY`, and pass `--include-jev`. For local Laya, install `typedrank[laya]` and pass `--include-laya-local`, optionally setting `--laya-device`, `--laya-model`, and `--laya-batch-size`. For a separate Laya service, install `typedrank[http]` and pass `--include-laya-http --laya-endpoint URL --allow-laya-http-provider-truncation`. Hybrid model rows run only when both an embedding model and the corresponding model backend are enabled.

Measured rows report NDCG@5, MRR@5, Recall@5, elapsed latency, cases per second, model/API calls, tokens when usage is complete, and provider API cost when known. Local Laya rows report `configured_device`, `configured_max_batch_size`, `first_case_latency_ms`, and `later_case_median_latency_ms`. These are configuration and heterogeneous case timings, not observed device, actual batch size, isolated cold start, or warm inference measurements; `cold_start_ms` and `warm_latency_ms` remain null until a comparable repeated-load benchmark is added. The harness records actual runs in this project; it does not reuse Laya's published performance figures.

Jev cost is an estimate only when you supply both `--jev-input-price-per-million-usd` and `--jev-output-price-per-million-usd`; otherwise the value remains unknown. Use your current contracted prices. Local inference does not receive an invented dollar cost.

External model versions, machine hardware, rate limits, and candidate order affect results. Never compare model quality using this toy dataset alone.

The separate [Python overhead probe](../benchmarks/python_overhead.py) measures fake-model calls and an automatic cascade without network or model inference. Run `python benchmarks/python_overhead.py` to reproduce it. For production decisions, use a held-out labeled pool from your own domain, collect uncertainty across multiple queries and runs, and compare quality against latency and cost at the same candidate budget.
