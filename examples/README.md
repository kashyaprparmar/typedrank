# Examples

Run an example from the repository root with `python examples/FILE.py`. Model-backed examples need the matching optional extra, local service, or credential described below.

| Topic | Example | Notes |
| --- | --- | --- |
| RAG | [02_rag_reranking.py](02_rag_reranking.py) | Retrieve a pool, rerank it, then prepare answer context. |
| Entities | [03_entity_reranking.py](03_entity_reranking.py) | Rank typed entity records. |
| Custom metrics | [04_custom_metrics.py](04_custom_metrics.py) | Combine lexical relevance with an application metric. |
| Hybrid pipeline | [06_hybrid_reranking.py](06_hybrid_reranking.py) | Combine retrieval signals and reranking stages. |
| Agent tools | [08_tool_reranking.py](08_tool_reranking.py) | Rank tool descriptions against a task. |
| SQL schemas | [09_sql_schema_reranking.py](09_sql_schema_reranking.py) | Select relevant schema objects. |
| Evaluation | [10_evaluation.py](10_evaluation.py) | Evaluate rankings against labeled examples. |
| Agent memory | [13_agent_memory.py](13_agent_memory.py) | Rank memory records for a query. |
| Arbitrary Python objects | [12_custom_objects.py](12_custom_objects.py) | Preserve original application objects in results. |
| Hosted Jev | [16_jev_backend.py](16_jev_backend.py) | Requires `typedrank[jev]` and `TYPESAFE_API_KEY`; calls a hosted API. |
| Local Laya | [17_laya_local.py](17_laya_local.py) | Requires `typedrank[laya]` and local model resources. |
| Laya over HTTP | [18_laya_http.py](18_laya_http.py) | Requires `typedrank[http]` and a running compatible service. |
| Backend fallback | [19_backend_fallback.py](19_backend_fallback.py) | Prefers local Laya and configures Jev as fallback. |

The provider examples call external or local model services. The remaining examples primarily use deterministic or fake/local ranking and can be inspected without provider credentials.
