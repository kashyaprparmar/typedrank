"""Fixed-pool ranking comparison. Model calls require explicit flags.

Run: uv run python benchmarks/compare.py
Optional: uv run --extra embeddings python benchmarks/compare.py --embedding-model <model-id>
Optional Jev: TYPESAFE_API_KEY=... uv run --extra jev python benchmarks/compare.py --include-jev
Optional Laya: uv run --extra laya python benchmarks/compare.py --include-laya-local
Optional Laya HTTP: uv run --extra http python benchmarks/compare.py --include-laya-http
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import statistics
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from typedrank import Reranker
from typedrank.backends import JevBackend, LayaBackend, LayaHTTPBackend, SentenceTransformerBackend
from typedrank.candidates import CandidateView
from typedrank.context import RerankContext
from typedrank.evaluation import MRR, NDCG, EvaluationCase, EvaluationDataset, Recall, evaluate
from typedrank.metrics import BM25Metric, EmbeddingSimilarity, LexicalRelevance, WeightedMetrics
from typedrank.pipeline import BM25Filter, EmbeddingReranker, ModelReranker, RerankPipeline
from typedrank.strategies import EvaluationServices, MetricStrategy
from typedrank.types import ExecutionStage, RankingEntry, RankingOutcome, ScoreKind

DATA = Path(__file__).parent / "data" / "synthetic_v1.json"
METRICS = (NDCG(k=5), Recall(k=5), MRR(k=5))
ORDER_SEED = 20260923


class BaselineOrder:
    name = "baseline_order"

    async def rank(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[Any]],
        context: RerankContext,
        services: EvaluationServices,
        top_k: int | None,
    ) -> RankingOutcome:
        del query, context, services
        selected = candidates if top_k is None else candidates[:top_k]
        return RankingOutcome(
            tuple(RankingEntry(candidate.occurrence_id, None) for candidate in selected),
            score_kind=ScoreKind.RANK_ONLY,
            stages=(ExecutionStage(self.name, self.name, len(candidates), len(selected)),),
        )


def load_dataset() -> tuple[EvaluationDataset[str], str]:
    raw = DATA.read_bytes()
    data = json.loads(raw)
    rng = random.Random(ORDER_SEED)
    cases = []
    for entry in data["cases"]:
        candidates = list(entry["candidates"])
        rng.shuffle(candidates)
        cases.append(
            EvaluationCase(
                case_id=entry["case_id"],
                query=entry["query"],
                candidates=tuple(candidate["text"] for candidate in candidates),
                relevance=tuple(float(candidate["relevance"]) for candidate in candidates),
            )
        )
    return (
        EvaluationDataset(data["name"], data["version"], tuple(cases)),
        hashlib.sha256(raw).hexdigest(),
    )


async def measured(name: str, ranker: Reranker, dataset: EvaluationDataset[str]) -> dict[str, Any]:
    try:
        report = await evaluate(ranker, dataset, metrics=METRICS, rerank_top_k=5)
        performance = report.performance
        return {
            "strategy": name,
            "status": "measured",
            "quality": dict(report.metrics),
            "latency_ms": performance.elapsed_ms,
            "throughput_cases_per_s": performance.throughput_cases_per_s,
            "model_calls": performance.model_calls,
            "input_tokens": performance.input_tokens if performance.usage_complete else None,
            "output_tokens": performance.output_tokens if performance.usage_complete else None,
            "estimated_cost_usd": performance.estimated_cost_usd,
            "cache_hit_rate": performance.cache_hit_rate,
            "case_latency_ms": [case.stats.total_latency_ms for case in report.cases],
        }
    except Exception as exc:
        return {"strategy": name, "status": "failed", "error_type": type(exc).__name__}
    finally:
        await ranker.aclose()


def unmeasured(name: str, reason: str) -> dict[str, Any]:
    return {"strategy": name, "status": "unmeasured", "reason": reason}


async def measured_model(
    name: str,
    backend: JevBackend | LayaBackend | LayaHTTPBackend,
    dataset: EvaluationDataset[str],
    *,
    strategy: str | RerankPipeline[str] = "listwise",
    embedding: SentenceTransformerBackend | None = None,
) -> dict[str, Any]:
    try:
        row = await measured(
            name,
            Reranker(backend=backend, strategy=strategy, embedding_backend=embedding),
            dataset,
        )
        if isinstance(backend, LayaBackend):
            row["configured_device"] = backend.device or "auto"
            row["configured_max_batch_size"] = backend.max_batch_size
            if row["status"] == "measured":
                latencies = row.pop("case_latency_ms")
                row["first_case_latency_ms"] = latencies[0]
                row["later_case_median_latency_ms"] = (
                    statistics.median(latencies[1:]) if len(latencies) > 1 else None
                )
                row["cold_start_ms"] = None
                row["warm_latency_ms"] = None
        else:
            row.pop("case_latency_ms", None)
        return row
    finally:
        await backend.aclose()


def hybrid_pipeline(embedding: SentenceTransformerBackend) -> RerankPipeline[str]:
    return RerankPipeline(
        [
            BM25Filter(limit=6),
            EmbeddingReranker(EmbeddingSimilarity(embedding), limit=5),
            ModelReranker(limit=5),
        ]
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", help="Installed sentence-transformers model ID")
    parser.add_argument("--embedding-revision", help="Pinned model revision, if available")
    parser.add_argument("--include-jev", action="store_true", help="Permit billable Jev calls")
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--jev-input-price-per-million-usd", type=Decimal)
    parser.add_argument("--jev-output-price-per-million-usd", type=Decimal)
    parser.add_argument(
        "--include-laya-local", action="store_true", help="Run local Laya inference"
    )
    parser.add_argument("--laya-model", default="auto")
    parser.add_argument("--laya-device", help="Local inference device, for example cpu or cuda")
    parser.add_argument("--laya-batch-size", type=int, default=16)
    parser.add_argument("--include-laya-http", action="store_true", help="Call a Laya HTTP service")
    parser.add_argument("--laya-endpoint", default="http://localhost:8000/v1/systemone")
    parser.add_argument(
        "--allow-laya-http-provider-truncation",
        action="store_true",
        help="Explicitly accept unverified Laya HTTP context validation",
    )
    parser.add_argument(
        "--laya-http-api-key", help="Bearer token for a protected Laya HTTP service"
    )
    parser.add_argument("--output", type=Path, help="Write JSON report to this file")
    args = parser.parse_args()
    if (args.jev_input_price_per_million_usd is None) != (
        args.jev_output_price_per_million_usd is None
    ):
        parser.error("supply both Jev token prices or neither")
    if args.laya_batch_size < 1:
        parser.error("--laya-batch-size must be positive")
    if args.include_laya_http and not args.allow_laya_http_provider_truncation:
        parser.error("Laya HTTP benchmark needs --allow-laya-http-provider-truncation")
    dataset, digest = load_dataset()
    rows: list[dict[str, Any]] = []
    rows.append(await measured("baseline_order", Reranker(strategy=BaselineOrder()), dataset))
    rows.append(
        await measured(
            "bm25",
            Reranker(strategy=MetricStrategy(WeightedMetrics.equal((BM25Metric(),)))),
            dataset,
        )
    )

    embedding = (
        SentenceTransformerBackend(args.embedding_model, revision=args.embedding_revision)
        if args.embedding_model
        else None
    )
    if embedding is None:
        for name in ("sentence_transformer", "hybrid_lexical_embedding", "hierarchical"):
            rows.append(unmeasured(name, "pass --embedding-model and install the embeddings extra"))
    else:
        rows.append(
            await measured(
                "sentence_transformer",
                Reranker(
                    strategy=MetricStrategy(
                        WeightedMetrics.equal((EmbeddingSimilarity(embedding),))
                    ),
                    embedding_backend=embedding,
                ),
                dataset,
            )
        )
        rows.append(
            await measured(
                "hybrid_lexical_embedding",
                Reranker(
                    strategy=MetricStrategy(
                        WeightedMetrics(
                            (LexicalRelevance(), EmbeddingSimilarity(embedding)),
                            {"lexical": 0.4, "semantic": 0.6},
                        )
                    ),
                    embedding_backend=embedding,
                ),
                dataset,
            )
        )
        rows.append(
            await measured(
                "hierarchical",
                Reranker(
                    strategy=RerankPipeline(
                        [
                            BM25Filter(limit=6),
                            EmbeddingReranker(EmbeddingSimilarity(embedding), limit=5),
                        ]
                    ),
                    embedding_backend=embedding,
                ),
                dataset,
            )
        )
    if args.include_jev and os.getenv("TYPESAFE_API_KEY"):
        for mode in ("pointwise", "listwise"):
            rows.append(
                await measured_model(
                    f"jev_{mode}",
                    JevBackend(
                        model=args.jev_model,
                        input_price_per_million_usd=args.jev_input_price_per_million_usd,
                        output_price_per_million_usd=args.jev_output_price_per_million_usd,
                    ),
                    dataset,
                    strategy=mode,
                )
            )
    else:
        reason = "pass --include-jev with TYPESAFE_API_KEY to permit billable API calls"
        rows.extend((unmeasured("jev_pointwise", reason), unmeasured("jev_listwise", reason)))

    if args.include_laya_local:
        rows.append(
            await measured_model(
                "laya_local",
                LayaBackend(
                    model=args.laya_model,
                    device=args.laya_device,
                    max_batch_size=args.laya_batch_size,
                ),
                dataset,
            )
        )
    else:
        rows.append(
            unmeasured("laya_local", "pass --include-laya-local and install the laya extra")
        )

    if args.include_laya_http:
        rows.append(
            await measured_model(
                "laya_http",
                LayaHTTPBackend(
                    context_policy="allow_provider_truncation",
                    endpoint=args.laya_endpoint,
                    api_key=args.laya_http_api_key,
                    model=args.laya_model,
                ),
                dataset,
            )
        )
    else:
        rows.append(
            unmeasured("laya_http", "pass --include-laya-http with a running compatible service")
        )

    for name, enabled, make_backend, reason in (
        (
            "hybrid_bm25_embedding_jev",
            args.include_jev and bool(os.getenv("TYPESAFE_API_KEY")),
            lambda: JevBackend(
                model=args.jev_model,
                input_price_per_million_usd=args.jev_input_price_per_million_usd,
                output_price_per_million_usd=args.jev_output_price_per_million_usd,
            ),
            "requires an embedding model, --include-jev, and TYPESAFE_API_KEY",
        ),
        (
            "hybrid_bm25_embedding_laya_local",
            args.include_laya_local,
            lambda: LayaBackend(
                model=args.laya_model,
                device=args.laya_device,
                max_batch_size=args.laya_batch_size,
            ),
            "requires an embedding model and --include-laya-local",
        ),
        (
            "hybrid_bm25_embedding_laya_http",
            args.include_laya_http,
            lambda: LayaHTTPBackend(
                context_policy="allow_provider_truncation",
                endpoint=args.laya_endpoint,
                api_key=args.laya_http_api_key,
                model=args.laya_model,
            ),
            "requires an embedding model and --include-laya-http",
        ),
    ):
        if embedding is None or not enabled:
            rows.append(unmeasured(name, reason))
        else:
            rows.append(
                await measured_model(
                    name,
                    make_backend(),
                    dataset,
                    strategy=hybrid_pipeline(embedding),
                    embedding=embedding,
                )
            )
    if embedding is not None:
        await embedding.aclose()

    report = {
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "dataset_sha256": digest,
        "candidate_order_seed": ORDER_SEED,
        "python": platform.python_version(),
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "jev_model": args.jev_model if args.include_jev else None,
        "jev_input_price_per_million_usd": (
            str(args.jev_input_price_per_million_usd)
            if args.jev_input_price_per_million_usd is not None
            else None
        ),
        "jev_output_price_per_million_usd": (
            str(args.jev_output_price_per_million_usd)
            if args.jev_output_price_per_million_usd is not None
            else None
        ),
        "laya_model": args.laya_model
        if args.include_laya_local or args.include_laya_http
        else None,
        "laya_endpoint": args.laya_endpoint if args.include_laya_http else None,
        "metrics": [metric.name for metric in METRICS],
        "rows": rows,
        "note": "Synthetic smoke data; quality figures do not establish real-world effectiveness.",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    asyncio.run(main())
