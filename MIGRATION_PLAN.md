# TypedRank migration plan

Migration plan for the TypedRank port. The Step 2 baseline is implemented; subsequent phases remain roadmap items.

## 1. Source boundary and chosen destination

Source: [kashyaprparmar/jev-rankkit at e027d12e210f25265c8239ac4237d8d6a779427c](https://github.com/kashyaprparmar/jev-rankkit/tree/e027d12e210f25265c8239ac4237d8d6a779427c). The Step 2 port was made into a separate local repository. The original source checkout remains unchanged and keeps its original history and remote.

Later create a separate `typedrank` repository with `src/typedrank`, distribution name `typedrank`, import name `typedrank`, and its own history/remote. Copy these two planning documents into that repository when Step 2 is authorized. This step creates neither that repository nor a Python scaffold. Planning files are saved in the sibling `typedrank-planning` directory.

Laya reference: [NandhaKishorM/laya at 010bacef009c855ccba814b51f7c8e1d38ab5e3f](https://github.com/NandhaKishorM/laya/tree/010bacef009c855ccba814b51f7c8e1d38ab5e3f), published version 0.3.7. Static comparison verified all nine package Python modules against the published wheel. Consume it as an optional dependency or external server; do not transplant its implementation, models, tests, assets, or research notebooks.

Terminology in this plan:

- **Copy:** preserve behavior; update import/distribution references as necessary.
- **Rename:** preserve implementation while correcting generic terminology or package identity.
- **Refactor:** targeted behavior/interface change with an explicit acceptance test.
- **Delete/omit:** exclude from the new repository; never delete from the original.
- **Add:** new adapter or contract required by the architecture, in its later phase.

## 2. Complete source migration matrix

Source and destination paths in this subsection are relative to `src/jev_rankkit/` and `src/typedrank/`, respectively. A row containing several explicit files applies to each file. All 37 source package files are covered, including package markers and `py.typed`.

| Source file / class | TypedRank destination | Action | Reason / implementation boundary |
| --- | --- | --- | --- |
| `__init__.py` / public facade exports | `__init__.py` | Rename | Preserve small facade; replace generic package branding; export the new backend-neutral names where appropriate. |
| `api.py` / `Reranker`, `AutoReranker` | `api.py` | Refactor | Retain projection, validation, metrics, result assembly, and scoped lifecycle. Replace model-string provider construction with explicit `backend=`; separate route resolution from strategy selection. |
| `auto.py` / `AutoStrategy`, `AutoDecision` | `auto.py` | Refactor | Preserve deterministic planning/cascade logic; use `ModelReranker`; replace remote-only assumptions with locality/readiness and standardized estimates. Make thresholds configurable. |
| `candidates.py` / `CandidateView`, `CandidateAdapter`, `prepare_candidates` | `candidates.py` | Copy | Retain object identity, eligibility ordering, immutable metadata, occurrence IDs, display IDs, duplicate occurrences, and explicit text projection. |
| `config.py` / `Budget`, `RetryConfig`, `CacheConfig`, `RerankerConfig` | `config.py` | Refactor | Preserve immutable validation and budgets; add explicit auto thresholds and context policy boundaries; distinguish backend retries from strategy fallback. |
| `context.py` / `RerankContext`, `QualityMode` | `context.py` | Refactor | Preserve request/tenant/auth/time fields. Add language, network policy, and backend override; retire offline as a quality level after the baseline port. |
| `errors.py` / error hierarchy | `errors.py` | Copy, then extend | Preserve typed failures and safe details. Add specific readiness/closed-runtime errors only if dispatch needs them; provider exceptions stay behind adapters. |
| `prompts.py` / `RerankPrompt`, `PromptExample` | `prompts.py` | Copy | Preserve user prompt fields/fingerprints. Add backend render profiles in the codec, not provider branches in generic prompts. |
| `types.py` / results, plans, statistics, usage | `types.py` | Refactor | Preserve immutable typed results and coverage. Add actual backend/checkpoint provenance, confidence/raw kind, context validation, and canonical `statistics` with `stats` alias. |
| `pipeline.py` / `JevReranker` | `pipeline.py` / `ModelReranker` | Rename, then refactor | Stage already calls generic services. Preserve ranking implementation; bind model-stage routing and whole-stage fallback without changing pruning behavior. |
| `pipeline.py` / filters, embeddings, diversity, `RerankPipeline`, checkpoint error | Same names in `pipeline.py` | Copy | Retain survivor passing, full-order pruning guard, previous-stage checkpoint, and stage counts. Add selected backend details through the shared execution path. |
| `cache.py` / protocol, record, `MemoryCache` | `cache.py` | Copy | Preserve bounded TTL/LRU and namespace behavior. Introduce new cache schema/namespace at key construction rather than rewriting storage. |
| `observability.py` / dispatcher, events | `observability.py` | Rename, extend | Rename internal task label; preserve bounded observer queue. Add safe backend/routing attributes without raw candidate data. |
| `convenience.py` / document, entity, tool, memory helpers | `convenience.py` | Copy | Helpers project arbitrary objects independently of any backend. |
| `integrations.py` / framework/vector-store adapters | `integrations.py` | Copy | Preserve duck-typed adapters without importing third-party frameworks. |
| `py.typed` | `py.typed` | Copy | Preserve PEP 561 typing marker in the built wheel. |
| `backends/__init__.py` | `backends/__init__.py` | Refactor | Export contracts and optional adapter classes without importing their heavy runtime dependencies. |
| `backends/base.py` / `ModelBackend`, `EmbeddingBackend`, request/response records | `backends/base.py` | Refactor | Preserve structural protocols and existing fields; add defaulted capabilities/provenance and an optional typed preparation/estimate extension. |
| `backends/jev.py` / `JevBackend` | `backends/jev.py` / `JevBackend` | Refactor | Keep TypeSafe credentials, endpoint, model IDs, pricing, and provider limits. Delegate shared codec/HTTP implementation. |
| `backends/jev.py` / `JevHttpResponse`, `JevTransport`, `_HttpxTransport` | `backends/_http.py` / neutral response/transport protocols and `SystemOneHTTPTransport` | Rename, extract | Response/transport shapes are HTTP concepts, not Jev concepts. Keep injection, byte limit, pooling, Retry-After, and strict parsing. |
| `backends/jev.py` / `_payload`, `_parse_response`, `_unique_object`, token/usage parsing | `backends/systemone.py` and `_http.py` | Refactor | Extract reusable request/answer codec and raw decoder once; parameterize render profile and provider-specific error policy. |
| `backends/sentence_transformers.py` / `SentenceTransformerBackend` | Same path/class | Copy, then targeted refactor | Preserve lazy loading and query/document encoders. Correct cancellation/close lifetime with shared tracked-worker support. |
| `backends/fake.py` / `FakeModelBackend` | `backends/fake.py` / `FakeBackend` | Rename, targeted refactor | Source class is actually `FakeModelBackend`. Preserve deterministic test hooks; fix fixture-sensitive cache identity and declare ID-sensitive scoring. Temporary alias is acceptable during the port. |
| `strategies/__init__.py` | Same path | Copy | Preserve strategy exports and update imports. |
| `strategies/base.py` / `RankingStrategy`, `EvaluationServices` | Same path/classes | Copy, then extend | Preserve service-based separation; add stage-binding hooks only where the router requires them. No provider imports. |
| `strategies/pointwise.py` / `PointwiseStrategy` | Same path/class | Copy | Preserve independent scoring and stable tie ordering. Do not conflate future physical batching with listwise semantics. |
| `strategies/listwise.py` / `ListwiseStrategy` | Same path/class | Copy | Preserve one bounded group and rejection of independently scored chunks. Apply selected child's context/capability checks. |
| `strategies/metrics.py` / metric and lexical strategies | Same path/classes | Copy | Keep deterministic normalized aggregation and sorting. |
| `metrics/__init__.py` | Same path | Rename exports | Export `ModelRelevance`; update aliases consistently. |
| `metrics/base.py` / `Metric`, `BatchMetric`, `WeightedMetrics`, utility validation | Same path/classes | Copy | Retain protocol and finite/nonnegative weight validation; no backend-specific primitives. |
| `metrics/builtin.py` / `LLMRelevance` | Same file / `ModelRelevance` | Rename, refactor | Model relevance is generic; stop claiming every model returns Noul. Keep prompt/identity in cache keys and executor-mediated model budgeting. |
| `metrics/builtin.py` / lexical, BM25, embedding, recency, numeric, callable metrics | Same file/classes | Copy | Preserve working formulas, metadata requirements, asymmetric embeddings, callback cache dependencies, and explicit normalization. |
| `selection/__init__.py` | Same path | Copy | Preserve public selection exports. |
| `selection/fusion.py` / reciprocal-rank fusion | Same path/functions | Copy | Preserve ID-based fusion, validated weights, and deterministic ordering; do not reinterpret fusion values as probabilities. |
| `selection/diversity.py` / MMR helpers | Same path/functions | Copy | Preserve relevance versus selection scores and deterministic tie behavior. |
| `evaluation/__init__.py` | Same path | Copy | Preserve evaluation exports. |
| `evaluation/metrics.py` / ranking evaluation metrics | Same path/classes | Copy | Preserve NDCG, recall, precision, MRR, MAP, hit/success metrics and judgment semantics. |
| `evaluation/runner.py` / dataset, runner, report | Same path/classes | Copy, extend report | Retain fixed-pool judgments, position alignment, unknown usage, and latency/throughput measurements; add backend/model/device provenance. |
| `_runtime/__init__.py` | Same path | Copy | Preserve package boundary. |
| `_runtime/cache_keys.py` / canonical hashing | Same path | Copy | Preserve hashing; add new input dimensions in the executor and schema version. |
| `_runtime/executor.py` / `ExecutionServices`, `BudgetLedger` | Same path/classes | Refactor selectively | Preserve reservations, reconciliation, safe gathering, validation, and metric scheduling. Remove Noul/token-price assumptions; bind selected child; route before cache; preserve one ledger through fallback. |

### New files, not ports

| New destination | Purpose | First implementation phase |
| --- | --- | --- |
| `backends/systemone.py` | Shared typed records, codecs, rendering profiles, generic HTTP backend | Step 3 |
| `backends/_http.py` | Shared optional HTTP transport and raw JSON/error machinery | Step 3 |
| `backends/laya.py` | Local adapter, stage checkpoint preparation, compact rendering and guarded context validation | Step 4 |
| `backends/laya_http.py` | Laya HTTP configuration and deployment capability/identity policy | Step 4 |
| `backends/routing.py` | Explicit stage resolver, deterministic policies and fallback decision records | Step 5 |
| `_runtime/workers.py` | Bounded tracked synchronous jobs and safe shutdown for local model/embedding calls | Step 4; embedding adoption by Step 6 |

## 3. Exact Jev-as-generic coupling audit

Line references below are to the pinned original source, not to future TypedRank files. The inventory separates incorrect generic assumptions from provider-specific names that should remain.

### Direct source couplings to change

1. **`pipeline.py:136,140`: class/name `JevReranker`.** It uses `EvaluationServices` and pointwise/listwise strategies without any Jev requirement. Rename class and emitted stage label to `ModelReranker`.
2. **`auto.py:19,264`: imports/construction of that stage.** Rename both. Its prospective stage at approximately line 267 already says `ModelReranker`, so the current prospective and executed names disagree.
3. **`api.py:13,71-117`: `JevBackend` import, overloaded positional `model`, `_resolve_backend`, and `typesafe:` construction.** Move provider selection out of the generic facade; accept backend objects. `_owns_backend` and `_model_spec` at lines 84-85 currently infer resource ownership from strings and must change alongside that API.
4. **`api.py:730-763`: sync reconstruction and `AutoReranker` constructor.** Sync reconstructs via `_resolve_backend`, tying lifecycle to Jev strings. Preserve borrowed-resource guards, switch AutoReranker to explicit backend objects, and defer an explicit sync factory rather than introducing hidden backend reconstruction.
5. **`auto.py:68-100`: introspection of `.retry.max_attempts`, zero cost interpreted as zero model budget, and offline/fast unconditional lexical routing.** Retry ceilings belong in standardized estimates. Network policy must allow provisioned local inference, and known-zero external charge must not prohibit local models. “No remote model is allowed” is not an accurate explanation for every no-backend route.
6. **`auto.py:110-189,240-285`: context, token and call estimation.** Separate maximum sequence length, total processed tokens, retry attempts, and actual selected child. Do not plan against a router's capability union or call a Laya tokenizer synchronously on the loop. Thresholds such as 10/100 and retention/embedding factors become config with unchanged baseline defaults until measured.
7. **`_runtime/executor.py:217-227`: `_estimated_cost`.** Generic code reads `input_price_per_million_usd` and `output_price_per_million_usd`; move that tariff calculation into Jev/HTTP adapters. Use typed estimates/bounds for local zero external charge, per-request pricing, and unknown prices.
8. **`_runtime/executor.py:336-339`: `llm_relevance` and hard-coded `noul_probability`.** Use `model_relevance` and backend-provided raw kind/primitive/normalizer; a custom backend need not implement Noul.
9. **`_runtime/executor.py:454,676,691-704`: metric name, `LLMRelevance` special case and validation messages.** Rename to `ModelRelevance` consistently. Retain the invariant that model metrics use the same selected execution service and budget ledger, rather than calling an unrelated backend outside runtime accounting.
10. **`metrics/builtin.py:294-313` and `metrics/__init__.py:7,17`: `LLMRelevance`, key `llm_relevance`, version `noul-relevance-v1`.** Rename the generic metric and version; keep Noul labeling only on actual typed Noul outputs. Update facade metric string resolution at `api.py:206-210`.
11. **`backends/jev.py:38-103`: provider-named response/transport protocols and generic HTTP mechanics.** Extract neutral internal abstractions. TypeSafe-specific error text/status interpretation remains in a provider policy, not the shared client.
12. **`backends/jev.py:159` onward: payload and answer/usage parsing.** These contain reusable System One logic mixed with Jev context estimates, reasoning restrictions, authentication, and price accounting. Separate them carefully; do not copy this whole file into a Laya class.
13. **Package labels, not semantic dependencies:** `__init__.py:1`; task names in `observability.py:53` and `_runtime/executor.py:394`; missing-extra messages in `backends/jev.py:62` and `backends/sentence_transformers.py:51`. Update to TypedRank naming without changing provider identity.
14. **`backends/__init__.py:11,20-21`: exports.** `JevBackend` remains a valid concrete provider export. Replace `JevTransport` with the shared transport seam or a transitional provider alias; imports must remain dependency-light.

No `isinstance(backend, JevBackend)` branch was found in the inspected generic executor/strategy code. Preserve that good property. `backends/base.py` is already provider-independent; extend it rather than replacing it with provider inheritance.

### Dependent references to migrate deliberately

- `tests/test_pipeline_auto.py:16,66,147,160`, `examples/07_hierarchical_reranking.py:7,11`, and `README.md:81` reference `JevReranker` as a generic stage; rename all.
- `tests/test_foundation.py` tests sync reconstruction through the current string-resolution seam; replace that test only when replacing the API, retaining ownership and active-loop assertions.
- `tests/test_pipeline_auto.py:97` bundles offline mode and zero-call-budget expectations. Split into network-deny/local-ready, network-deny/remote-rejected, and zero-model-call cases; do not weaken the latter.
- `docs/usage.md:127,163`, `README.md:45`, and `benchmarks/compare.py:181` demonstrate the facade's `typesafe:` constructor. Use `backend=JevBackend(...)` while keeping Jev-specific examples explicitly labeled.
- `benchmarks/data/synthetic_v1.json` has a package-branded dataset name. Preserve data and provenance; if renaming the dataset, version and record the mapping so reports remain comparable.
- Every Python `jev_rankkit` import, build wheel path, mypy/coverage target, CI command, and installation URL changes to TypedRank. References to the origin of copied code remain correct and should stay.
- Jev-specific benchmark flags/model fields and opt-in live tests remain provider-specific. Rename project-prefixed environment controls to `TYPEDRANK_JEV_LIVE` / `TYPEDRANK_JEV_MODEL` with documented migration; never rename the provider credential `TYPESAFE_API_KEY` to a generic token.

### Correct provider details to retain

Keep `JevBackend`, TypeSafe endpoint defaults, bearer-key requirements, `TYPESAFE_API_KEY`, Jev model names and revision validation, configured Jev prices, Jev capability limits, and provider test fixtures. `noul` itself is a shared typed primitive, not obsolete Jev branding. Renaming every occurrence of “Jev” or “Noul” would corrupt the integration rather than improve neutrality.

## 4. Tests, documentation, examples, and repository metadata matrix

| Source asset | Destination / action | Reason |
| --- | --- | --- |
| `tests/test_foundation.py` | Copy, then adapt explicit backend/ownership API tests | Preserve arbitrary objects, duplicate occurrences, no-op, eligibility, sync and observer regressions. |
| `tests/test_strategies.py` | Copy | Preserve pointwise/listwise contracts, stable ties, ID/score validation, and caching semantics. |
| `tests/test_metrics_prompts.py` | Copy, rename model metric references | Preserve prompt fingerprints, metadata, callable metric dependencies, embedding behavior, and weights. |
| `tests/test_pipeline_auto.py` | Copy, rename stage; refactor routing assertions later | Preserve fusion, MMR, pruning, checkpoints, budget/usage tests; explicitly distinguish changed auto semantics. |
| `tests/test_concurrency_review.py` | Copy, then add local worker cases | Existing coroutine cancellation tests do not prove running thread safety. Preserve them and add worker-lifetime checks. |
| `tests/test_jev_backend.py` | Preserve provider tests; factor shared assertions into codec/HTTP tests | Keep Jev wire behavior and status mapping; share neutral validation without losing provider-specific coverage. |
| `tests/test_jev_live.py` | Copy, update imports/opt-in controls | Preserve explicitly enabled billable provider contract tests; no default network calls. |
| `tests/test_evaluation.py` | Copy, extend statistics provenance assertions | Preserve fixed-pool and judgment alignment semantics. |
| `tests/test_integrations.py` | Copy | Preserve dependency-free framework/vector-store projections. |
| `examples/01_basic_reranking.py` through `examples/15_framework_adapters.py` | Copy all fifteen; update imports and model-stage/API names | Existing coverage of documents, entities, metrics, prompts, hybrid/cascades, tools, SQL, evaluation, budgets, objects, memory, graph and integrations is reusable. |
| `benchmarks/python_overhead.py` | Copy, rename package imports | Preserve measurement of Python overhead separately from model performance. |
| `benchmarks/compare.py`, `benchmarks/data/synthetic_v1.json` | Copy; later add explicit Laya modes | Preserve reproducible original data and baseline rows; never relabel old measurements as new backend results. |
| `docs/usage.md`, `docs/integrations.md`, `docs/evaluation.md` | Refactor for new API, retain generic material | Add backend ownership and provider-separated setup; keep useful operational guidance. |
| `docs/benchmarks.md` | Refactor | Keep methodology/caveats; add cold/warm local model and route provenance later. |
| `docs/comparison.md` | Rewrite selectively | Existing Jev-package comparison is not evidence for multi-backend TypedRank quality. |
| `docs/research.md` | Omit as active product spec; link origin where useful | Large source research history is not a current TypedRank contract; migrate only still-supported rationale. |
| `docs/assets/jev-rankkit-ranking-flow.png` | Omit | Provider-branded artwork should not define the new architecture. A new neutral diagram can be added later. |
| `README.md` | New concise TypedRank README | Step 2 documents the working baseline; later add only features actually implemented. Do not carry the old publication claim. |
| `IMPLEMENTATION_PLAN.md`, `FINAL_AUDIT.md`, `FINAL_AUDIT_RESOLUTION.md` | Omit from new project root | Historical source artifacts; they must not imply TypedRank has passed an audit. Preserve useful regression tests instead. |
| `RELEASE_CHECKLIST.md`, `CONTRIBUTING.md` | Refactor | Preserve useful build/review instructions; replace package/release names and add backend certification gates. |
| `CHANGELOG.md` | New TypedRank history with source-provenance note | Do not claim Jev Rankkit releases were TypedRank releases. |
| `LICENSE` | Copy MIT text/copyright; add TypedRank attribution | Preserve source license notices; do not erase Jev Rankkit attribution. |
| `pyproject.toml` | Refactor package name, URLs, wheel path, typing targets, extras | Keep lightweight base; actual `dev` extra and separate HTTP extra; defer Laya runtime implementation to Step 4. |
| `uv.lock` | Regenerate in the new repository | New distribution metadata and optional dependencies require a new resolver result. |
| `.github/workflows/ci.yml` | Copy, update package targets; add install/import lanes | Preserve Windows/Linux and Python compatibility intent; add no-heavy-import and optional dependency checks. |
| `.gitignore` | Copy and adjust only for real new outputs | Do not transfer caches, environments, generated reports, wheels, coverage data, or model checkpoints. |
| Original `.git`, credentials, `.venv`, `dist`, caches | Omit | Independent repository and reproducible installation; no copying secrets or build state. |

## 5. Implementation sequence for later steps

The phase numbers match the supplied plan. Step 2 is complete in this repository. Steps 3 onward remain future work, with a reviewable checkpoint for each phase.

### Step 2 — completed independent repository and mechanical port

1. Create the separate target checkout/repository and record the pinned source SHA. Copy the MIT notice and these planning documents. Check package/repository name availability before publishing; do not assume this planning decision reserves names.
2. Copy the package and retained tests/examples using the matrix. Update distribution/import/build targets, `py.typed`, CI paths, and package branding. Rename `JevReranker` to `ModelReranker` immediately because that changes terminology, not provider behavior.
3. Preserve current Jev and generic behavior for a passing baseline. Keep the legacy facade model-string seam internally at this checkpoint if needed; its removal belongs to Step 3. Keep temporary aliases only to isolate API churn, and mark the README as a port baseline.
4. Run the original non-live regression suite in the new repository, Ruff lint/format, strict type checking, build, and installed-wheel smoke checks. No Laya implementation, model downloads, routing fallback, or unmeasured performance changes.

Completed: separate TypedRank directory and Git metadata, pinned-source code/tests/examples/docs copied per matrix, wheel/package paths updated, generic stage renamed, Jev behavior preserved, original checkout clean, and requested non-live checks passing. Build artifacts and exact check results are reported by the implementing task.

### Step 3 — neutral contracts and System One extraction

1. Establish `backend=` and explicit ownership; update generic examples/tests. Keep custom backend protocol conformance simple. Reject an explicitly requested model strategy without a backend.
2. Introduce defaulted score provenance/locality and the optional typed estimate/preparation extension. Rename `ModelRelevance` and generic transport types. Keep aliases bounded to this migration; do not add a legacy `jev_rankkit` package.
3. Extract the codec and shared HTTP implementation. Retain `jev-v1` payload fixtures byte/structure-equivalent where ordering is meaningful. Do not change prompt rendering and extraction at the same time for Jev.
4. Move Jev pricing/limits/error policy into its adapter. Keep raw JSON duplicate-key, response-size, score/type, ID-coverage, retry/cancellation, and auth-redaction regressions.
5. Add reusable backend contract tests; use a minimal custom backend with utility scores and no Noul semantics to prove provider independence. Add import-boundary checks.

Exit: generic engine has no TypeSafe construction, endpoint, credential, tariff, or Noul assumptions. Jev remains functional through the extracted shared path. No Laya backend is claimed yet.

### Step 4 — local and HTTP Laya

1. Add lazy local adapter and mock injection. Certify the published 0.3.7 API with an optional dependency lane; unit tests use injected doubles without installing/importing Torch.
2. Add bounded tracked-worker lifetime, asynchronous initialization/preload, borrowed/owned runtime handling, close-after-drain, and conservative single inference concurrency. Test against long-running blocking doubles and cancellation.
3. Implement compact `laya-v1` rendering and exact/conservative tokenizer fit validation in the isolated compatibility helper. Measure head, criterion, and state budgets separately. Unknown/unvalidated layouts fail closed by default.
4. Resolve/pin a stage checkpoint from pool text or language override. Forward Router parameters according to actual signatures; use `models`/Agent for explicit custom paths. Preserve useful routing metadata and actual device changes.
5. Add Laya HTTP via the shared transport/codec. Endpoint is explicit, authentication optional, model override validated. Do not send unsupported stock-server language/task controls. Distinguish HTTP 422 from Jev's existing interpretation.
6. Make HTTP context limitations explicit: strict mode requires an injected verified validator/deployment contract; unverified stock-server use requires the documented approximate opt-in. Do not silently enable listwise on it.
7. Add `test_laya_backend.py` and `test_laya_http_backend.py`, then optional `laya_live` tests with pre-provisioned models. No use of choice shortlisting for ordinary binary relevance.

Exit: both Laya modes pass neutral contract tests, no default imports/downloads, cancellation does not free a live worker slot, context loss cannot masquerade as fully evaluated input, and checkpoint/device metadata is honest.

### Step 5 — inspectable multi-backend routing

1. Add stage-level BackendRouter, not a per-batch fallback wrapper. Its deterministic policy consumes requirements and returns an immutable selected backend/reason.
2. Split quality, locality/network constraints, and inference budgets. No-network local inference requires provisioned local artifacts or ready borrowed runtimes; no implicit Hub downloads.
3. Bind the actual backend/checkpoint before cache lookup. Implement whole-stage fallback with one ledger/deadline, strict default error classes, explicit override behavior, and no mixing of primary/fallback candidate scores.
4. Update AutoReranker composition. Make numeric thresholds configurable and retain baseline heuristics pending domain evidence. A 500 -> 100 BM25 -> 30 embeddings -> 10 local-model cascade is a configurable example, not a measured default or guaranteed winner.
5. Record prospective versus executed backend/strategy decisions and grouped attempts in response plans/statistics. Preserve changed semantics when explicitly configured strategy fallback occurs.

Exit: explicit and automatic strategy/backend selection are independent and explainable; simultaneous requests do not share mutable last-route state; failed or unavailable backends do not contaminate cached or final scores.

### Step 6 — focused reliability and cache certification

1. Finish standardized token/cost/attempt accounting across retries/fallbacks; distinguish longest-sequence limits from processed/billable token totals and strict bounds from estimates.
2. Certify cache partitioning by backend, endpoint, immutable checkpoint/calibration identity, render profile, language, primitive and tenant. Fix ID-sensitive FakeBackend collisions. Disable persistent caching where immutability cannot be established.
3. Apply tracked-worker shutdown to SentenceTransformerBackend; its current `asyncio.Lock` around `to_thread` does not retain the lock after cancellation while the thread continues.
4. Verify multiple simultaneous reranks, queue deadlines, owned/borrowed cleanup, timeout accounting, unknown usage, cold model loads, and device changes. Preserve safe observer behavior.
5. Run full checks and optional backend certification before proceeding to the later documentation/benchmark/release work in the supplied plan. Those later tasks remain outside the Step 2 baseline.

## 6. Acceptance tests that protect real behavior

Retain the current regression tests instead of replacing them with implementation-mirroring mocks. Add tests at the following observable boundaries:

- **Objects and IDs:** identical objects at different positions survive independently; display IDs remain unchanged; eligible candidates are projected before backend use; top-k zero does not project or invoke a model. A backend that returns missing/duplicate/unknown IDs fails; malformed local/HTTP values include booleans, strings, NaN and infinities.
- **Semantics:** independent pointwise input cannot expose other candidates; listwise includes the entire ordered group and refuses unrelated chunk merging; raw utility from a custom backend is never called Noul; confidence cannot substitute for relevance.
- **Shared HTTP:** neutral codec fixtures are exercised through Jev and Laya profiles. Verify raw duplicate keys, response byte ceiling, optional versus required auth, known and unknown model overrides, 422 classification, bounded retry/backoff, total deadline, cancellation, and client ownership.
- **Local lifecycle:** initialization and inference let an event-loop ticker progress; concurrent cold starts produce one runtime; a cancelled running call retains its permit; no second inference/unload runs while that call remains active; late exceptions are retrieved; close drains owned work and does not unload a borrowed runtime.
- **Model residency/device:** lazy versus named/all preload and effective residency cap are explicit; route switches do not create duplicate instances; CPU fallback is surfaced; shared-runtime concurrency >1 is rejected for the certified Laya version.
- **Language and context:** a non-English candidate with an English query is routed using the candidate pool; a single stage pins one checkpoint; explicit language/model override is inspectable. Boundary cases exercise instructions, criterion descriptions, special tokens, and state independently; a too-long prompt is rejected rather than dropped. Strict HTTP requests without validation support fail before dispatch; opt-in unverified truncation is labeled approximate.
- **Routing/fallback:** no-network mode rejects remote/download work while allowing ready local inference; zero model calls prohibits local inference too; zero API-cost budget allows a known-zero-charge local backend. Unsupported strategy cannot slip through a capability union. Primary partial success followed by failure reruns all survivors on fallback or raises; it never mixes score sources.
- **Cache:** different endpoints, checkpoints, render profiles, languages, tenants, primitives, calibration artifacts, and fake fixtures cannot collide. A fallback result cannot populate the primary key; cancelled/partial jobs do not write complete entries; a cache hit adds no new usage/cost. Routing changes cannot be hidden by a stale aggregate router identity.
- **Budgets and statistics:** account every started attempt across the whole rerank; no fresh ledger on fallback; settle reservations for attempts that did not start; unknown token/cost usage stays unknown. Local external API cost can be zero while inference still consumes latency/tokens/calls. Prospective plans and actual attempts remain distinguishable.
- **Packaging:** in a clean base installation, import public APIs with heavy modules blocked and verify none load; core fake/lexical functionality works. Check each extra independently and `all` resolution separately. Built wheel contains `typedrank/py.typed` and no `jev_rankkit` package, model weights, secrets, caches, or source build outputs.

Routine commands for the future target repository are `pytest`, `ruff check .`, `ruff format --check .`, `mypy src/typedrank`, and `python -m build`, using its locked dev environment. Run opt-in live tests separately and record their model/device versions. Do not run billable or model-download tests as a default CI side effect.

## 7. Known integration limits and release blockers

These findings come from inspected source, not speculative provider features:

1. **No stable Laya public preflight API.** A narrow version-tested tokenizer/configuration guard is required. Keep it isolated and fail closed on incompatible layouts. Do not claim an arbitrary character cutoff proves no truncation.
2. **Stock HTTP cannot prove full-context evaluation.** Require a validator/deployment contract for strict mode, or make provider truncation an explicit visibly approximate choice. The client cannot fix or observe hidden upstream truncation retrospectively.
3. **Stock server synchronously runs inference in its async handler.** Document deployment behavior and measure end-to-end concurrency. Do not copy/patch the Laya server as part of TypedRank; propose upstream improvements separately if needed.
4. **Agent revision pinning is absent from the constructor.** Use immutable local artifacts with verified identity for stable caching; otherwise disable persistent model caching. A human-readable repository path or `laya-rl-agent` is insufficient.
5. **Automatic checkpoint routing is not calibration evidence.** Default to one checkpoint per model stage. Any later mixed-checkpoint ranking needs explicit quality/calibration evaluation or rank fusion.
6. **Confidence, multilingual coverage, and latency are not guarantees.** Preserve upstream outputs and honest metadata; benchmark reranking and candidate recall on actual target domains before choosing thresholds or marketing claims.
7. **Runtime compatibility must be tested.** Laya and embedding extras have substantial transitive dependencies. The verified 0.3.7 wheel/source match does not certify every Torch/Transformers version or GPU platform.

This baseline implements Step 2 only; the subsequent backend refactor and Laya/routing work remain roadmap items.

## 8. Original Step 1 design review

- [x] Original source and Laya implementation inspected at immutable SHAs.
- [x] Published Laya release compared to inspected source without importing the model runtime.
- [x] Architecture chooses explicit APIs, shared transport/codec, local lifecycle, score semantics, routing, caching, dependency boundaries, and ownership.
- [x] All source package files and retained repository assets have a migration disposition.
- [x] Generic Jev couplings are distinguished from valid provider-specific code.
- [x] Design critique completed and incorporated in architecture section 11.
- [x] Implementation acceptance gates and unresolved upstream limitations documented.
- [x] Only the two requested Markdown documents created as deliverables; package/repository implementation deferred.

The source review above was completed before the port. Current baseline verification results are recorded with the implementation report; later-phase acceptance checks remain future work.
