# TypedRank final adversarial audit

Date: 2026-09-23  
Decision: **Not ready to certify the multi-backend release candidate.**  
Findings: **1 BLOCKER, 6 HIGH, 9 MEDIUM, 1 LOW.**

## Scope and evidence

Reviewed the current working tree, including the uncommitted Step 7/8 changes, on top of commit `95ca37c9745ac1bf1894b802b4444023bb713515`. Compared `TYPEDRANK_ARCHITECTURE.md` and `MIGRATION_PLAN.md` against the facade, backend contracts/adapters, executor, automatic planner, strategies, pipelines, candidate handling, scores, caches, budgets, statistics, evaluation, dependencies, packaging, tests, examples, documentation, and benchmark harness.

This is a review, not a patch: no implementation or test changes were made. Short in-memory reproductions used injected doubles and no live API calls, model downloads, or GPU inference. The earlier release checks reported 162 passed, 3 skipped and 89% statement coverage; those results do not establish the missing lifecycle and tokenizer guarantees below. I did not rerun the entire suite during this audit.

The Laya compatibility review used the exact upstream commit cited by the architecture, not README performance claims. Relevant primary sources: [sequence construction](https://github.com/NandhaKishorM/laya/blob/010bacef009c855ccba814b51f7c8e1d38ab5e3f/laya/common.py#L45), [Router implementation](https://github.com/NandhaKishorM/laya/blob/010bacef009c855ccba814b51f7c8e1d38ab5e3f/laya/router.py), and [Agent implementation](https://github.com/NandhaKishorM/laya/blob/010bacef009c855ccba814b51f7c8e1d38ab5e3f/laya/agent.py). Compatibility with every version allowed by the published dependency range is unverified.

## BLOCKER

### B01 — Laya can silently score truncated candidate or prompt content

**Files/classes:** `src/typedrank/backends/systemone.py:183` / `relevance_payload`, especially line 220; `src/typedrank/backends/laya.py:101` and `:290` / both context estimators; `TYPEDRANK_ARCHITECTURE.md` §6 “Context integrity is a release gate.”

Pointwise Laya payloads put candidate text in the instruction object's `target`, while state contains the query. Both adapters gate requests using serialized character length divided by three against a single 512-token limit. Neither measures the instruction, options, special tokens, and state with the checkpoint tokenizer. There is no strict context policy or explicit provider-truncation opt-in.

The upstream sequence builder truncates the instruction head independently from the state, with a default head budget of 192 tokens that also accommodates options. Passing a whole-request character estimate therefore cannot establish that the candidate or rubric survives. Listwise state has a separate truncation risk. See the pinned [sequence builder](https://github.com/NandhaKishorM/laya/blob/010bacef009c855ccba814b51f7c8e1d38ab5e3f/laya/common.py#L45).

**Evidence:** a default pointwise request with a 100-word candidate received a context estimate of 346 and passed the 512 gate. That proves admission without the required component checks, not the precise number of tokens lost by a particular real tokenizer. Existing injected-router tests do not tokenize anything.

**Recommended fix:** implement the architecture's version-tested component validator and move pointwise candidate content into appropriate state. Fail closed when exact local validation is unavailable. Require a deployment validator for strict HTTP mode, or an explicit opt-in that marks results approximate and context validation unverified. Add real-tokenizer boundary tests before certifying Laya listwise support.

## HIGH

### H01 — Unsupported concurrent inference is allowed on one mutable Laya runtime

**Files/classes:** `src/typedrank/backends/laya.py:52`, `:135` / `LayaBackend.__post_init__`, `_worker`.

Any positive `max_inference_concurrency` is accepted and becomes the executor worker count. The lock protects Router construction only. A value above one permits overlapping predictions on the same runtime. This contradicts the architecture's explicit requirement to reject concurrency above one for the inspected Laya release.

The upstream Router lock protects lifecycle operations, while inference occurs outside it; Agent inference can move its model to CPU after a device error. Those operations cannot safely be assumed independent. See [Router.predict](https://github.com/NandhaKishorM/laya/blob/010bacef009c855ccba814b51f7c8e1d38ab5e3f/laya/router.py#L377) and [Agent.system_one](https://github.com/NandhaKishorM/laya/blob/010bacef009c855ccba814b51f7c8e1d38ab5e3f/laya/agent.py#L334).

**Recommended fix:** reject values above one for the certified shared runtime. Scale through separately provisioned processes until concurrent operation is supported and tested. Do not change global Torch thread counts implicitly; document application-level thread limits. No actual GPU race or allocator leak was exercised in this audit.

### H02 — Cancellation during local shutdown permanently skips model unloading

**Files/classes:** `src/typedrank/backends/laya.py:219` / `LayaBackend.aclose`.

Close sets `_closed=True` before awaiting executor shutdown and then unloading the Router. If cancelled during the shutdown wait, unloading is skipped. Every subsequent close returns immediately because the backend is already marked closed. The retained Router can retain loaded models.

**Reproduced:** start an 80 ms injected prediction; start close; cancel close after 10 ms; wait for prediction; close again. Result: `closed=True, unloaded=False, router_retained=True`.

**Recommended fix:** distinguish closing from fully closed and retain one cleanup task that drains work and unloads even when a caller stops waiting. Later close calls should await that same task. Reject queued/new submissions once closing starts; unload only resources whose ownership is explicitly assigned to the adapter.

### H03 — Cancelling an embedding call releases serialization while inference continues

**Files/classes:** `src/typedrank/backends/sentence_transformers.py:39`, `:45`, `:66` / `SentenceTransformerBackend._embed`, `_encode`, `aclose`.

The async lock surrounds `await asyncio.to_thread(...)`. Cancelling that await releases the lock, but the worker thread continues. A second embedding request then overlaps the first. Close also clears `_model` without draining outstanding workers, allowing a later call to allocate another model while an earlier one is still in use.

**Reproduced:** block the first injected encoder in a worker, cancel its await, start a second embed, then release both. Maximum active encoders was **2** despite the lock.

**Recommended fix:** keep a tracked worker job and retain serialization until the underlying job finishes; drain before clearing model references. A small shared worker-lifecycle helper for Laya and embeddings is sufficient. Add cancellation-followed-by-request and close-during-inference regressions.

### H04 — Laya HTTP auto mode mixes checkpoint score distributions within one stage

**Files/classes:** `src/typedrank/backends/laya.py:253`, `:282`, `:307` / `LayaHTTPBackend`; `src/typedrank/_runtime/executor.py:273`, `:415`, `:774`.

HTTP auto mode has no stage preparation/pinning hook. Every pointwise batch omits the model selector, and the executor does not require successive responses to resolve to the same checkpoint. Statistics retain only the last resolved model.

**Reproduced:** `LayaHTTPBackend(max_batch_size=1, model="auto")` with a transport returning English for one candidate and multilingual for the next produced one result with `status=ok` and only `resolved_model=multilingual`.

**Recommended fix:** require a pinned checkpoint for a multi-request HTTP stage, or use an explicit supported stage-resolution contract before scoring. Reject inconsistent resolved checkpoints. Do not merge scores across model calibrations unless the user explicitly selected a documented fusion/calibration policy.

### H05 — Local multilingual routing ignores candidate language

**Files/classes:** `src/typedrank/backends/laya.py:159` / `LayaBackend._route`; `src/typedrank/_runtime/executor.py:295` and `:330` / stage preparation.

The preparation hook receives only query and an optional language hint. The adapter calls Router with a query-only state and empty questions. It cannot detect a multilingual or non-English survivor pool when the query is English, then pins that potentially wrong checkpoint for the entire stage.

**Evidence:** direct inspection of the hook and call sites; no model-quality claim is inferred from mocks. The pinned upstream Router detects language from the supplied state. Architecture §6 explicitly requires the query plus projected survivor texts or an explicit language override.

**Recommended fix:** pass projected survivor content to the small preparation hook and apply the architecture's mixed-language rule. Preserve the reason and selected checkpoint. Do not add a second language-routing service in the generic engine.

### H06 — Network denial fails open for custom backends with unknown locality

**Files/classes:** `src/typedrank/backends/base.py:20` / `BackendCapabilities.execution_location`; `src/typedrank/backends/router.py:97` / `_eligible`; `src/typedrank/_runtime/executor.py:286`.

The default locality is `unknown`. Explicit execution and router eligibility reject only `remote` under `network_policy="deny"` or offline mode. Unknown backends are still invoked. AutoReranker instead excludes anything not declared local, creating inconsistent policy enforcement.

**Reproduced:** an otherwise ordinary unknown-locality backend was called once under `network_policy="deny"`.

**Recommended fix:** require an explicit local capability under deny/offline across all execution paths. Document that declarations are trusted and this policy is not an OS network sandbox. Test unknown locality as well as declared remote locality.

## MEDIUM

### M01 — Laya HTTP inherits Jev diagnostics and provider lifecycle

**Files/classes:** `src/typedrank/backends/laya.py:25`, `:233`, `:243`, `:309`, `:314`, `:323`; `src/typedrank/backends/jev.py:155`.

The shared codec extraction is useful, but `LayaHTTPBackend(JevBackend)` still reuses provider-specific initialization and request execution. HTTP status handling is overridden; timeout handling is not.

**Reproduced:** a mock Laya HTTP timeout raised `TypeSafe request timed out` with `details.stage="jev"`. A Laya reasoning-level configuration error also inherits Jev terminology.

**Recommended fix:** move reusable request/retry/client ownership code behind the shared System One transport/backend seam and keep small provider configurations. Preserve Jev credentials and status policy. This needs one shared implementation, not a hierarchy of generic service registries.

### M02 — Legacy Jev configuration remains in the generic facade and accounting path

**Files/classes:** `src/typedrank/backends/factory.py:11` / `backend_from_spec`; `src/typedrank/api.py:73`, `:784` / `Reranker`; `src/typedrank/_runtime/executor.py:226` / `_estimated_cost`.

The factory supports only `typesafe:`; string ownership drives sync reconstruction, while `rerank_sync` rejects every caller-supplied model backend. The generic executor still falls back to reading `input_price_per_million_usd` and `output_price_per_million_usd`. These are precisely the legacy assumptions the architecture said to retire.

**Recommended fix:** remove/deprecate the mechanical-port string facade as agreed in the architecture; retain the simple explicit backend API. Use one documented backend estimate hook rather than tariff introspection. Either clearly limit synchronous model support or add a small explicit backend factory/ownership option later; do not infer lifecycle from a provider string.

### M03 — Different FakeBackend configurations share cached scores

**Files/classes:** `src/typedrank/backends/fake.py:60` / `FakeBackend.cache_identity`; `src/typedrank/_runtime/executor.py:243`, `:486`.

Every FakeBackend declares itself stable with identity `fake:fake-v1`, irrespective of its score dictionary/callback. Two rankers sharing a cache can return each other's judgments.

**Reproduced:** a backend returning 0.2 populated the cache; a different backend returning 0.9 returned **0.2** with **zero** calls to the second backend.

**Recommended fix:** give fake configurations distinct identities or disable persistent caching unless an explicit immutable configuration identity is supplied. The issue affects tests and integrations using the public fake backend; it is not evidence that Jev and Laya currently share identities.

### M04 — Partial-result coverage is derived after top-k selection

**Files/classes:** `src/typedrank/api.py:619` / final coverage construction; `src/typedrank/strategies/listwise.py:62` / shortlist slicing.

Coverage infers missing count from stage input size minus the final output entries when an omission warning exists. This counts successfully scored but unselected items as missing.

**Reproduced:** a backend scored two of three candidates and declared one missing; `top_k=1` returned `scored_count=1, missing_count=2`. Correct values are two scored and one missing.

**Recommended fix:** carry scored/missing coverage independently of selection, using validated backend IDs. Test partial responses with top-k and score thresholds, not only untrimmed partial results.

### M05 — Per-call timeouts and context errors bypass configured backend fallback

**Files/classes:** `src/typedrank/_runtime/executor.py:351`, `:721`; `src/typedrank/errors.py`.

Router fallback catches BackendError and CapabilityError. The runtime converts a per-call timeout to DeadlineExceededError, and adapter context errors are ContextLimitError; neither is caught. Thus a primary call can fail while a configured alternate and request budget remain available, yet no alternate is tried.

**Reproduced:** a delayed primary with `timeout_s=0.005`, a healthy fallback, and no total request deadline raised DeadlineExceededError; fallback calls were zero.

**Recommended fix:** distinguish an exhausted total deadline from a backend attempt timeout. Define explicit eligible fallback categories, including context incompatibility where intended, and preserve the same budget ledger. Never retry after a genuinely exhausted total deadline or cancellation.

### M06 — AutoReranker filters router choices for pointwise before selecting a strategy

**Files/classes:** `src/typedrank/auto.py:68` / `AutoStrategy.decide`.

Auto planning asks the router for pointwise-capable routes first. A backend that supports only listwise is discarded even when a small pool fits one valid listwise call.

**Reproduced:** the same listwise-only backend selected `listwise` when passed directly and `lexical` when wrapped in BackendRouter.

**Recommended fix:** consider eligible capabilities for the strategy being evaluated rather than applying a pointwise filter up front. Ensure planned backend/strategy pairs and execution agree; test listwise-only and differently capable primary/fallback pairs.

### M07 — Custom backend accounting can supply negative or invalid usage

**Files/classes:** `src/typedrank/types.py:54`, `:63` / `Usage`, `RequestStatistics`; `src/typedrank/_runtime/executor.py:690`, `:774` / reconciliation and statistics.

Generic response validation validates candidate IDs and scores, but not attempts, latency, token counts, or cost before they reach the ledger. System One's decoder validates its own token fields; custom ModelBackend implementations do not receive the same protection.

**Reproduced:** a custom response with `input_tokens=-100` and `cost_usd=-1` was returned as `usage_complete=True` with negative statistics. Negative actual usage also reduces ledger consumption.

**Recommended fix:** validate finite nonnegative numeric accounting and positive integer attempt counts at the generic response boundary, before reconciliation. Keep unknown usage distinct from zero and reject invalid values instead of normalizing them silently.

### M08 — Optional Laya compatibility claims outrun dependency and test boundaries

**Files/classes:** `pyproject.toml:28`, `:29`; `tests/test_laya_live.py`, `tests/test_laya_http_live.py`, `tests/test_release_scenarios.py`; architecture §9.

The published range is `laya>=0.3.7,<1`, although the architecture specifies initial certification of 0.3.7 and a narrow tokenizer compatibility boundary. The lock resolves 0.3.7 but does not constrain an end user's pip extra installation. Mock contracts bypass real tokenization, device fallback, loading, and eviction. Live tests check a short valid score, not these guarantees.

**Recommended fix:** initially bound the supported runtime to the actually certified release, then broaden through compatibility tests. Keep heavy dependencies optional. Add opt-in tests for context boundaries, device reporting, and lifecycle; document which OS/Python/device combinations and combined extras are actually exercised. Treat skipped live tests as missing evidence, not backend certification.

### M09 — Local benchmark fields do not establish actual device or warm inference

**Files/classes:** `benchmarks/compare.py:110` / `measured_model`; `src/typedrank/backends/laya.py`; `docs/benchmarks.md`.

The harness reports requested device (or “auto”) as `device` and configured maximum decision batch size as `batch_size`. It calls the first dataset case cold and the median remaining heterogeneous cases warm. Later cases can select and load another checkpoint. The adapter does not report actual device transitions, so CPU fallback can remain invisible in the report.

**Recommended fix:** label configuration values as configuration, record observed device/checkpoint and actual batch sizes, and separate load events from repeated warmed inference. Warm each selected checkpoint before timing comparable repeats. The existing local report contains only baseline/BM25 measurements; this finding does not accuse it of fabricated Laya numbers.

## LOW

### L01 — Smaller-batch listwise fallback is a stale, ineffective option

**Files/classes:** `src/typedrank/api.py:730` / `Reranker._fallback`; `src/typedrank/strategies/listwise.py:35`.

SMALLER_BATCHES halves the allowed listwise size but resubmits the same candidate set. ListwiseStrategy intentionally refuses independently scored chunks. Reducing the limit therefore cannot rescue an oversized group and does not reduce a context-overflowing request's content.

**Recommended fix:** remove or explicitly mark this option unsupported for listwise. Keep the no-uncalibrated-chunk-merge rule; use an explicitly configured pointwise or pruning strategy when that semantic change is acceptable.

## Areas that held up, and remaining limits

- **Simple API and object handling:** `Reranker(backend=backend)` plus `await rerank(...)` remains straightforward. Objects retain their identity; explicit projection and positional occurrence IDs avoid conflating duplicate business IDs. No elaborate abstraction replacement is justified.
- **Shared score validation:** adapters use the typed Noul probability directly, not confidence or an invented cross-candidate softmax. Generic paths validate finite utilities and reject duplicate, unexpected, or inconsistent missing IDs. Full listwise groups are kept intact.
- **Configured fallback:** successful backend fallback reruns the complete model stage under one ledger, reports FALLBACK, and settles child tasks. Default configuration remains strict; the requested Jev/Laya fallback chains are not silently installed. M05 covers categories that escape this behavior.
- **Caching:** real provider identities include backend/endpoint/model/profile information, and local checkpoint selection participates in keys. HTTP auto caching is disabled; a declared Laya revision is needed for opt-in cache stability. A revision label does not itself pin weights. M03 is a concrete exception in FakeBackend.
- **Cost and budgets:** strict remote financial caps require a declared chargeable-cost bound. Missing provider cost stays unknown; retry responses do not pretend all token usage was reported. Local cost is not presented as measured infrastructure cost. The remaining generic tariff coupling and invalid custom usage are M02/M07.
- **Async and resources:** normal Laya loading/prediction uses a dedicated worker with one worker by default; construction locking prevents duplicate Router construction within an instance. No evidence supports claiming an unconditional GPU leak during ordinary successful use. Cancellation/close and optional concurrency defects are specifically H01–H03. Separately created backends can still load separate models.
- **HTTP and security:** lazy owned clients have an explicit close path; supplied transports are borrowed. Keys are excluded from dataclass repr, HTTP responses have a byte cap, raw JSON rejects duplicate keys, and candidate text is treated as data. Prompt wording alone is not proof against model prompt injection. Network-deny is a declaration-based application policy, with the fail-open issue in H06.
- **Provider metadata:** opaque `backend_metadata` is an appropriate place for Laya routing details; it does not require Laya fields in the result's core schema. Last-response model metadata is insufficient for H04, however.
- **Imports and packaging:** an isolated import of `typedrank` and `typedrank.backends` loaded none of torch, laya, transformers, sentence_transformers, numpy, httpx, or fastapi. The built wheel includes `typedrank/py.typed`; inspection found no legacy package, model weights, environment file, Git directory, or bytecode cache paths.
- **Tests and developer documentation:** current tests preserve BM25, metrics, embeddings, RRF, diversity, pipeline, sync and async behavior. Their green status missed the reproduced scenarios. Backend comparisons are appropriately non-marketing, but architecture acceptance gates, runtime limitations, sync ownership restrictions, and benchmark labels need to agree before release. Recommended fixes above should get focused regressions rather than tests that merely mirror code.

## Reproduction record

All of the following were executed in the existing project environment with in-memory doubles, without changing source or adding tests:

```text
CACHE: first score 0.2; second configured score 0.9 returned 0.2; second calls 0
HTTP_AUTO: observed checkpoints [english, multilingual]; status ok; final model multilingual
PARTIAL: input 3; backend scored 2; top_k 1; reported scored 1 and missing 2
UNKNOWN_LOCALITY_DENY_CALLS: 1
TIMEOUT_FALLBACK: DeadlineExceededError; fallback calls 0
EMBED_AFTER_CANCEL_MAX_ACTIVE: 2
CANCELLED_CLOSE: closed True; unloaded False; router retained True
LISTWISE_ONLY_AUTO: direct listwise; routed lexical
LAYA_TIMEOUT_DIAGNOSTIC: TypeSafe request timed out; stage jev
LAYA_CONTEXT_ESTIMATE: 346; admitted under 512 without tokenizer component checks
CUSTOM_ACCOUNTING: input_tokens -100; estimated_cost_usd -1.0; usage_complete True
```

These are audit probes, not claims that live model quality, throughput, GPU behavior, or compatibility across all extras was measured.

## Exhaustive requested-term classification

The scan covered Git-tracked files plus non-ignored untracked working-tree files, including hidden CI files and `uv.lock`. It excluded Git internals, virtual environments, caches/build outputs ignored by Git, and this generated report (to avoid self-counting). Search was case-insensitive for `TYPESAFE_API_KEY|JevReranker|System One|typesafe|jev`; the longer names are counted once rather than also counting their embedded substring.

**438 occurrences across 32 files: 425 legitimately provider-specific; 13 incorrect generic coupling.** The latter are the provider-only factory and Laya's dependency on Jev implementation types. M02 additionally finds coupling through pricing/ownership fields that do not contain the requested search terms.

The requested binary label **legitimately provider-specific** also covers legitimate shared System One protocol references, historical migration descriptions, license provenance, provider fixture names, and explicit backend examples. It does not imply System One belongs exclusively to Jev. A correct Jev error string in `jev.py` is legitimate at its definition; executing it for Laya is classified at the inappropriate inheritance/call sites. Architecture descriptions of old coupling are legitimate audit/history references, not executable coupling.

Each entry below lists every match as `line:column term` using one-based positions in the audited working tree. Files with both classifications have separate entries.

### CHANGELOG.md

**legitimately provider-specific** (6): `5:75 Jev`, `6:12 Jev`, `9:14 System One`, `11:84 Jev`, `14:68 Jev`, `16:41 Jev`.

### CONTRIBUTING.md

**legitimately provider-specific** (2): `5:334 Jev`, `7:51 Jev`.

### LICENSE

**legitimately provider-specific** (1): `3:20 Jev`.

### MIGRATION_PLAN.md

**legitimately provider-specific** (68): `7:25 jev`, `7:116 jev`, `23:70 jev`, `36:20 JevReranker`, `45:13 jev`, `45:24 Jev`, `45:48 jev`, `45:59 Jev`, `45:89 TypeSafe`, `46:13 jev`, `46:24 Jev`, `46:43 Jev`, `46:235 Jev`, `47:13 jev`, `80:13 Jev`, `86:41 JevReranker`, `86:132 Jev`, `88:27 Jev`, `88:103 typesafe`, `89:139 Jev`, `92:178 Jev`, `96:17 jev`, `96:146 TypeSafe`, `97:17 jev`, `97:96 System One`, `97:124 Jev`, `98:173 jev`, `99:52 Jev`, `99:115 Jev`, `101:25 Jev`, `105:124 JevReranker`, `108:102 typesafe`, `108:139 Jev`, `108:170 Jev`, `110:17 jev`, `111:3 Jev`, `111:152 JEV`, `111:175 JEV`, `111:251 TYPESAFE_API_KEY`, `115:7 Jev`, `115:20 TypeSafe`, `115:74 TYPESAFE_API_KEY`, `115:93 Jev`, `115:145 Jev`, `115:157 Jev`, `115:264 Jev`, `115:308 Jev`, `126:15 jev`, `126:112 Jev`, `127:15 jev`, `135:57 Jev`, `137:16 jev`, `141:85 Jev`, `142:114 Jev`, `156:161 JevReranker`, `157:21 Jev`, `160:169 Jev`, `162:36 System One`, `165:219 jev`, `166:62 jev`, `166:210 Jev`, `167:9 Jev`, `170:29 TypeSafe`, `170:103 Jev`, `178:210 Jev`, `208:65 Jev`, `215:269 jev`, `239:15 Jev`.

### README.md

**legitimately provider-specific** (21): `9:34 jev`, `9:43 Jev`, `15:65 Jev`, `15:86 TypeSafe`, `15:95 System One`, `15:120 TYPESAFE_API_KEY`, `15:158 Jev`, `21:32 Jev`, `23:11 Jev`, `38:12 TYPESAFE_API_KEY`, `38:78 Jev`, `56:36 Jev`, `59:47 Jev`, `63:14 Jev`, `63:32 jev`, `83:218 Jev`, `85:46 Jev`, `85:56 jev`, `85:147 Jev`, `85:180 jev`, `98:82 Jev`.

### TYPEDRANK_ARCHITECTURE.md

**legitimately provider-specific** (53): `5:150 jev`, `9:66 jev`, `9:134 Jev`, `14:50 Jev`, `27:13 Jev`, `31:20 System One`, `41:70 jev`, `41:113 jev`, `41:257 Jev`, `45:47 Jev`, `51:82 typesafe`, `51:102 JevReranker`, `71:32 Jev`, `88:1 jev`, `88:7 Jev`, `88:25 jev`, `88:41 TYPESAFE_API_KEY`, `94:26 jev`, `94:202 typesafe`, `94:363 jev`, `118:39 System One`, `143:1 System One`, `149:14 System One`, `151:144 Jev`, `157:5 jev`, `157:48 TypeSafe`, `157:70 Jev`, `163:9 jev`, `163:67 Jev`, `169:339 Jev`, `195:48 Jev`, `209:14 Jev`, `215:387 Jev`, `235:185 Jev`, `249:278 Jev`, `265:9 jev`, `280:51 System One`, `281:14 jev`, `291:321 Jev`, `293:72 Jev`, `299:64 Jev`, `311:47 jev`, `312:44 jev`, `312:106 jev`, `313:45 jev`, `313:107 jev`, `314:48 jev`, `314:110 jev`, `315:51 jev`, `315:113 jev`, `316:49 jev`, `316:111 jev`, `317:48 jev`.

### benchmarks/compare.py

**legitimately provider-specific** (44): `5:10 Jev`, `5:15 TYPESAFE_API_KEY`, `5:51 jev`, `5:94 jev`, `26:32 Jev`, `112:14 Jev`, `154:36 jev`, `154:85 Jev`, `155:28 jev`, `155:49 jev`, `156:28 jev`, `157:28 jev`, `171:14 jev`, `172:14 jev`, `174:35 Jev`, `239:21 jev`, `239:40 TYPESAFE_API_KEY`, `243:23 jev`, `244:21 Jev`, `245:36 jev`, `246:58 jev`, `247:59 jev`, `254:34 jev`, `254:43 TYPESAFE_API_KEY`, `255:34 jev`, `255:71 jev`, `293:36 jev`, `294:26 jev`, `294:50 TYPESAFE_API_KEY`, `295:21 Jev`, `296:28 jev`, `297:50 jev`, `298:51 jev`, `300:53 jev`, `300:62 TYPESAFE_API_KEY`, `346:10 jev`, `346:27 jev`, `346:53 jev`, `347:10 jev`, `348:22 jev`, `349:21 jev`, `352:10 jev`, `353:22 jev`, `354:21 jev`.

### benchmarks/data/synthetic_v1.json

**legitimately provider-specific** (1): `4:58 Jev`.

### benchmarks/local-smoke.json

**legitimately provider-specific** (12): `8:4 jev`, `9:4 jev`, `10:4 jev`, `85:33 jev`, `85:42 TYPESAFE_API_KEY`, `87:20 jev`, `90:33 jev`, `90:42 TYPESAFE_API_KEY`, `92:20 jev`, `105:57 jev`, `105:66 TYPESAFE_API_KEY`, `107:42 jev`.

### docs/backend-routing.md

**legitimately provider-specific** (2): `7:47 Jev`, `11:14 Jev`.

### docs/backends.md

**legitimately provider-specific** (9): `7:3 Jev`, `7:16 TypeSafe`, `7:41 TYPESAFE_API_KEY`, `7:85 jev`, `9:66 System One`, `16:32 Jev`, `18:29 Jev`, `26:163 Jev`, `26:168 jev`.

### docs/benchmarks.md

**legitimately provider-specific** (8): `10:105 Jev`, `12:216 Jev`, `12:249 jev`, `12:261 TYPESAFE_API_KEY`, `12:300 jev`, `16:1 Jev`, `16:54 jev`, `16:94 jev`.

### docs/jev.md

**legitimately provider-specific** (12): `1:3 Jev`, `3:2 Jev`, `3:60 TypeSafe`, `3:69 System One`, `3:121 TYPESAFE_API_KEY`, `3:165 Jev`, `8:34 jev`, `13:32 Jev`, `15:11 Jev`, `15:33 TYPESAFE_API_KEY`, `25:185 jev`, `25:255 Jev`.

### docs/laya.md

**legitimately provider-specific** (3): `3:198 System One`, `23:12 System One`, `39:203 System One`.

### docs/migration-from-jev-rankkit.md

**legitimately provider-specific** (15): `1:18 Jev`, `3:110 jev`, `11:32 Jev`, `13:29 Jev`, `16:111 Jev`, `16:116 Jev`, `16:145 Jev`, `16:159 TYPESAFE_API_KEY`, `16:178 TypeSafe`, `16:202 Jev`, `16:251 Jev`, `18:151 Jev`, `18:247 Jev`, `18:252 jev`, `20:55 jev`.

### docs/usage.md

**legitimately provider-specific** (23): `11:16 Jev`, `14:34 jev`, `89:11 Jev`, `91:27 TypeSafe`, `91:91 TYPESAFE_API_KEY`, `91:135 Jev`, `96:25 TypeSafe`, `97:6 TYPESAFE_API_KEY`, `99:17 TYPESAFE_API_KEY`, `105:12 TypeSafe`, `105:32 TYPESAFE_API_KEY`, `107:8 TYPESAFE_API_KEY`, `109:7 TYPESAFE_API_KEY`, `112:20 TypeSafe`, `118:32 Jev`, `128:15 Jev`, `148:1 Jev`, `156:47 Jev`, `159:48 Jev`, `178:46 Jev`, `178:51 jev`, `188:32 Jev`, `196:15 Jev`.

### examples/16_jev_backend.py

**legitimately provider-specific** (4): `1:35 Jev`, `1:57 TYPESAFE_API_KEY`, `6:32 Jev`, `14:15 Jev`.

### examples/18_laya_http.py

**legitimately provider-specific** (1): `1:34 System One`.

### examples/19_backend_fallback.py

**legitimately provider-specific** (3): `1:42 Jev`, `6:47 Jev`, `12:18 Jev`.

### examples/README.md

**legitimately provider-specific** (6): `16:10 Jev`, `16:20 jev`, `16:39 jev`, `16:77 jev`, `16:88 TYPESAFE_API_KEY`, `19:107 Jev`.

### pyproject.toml

**legitimately provider-specific** (1): `26:1 jev`.

### src/typedrank/backends/__init__.py

**legitimately provider-specific** (5): `11:7 jev`, `11:18 Jev`, `11:30 Jev`, `24:6 Jev`, `25:6 Jev`.

### src/typedrank/backends/factory.py

**incorrect generic coupling** (5): `8:7 jev`, `8:18 Jev`, `12:25 typesafe`, `15:39 typesafe`, `16:16 Jev`.

### src/typedrank/backends/jev.py

**legitimately provider-specific** (31): `1:10 TypeSafe`, `1:22 System One`, `1:35 Jev`, `42:1 Jev`, `43:1 Jev`, `47:7 Jev`, `48:8 Jev`, `51:19 jev`, `52:34 typesafe`, `58:16 Jev`, `65:36 Jev`, `73:17 typesafe`, `89:18 typesafe`, `93:31 jev`, `96:42 TYPESAFE_API_KEY`, `98:40 TypeSafe`, `102:70 jev`, `136:19 Jev`, `155:30 TypeSafe`, `157:40 jev`, `179:33 TypeSafe`, `209:43 Jev`, `213:39 jev`, `215:40 TypeSafe`, `218:18 TypeSafe`, `219:45 jev`, `222:38 TypeSafe`, `225:18 TypeSafe`, `226:45 jev`, `228:29 TypeSafe`, `232:19 Jev`.

### src/typedrank/backends/laya.py

**legitimately provider-specific** (2): `1:36 System One`, `234:27 Jev`.

**incorrect generic coupling** (8): `25:7 jev`, `25:18 Jev`, `25:30 Jev`, `233:23 Jev`, `243:9 Jev`, `309:26 Jev`, `314:26 Jev`, `323:43 Jev`.

### src/typedrank/backends/systemone.py

**legitimately provider-specific** (31): `1:25 System One`, `28:50 System One`, `60:42 System One`, `78:38 System One`, `82:42 System One`, `86:14 System One`, `88:20 System One`, `93:39 System One`, `105:14 System One`, `107:20 System One`, `142:66 System One`, `164:54 System One`, `173:37 System One`, `175:18 System One`, `187:21 jev`, `191:37 System One`, `271:20 jev`, `280:48 Jev`, `288:42 System One`, `298:46 System One`, `305:38 System One`, `307:38 System One`, `314:39 System One`, `336:38 System One`, `338:38 System One`, `342:38 System One`, `344:38 System One`, `353:38 System One`, `376:38 System One`, `381:42 System One`, `386:46 System One`.

### tests/test_backend_contract.py

**legitimately provider-specific** (7): `1:48 System One`, `11:5 Jev`, `63:17 jev`, `64:16 Jev`, `82:36 jev`, `96:36 jev`, `105:36 jev`.

### tests/test_jev_backend.py

**legitimately provider-specific** (29): `9:50 Jev`, `66:19 jev`, `76:16 jev`, `78:15 Jev`, `81:33 jev`, `90:16 jev`, `92:15 Jev`, `114:16 jev`, `117:15 Jev`, `127:16 jev`, `130:15 Jev`, `157:16 jev`, `158:15 Jev`, `169:16 jev`, `171:15 Jev`, `183:16 jev`, `184:15 Jev`, `194:10 jev`, `196:9 Jev`, `197:15 Jev`, `212:15 Jev`, `221:20 jev`, `226:15 Jev`, `236:12 Jev`, `237:12 Jev`, `241:15 Jev`, `249:15 Jev`, `261:16 jev`, `262:15 Jev`.

### tests/test_jev_live.py

**legitimately provider-specific** (9): `9:50 Jev`, `17:21 jev`, `18:29 JEV`, `19:36 JEV`, `20:23 TYPESAFE_API_KEY`, `21:22 TYPESAFE_API_KEY`, `22:15 Jev`, `23:36 JEV`, `23:49 jev`.

### tests/test_laya_http_live.py

**legitimately provider-specific** (1): `1:40 System One`.

### tests/test_release_scenarios.py

**legitimately provider-specific** (12): `11:60 Jev`, `86:17 jev`, `87:16 Jev`, `98:44 jev`, `112:77 jev`, `123:25 jev`, `124:11 jev`, `138:56 typesafe`, `138:82 jev`, `142:49 jev`, `162:50 jev`, `184:56 typesafe`.

### uv.lock

**legitimately provider-specific** (3): `1929:1 jev`, `1953:43 jev`, `1964:29 jev`.

