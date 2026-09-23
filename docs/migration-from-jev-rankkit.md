# Migration from Jev Rankkit

TypedRank is a separate distribution and import package. Install it and change generic library imports from `jev_rankkit` to `typedrank`:

```bash
python -m pip install typedrank
```

```python
from typedrank import Reranker
from typedrank.backends import JevBackend

reranker = Reranker(backend=JevBackend())
```

The generic pipeline stage is `ModelReranker`; it works with the `ModelBackend` contract rather than assuming Jev. Jev-specific names, such as `JevBackend`, `TYPESAFE_API_KEY`, TypeSafe endpoints, and Jev model IDs remain provider-specific. Existing Jev deployments can keep using that backend while adopting backend-neutral ranking strategies and candidate types.

The package also provides local and HTTP Laya backends and `BackendRouter`. These are optional paths; adopting TypedRank does not require migrating a Jev integration to Laya. Compare setup and deployment requirements in [Backends](backends.md), [Jev](jev.md), and [Laya](laya.md).

The Python project name is `typedrank`; the original `jev-rankkit` project remains separate. Review the [usage guide](usage.md) for generic object adapters, metrics, pipelines, and convenience methods.
