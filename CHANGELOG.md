# Changelog

## 0.1.0 — initial TypedRank development baseline

- Introduce the separate `typedrank` import and distribution, ported from Jev Rankkit source.
- Preserve Jev provider support and the existing generic ranking behavior.
- Rename the generic pipeline model stage to `ModelReranker`.
- Add the TypedRank `py.typed` marker and distribution extras.
- Add shared System One codec, local and HTTP Laya backends, explicit backend routing, and stage-level fallback.
- Record backend provenance and keep local monetary cost unknown; make Laya caching opt in with a declared weight revision.

This version is a new-project baseline; Jev Rankkit release history is not TypedRank release history.
