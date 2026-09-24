# TypedRank release checklist

## Completed for version 0.1.0

- [x] Confirm distribution and import names are `typedrank`; package metadata declares MIT and Python 3.11+.
- [x] Review release wording and provider names. Jev, TypeSafe, and Laya are identified only as provider/backend names or in source migration and license history.
- [x] Review optional extras: `jev`, `laya` pinned to `0.3.7`, `http`, `embeddings`, and `all`.
- [x] Verify internal README documentation links resolve to files in this source tree.
- [x] Run `python -m build`; inspect the wheel for `typedrank/py.typed`, backend modules, and package contents.
- [x] Exclude internal audit/architecture plans and generated smoke reports from the source distribution.
- [x] Run pytest with coverage: 187 passed, 3 opt-in live tests skipped; 89% statement coverage.
- [x] Run `ruff check .`, `ruff format --check .`, and `mypy src/typedrank benchmarks/compare.py`.
- [x] Install the built wheel into separate clean Python 3.11 environments for base, `[jev]`, `[laya]`, `[embeddings]`, and `[all]`; import public backend names and run smoke checks.
- [x] Confirm the base and Jev-only imports do not load Torch or Laya; Laya mock scoring works without `TYPESAFE_API_KEY`; Laya HTTP mock scoring works without local Torch or Laya.
- [x] Confirm the current local benchmark report includes only measurements actually collected by TypedRank.

## Release outcome

- [x] Create the public GitHub repository at `https://github.com/kashyaprparmar/typedrank` and push `main`.
- [x] Confirm the GitHub Actions matrix passes on Ubuntu and Windows with Python 3.11 and 3.13.
- [x] Review version `0.1.0` and the changelog entry.
- [x] Upload the wheel and source archive to PyPI as `typedrank==0.1.0` and install the published version in a clean environment.
- [ ] Optional live Jev and Laya checks remain skipped; their CI cases require provider credentials, an HTTP service, or downloading local model weights.

No GitHub Release was created. The PyPI publication was explicitly requested and completed.

## Version 0.1.1 image update

- [x] Add the architecture image at `docs/assets/typedrank-architecture.png` and link to it from README using a URL that renders on GitHub and PyPI.
- [x] Bump package metadata and lock entry to `0.1.1`; add the changelog entry.
- [x] Build sdist and wheel, pass `twine check`, and confirm the image is included in the sdist.
- [x] Install the wheel and verify the public imports and lazy optional dependency loading.
- [x] Push the source update; GitHub CI passed on Ubuntu and Windows with Python 3.11 and 3.13.
- [x] Publish the 0.1.1 wheel and sdist to PyPI and install the published wheel in a clean environment.
