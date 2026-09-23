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

## Required before publishing

- [ ] Create or confirm the GitHub repository at `https://github.com/kashyaprparmar/typedrank`. There is no Git remote configured in this checkout, and the README project links currently return 404 because the repository is not available yet.
- [ ] Review the intended release version and changelog entry after repository creation.
- [ ] Run opt-in Jev or Laya live checks only when service credentials, service availability, and any provider charges are acceptable.
- [ ] Have a maintainer review the release candidate and then publish the package or GitHub release manually.

No PyPI publication or GitHub release was performed during this cleanup.
