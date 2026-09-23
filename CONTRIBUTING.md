# Contributing

Set up the development environment with `python -m pip install -e ".[dev]"`. Run `pytest`, `ruff check .`, `ruff format --check .`, `mypy src/typedrank`, and `python -m build` before submitting a change.

Keep the core dependency-free, preserve the original Python objects returned by reranking, keep provider-specific behavior inside backends, and add regression tests for behavior changes. Live Jev tests are opt-in and can incur charges. Do not add Laya functionality as part of the mechanical baseline.

The project is MIT-licensed. Changes derived from Jev Rankkit retain its license notice.
