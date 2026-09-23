# Contributing

Set up the development environment with `python -m pip install -e ".[dev]"`. Run `pytest`, `ruff check .`, `ruff format --check .`, `mypy src/typedrank`, and `python -m build` before submitting a change.

Keep the core dependency-free, preserve the original Python objects returned by reranking, and keep provider-specific behavior inside backends. Document public API changes and include an example when introducing a user-facing feature. Keep backend comparison factual and describe optional dependencies and network requirements. Live Jev tests are opt-in and can incur provider charges; Laya live tests can download model weights and use local CPU/GPU resources.

The project is MIT-licensed. Changes derived from Jev Rankkit retain its license notice.
