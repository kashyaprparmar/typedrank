# Release checklist

- [ ] Run pytest, Ruff lint and format, mypy, and `python -m build`.
- [ ] Install the built wheel in a clean environment and verify `import typedrank` and `typedrank/py.typed`.
- [ ] Confirm the base install does not import optional providers or download models.
- [ ] Review version, distribution metadata, changelog, and MIT notices.
- [ ] Run opt-in live provider checks only with explicit authorization and valid test credentials.
- [ ] Publish only after the new TypedRank repository and package have been reviewed.
