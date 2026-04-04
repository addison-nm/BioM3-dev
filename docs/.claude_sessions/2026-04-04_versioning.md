# Session: Package Versioning (2026-04-04)

## Summary

Added single-source-of-truth versioning to the biom3 package at `0.1.0a1` (alpha pre-release), enabling other ecosystem repos (BioM3-workflow-demo, BioM3-data-share, etc.) to declare version dependencies on biom3. Fixed `get_biom3_version()` in build manifests to read the live `__version__` attribute instead of stale `importlib.metadata`.

## Pre-session state

```bash
git checkout 2f18fde  # feat: add app config system with bundled defaults and optional deps
```

## Changes

### 1. Version string in `src/biom3/__init__.py`

- Added `__version__ = "0.1.0a1"` — the single source of truth for the package version.
- `0.1.0a1` follows PEP 440 pre-release convention: `0.x` signals pre-stable, `a1` signals alpha.

### 2. Dynamic version in `pyproject.toml`

- Replaced the hardcoded `version = "0.0.1"` with `dynamic = ["version"]`.
- Added `[tool.setuptools.dynamic]` section pointing at `biom3.__version__` via the `attr` directive.
- This means `pyproject.toml` and `__init__.py` never drift apart.

### 3. Fixed `get_biom3_version()` in `src/biom3/core/run_utils.py`

- Was using `importlib.metadata.version("biom3")` which reads from installed `.dist-info` metadata — this can go stale in editable installs when the version changes without reinstalling.
- Changed to `from biom3 import __version__; return __version__` so build manifests always report the current version.
- `get_biom3_build()` (which calls `get_biom3_version()`) now produces correct output, e.g. `0.1.0a1+g2f18fde.dirty`.

## Versioning notes

- Python version ordering: `0.0.1 < 0.1.0a1 < 0.1.0b1 < 0.1.0rc1 < 0.1.0`
- `pip install biom3 >= 0.0.1` matches `0.1.0a1`.
- By default `pip install biom3` from a package index skips pre-release versions unless `--pre` is used or a pre-release specifier is given. Not relevant for editable installs from git.
- No git tag was created — tagging deferred until a formal release is ready.

## Discussed but not implemented

- **Print statements in entrypoints**: Briefly added `print(f"biom3 v{__version__}")` to all `__main__.py` entrypoints, then reverted — the user's intent was specifically about build manifest JSON files, which were already handled via `write_manifest()` in `run_utils.py`.
- **`setup.py`**: The IDE showed this file opened but it doesn't exist on disk. Not needed since `pyproject.toml` handles everything.
