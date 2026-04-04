# App Config System — Bundled Defaults with User Overrides

**Date:** 2026-04-04
**Branch:** `addison-main`
**Pre-session state:** `git checkout ace077e`

## Summary

Added a configuration system to `biom3.app` so that default settings are bundled with the package and users can override them by providing their own config file. Previously, the app's data browser had a hardcoded path to `configs/app_data_dirs.json` relative to the repo root, which only worked in an editable install from the repo directory.

Also consolidated all build configuration into `pyproject.toml`, removed the redundant `setup.py`, added optional `[app]` dependencies, and documented install commands with proper shell quoting.

## Changes

### New files

- **`src/biom3/app/default_settings.json`** — Default app settings bundled as package data. Contains the `data_dirs` list (Outputs, Weights, Test Data). Transparent: visible in source tree and in site-packages after install.
- **`src/biom3/app/config.py`** — Config loading module:
  - `get_default_settings()` — loads bundled defaults via `importlib.resources`
  - `load_settings(user_config_path=None)` — loads defaults, then shallow-merges user overrides on top. User config can be passed directly or via `BIOM3_APP_CONFIG` env var.

### Modified files

- **`src/biom3/app/__init__.py`** — `main()` entry point now accepts `--config <path>` CLI argument. Sets the `BIOM3_APP_CONFIG` env var before launching the Streamlit subprocess so the config path propagates correctly.
- **`src/biom3/app/_data_browser.py`** — Replaced hardcoded `DEFAULT_CONFIG_PATH` and local `_load_config()` / `get_data_dirs(config_path)` with a call to `load_settings()` from the new config module. Updated the "no directories configured" message to reference the new `--config` flag and env var.
- **`pyproject.toml`** — Consolidated all build config:
  - Added `[tool.setuptools.packages.find]` with `where = ["src"]` (moved from setup.py)
  - Added `[tool.setuptools.package-data]` to include `*.json` files from `biom3.app`
  - Added `[project.optional-dependencies]` with `app = ["streamlit", "py3Dmol", "biopython", "matplotlib"]`
- **`README.md`** — Expanded install section with pip install commands showing editable, GitHub, and optional `[app]` extra installs. Added note about required shell quoting for zsh.

### Deleted files

- **`setup.py`** — Removed. All configuration now lives in `pyproject.toml`. The `packages.find` and `package-data` directives were moved there.

### Renamed files

- `configs/app_data_dirs.json` → **`configs/app_settings.json`** — renamed to serve as a general app settings file (can grow beyond just data dirs).

## Usage

```bash
# Install with app dependencies
pip install -e '.[app]'

# From GitHub
pip install 'biom3[app] @ git+https://github.com/addison-nm/BioM3-dev.git'

# Use bundled defaults
biom3_app

# Override with a custom settings file
biom3_app --config my_settings.json

# Or via environment variable
BIOM3_APP_CONFIG=my_settings.json biom3_app
```

User config does a **shallow merge** — any top-level key in the user file replaces the corresponding default. This means providing `data_dirs` in a user config replaces the entire default list (no list appending).

## Discussion: broader config system

The session began with a discussion about applying this pattern to all stages (Stage 1/2/3 inference configs). The decision was to scope this work to the app only for now. The same pattern (bundled defaults + user overrides via `importlib.resources`) could be applied to the stage configs later by creating a similar `_default_configs/` directory and `load_config_with_defaults()` helper in `biom3.core`.
