"""App settings with bundled defaults and optional user overrides."""

from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path

SETTINGS_ENV_VAR = "BIOM3_APP_CONFIG"


def get_default_settings() -> dict:
    """Load the default settings bundled with the package."""
    ref = files("biom3.app").joinpath("default_settings.json")
    return json.loads(ref.read_text())


def load_settings(user_config_path: str | None = None) -> dict:
    """Load app settings, merging user overrides on top of defaults.

    Resolution order:
      1. Bundled ``default_settings.json``
      2. User config via *user_config_path* argument
      3. User config via ``BIOM3_APP_CONFIG`` environment variable

    User values override defaults at the top level (shallow merge).
    """
    settings = get_default_settings()
    path = user_config_path or os.environ.get(SETTINGS_ENV_VAR)
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p) as f:
            user_settings = json.load(f)
        settings.update(user_settings)
    return settings
