"""Database path resolution and configuration."""

import os
from pathlib import Path

from biom3.core.helpers import load_json_config
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

DATABASES_ROOT_ENV_VAR = "BIOM3_DATABASES_ROOT"
DEFAULT_CONFIG_PATH = "configs/dbio_config.json"


def _resolve_root(raw_path, config_path):
    """Resolve a possibly-relative path against the current working directory."""
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return Path.cwd().joinpath(p).resolve()


def _load_config(config_path=None):
    """Load the dbio config JSON, falling back to the default path."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return None
    return load_json_config(path)


def get_databases_root(config_path=None):
    """Resolve the root directory containing downloaded databases.

    Priority: BIOM3_DATABASES_ROOT env var > config file > raise.
    """
    env_val = os.environ.get(DATABASES_ROOT_ENV_VAR)
    if env_val:
        root = Path(env_val).resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"{DATABASES_ROOT_ENV_VAR}={env_val} does not exist"
            )
        return root

    config = _load_config(config_path)
    if config and "databases_root" in config:
        root = _resolve_root(config["databases_root"], config_path or DEFAULT_CONFIG_PATH)
        if root.is_dir():
            return root
        raise FileNotFoundError(
            f"databases_root={config['databases_root']} (resolved: {root}) does not exist"
        )

    raise RuntimeError(
        f"Set {DATABASES_ROOT_ENV_VAR} or provide databases_root in {DEFAULT_CONFIG_PATH}"
    )


def get_database_path(database_name, config_path=None):
    """Return path to a specific database subdirectory (e.g., 'ncbi_taxonomy')."""
    config = _load_config(config_path)
    subdir = database_name
    if config:
        subdir = config.get("databases", {}).get(database_name, database_name)
    root = get_databases_root(config_path)
    return root / subdir


def get_training_data_root(config_path=None):
    """Resolve the root directory containing training CSVs."""
    config = _load_config(config_path)
    if config and "training_data_root" in config:
        return _resolve_root(
            config["training_data_root"], config_path or DEFAULT_CONFIG_PATH
        )
    return get_databases_root(config_path)


def get_training_data_path(dataset_name, config_path=None):
    """Return path to a pre-built training CSV (e.g., 'swissprot_csv')."""
    config = _load_config(config_path)
    filename = dataset_name
    if config:
        filename = config.get("training_datasets", {}).get(dataset_name, dataset_name)
    root = get_training_data_root(config_path)
    return root / filename
