# Data directory

This directory is gitignored. It contains working data files (input CSVs, outputs) and a `databases/` subdirectory populated with symlinks to shared reference databases.

## Reference databases

The `databases/` subdirectory is set up via the sync script, which creates symlinks to the shared database path on each machine:

```bash
./scripts/sync_databases.sh <shared_databases_path> data/databases
```

See [docs/setup_databases.md](../docs/setup_databases.md) for machine-specific shared paths, the full list of database files, and configuration details.
