# 2026-05-05 — `--dry_run` preview for trainers + `gpu_devices` → `devices_per_node` rename

Branch: `addison-dev`. Three commits land in this session:

- `0204f59` fix(stage3): make nonestr_to_none pass through non-string values
- `81ef425` feat(stages): add --dry_run preview and rename gpu_devices → devices_per_node
- `<this note>` docs(session): 2026-05-05 dry-run + devices_per_node rename note

## Why

Two motivations, addressed together because both touch the same trainer plumbing:

1. **Pre-flight feedback for HPC training jobs.** Submitting to Spark / Polaris / Aurora costs queue time and GPU-hours. The only way to validate a config was to start the run and read the first 30 s of logs, and even then the *effective* arg values were not reported with provenance — users could not tell whether a value came from CLI, a base config, or a default.
2. **`gpu_devices` was a poor name.** It implies CUDA, but the same arg is used for XPU tile counts on Aurora. `devices_per_node` is the canonical term. Renaming was a deferred chore from earlier sessions; doing it alongside the dry-run feature lets the dry-run report ship with the corrected naming.

## `nonestr_to_none` int/float bug fix (`0204f59`)

Pre-existing bug spotted while reading [src/biom3/Stage3/run_PL_training.py](../../src/biom3/Stage3/run_PL_training.py): `nonestr_to_none` raised `ValueError` on any non-string non-`None` value, but argparse coerces `--checkpoint_every_n_steps` and `--checkpoint_every_n_epochs` to `int` *before* `retrieve_all_args` runs. The strict helper would have crashed on any CLI-supplied integer for those flags. The marker `# TODO: BUG FIX ISSUE` on line 976 was the breadcrumb.

Fix: pass through any non-string input unchanged. The helper is now idempotent across the CLI / JSON / default paths.

## `--dry_run` preview (part of `81ef425`)

CLI surface, identical across `biom3_train_stage{1,2,3}`:

```text
--dry_run True            print the report and exit; no training
--dry_run_output False    stdout only (default)
                  True    write JSON to <artifacts_dir>/dry_run_report.json
                  <path>  write JSON to that filepath
```

Report sections:

1. **Effective configuration** — every arg with its source: `CLI`, `JSON: <abs path>` (the specific file in the `_base_configs`/`_overwrite_configs` chain), or `default`.
2. **Output paths the run would create** — resolved absolute paths for `run_dir`, `logs_dir`, `artifacts_dir`, `checkpoint_dir`, `args.json`, `build_manifest.json`, `run.log`. Nothing is created on disk.
3. **Distributed / batch math** — `num_nodes`, `devices_per_node`, `world_size`, `micro_batch_size`, `acc_grad_batches`, `effective_batch_size`, `train_dataset_len`, `val_dataset_len`, `batches_per_epoch_per_rank`, `steps_per_epoch`, plus stage-relevant fields.
4. **Memory estimate** — strategy-aware:
   - `ddp` / `single_device`: full-replication math (`4·N + 4·N + 8·N` bytes for fp32 AdamW per rank).
   - `deepspeed_zero2`: per-rank sharded estimate via `deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_cold`, with the deepspeed estimator's stdout breakdown captured into the report.
   - Plus `per_minibatch_input_gb` from one micro-batch via `tensor.element_size() * tensor.numel()` recursion.
   - Activation memory is **not** included (requires a forward pass) — caveat printed.

Implementation: [src/biom3/core/dry_run.py](../../src/biom3/core/dry_run.py). The renderers are reused across stages; each stage's `main()` only adds a tiny branch:

```python
if getattr(args, 'dry_run', False):
    return run_dry_run(
        args,
        stage="stage3",
        dataset_probe=lambda: _stage3_dataset_probe(args),
        model_probe=lambda: _stage3_model_probe(args),
    )
```

The dataset / model probes build their artifacts on CPU. Failures degrade gracefully — the report still prints the parts that succeeded, with a `[Notes]` section listing what failed.

### CLI-vs-default detection — what we did *not* do

First attempt mutated `parser._actions[*].default` and `parser._defaults` to `argparse.SUPPRESS`, parsed argv again, and used the resulting namespace to infer "explicitly set on the CLI." Two problems surfaced in code review:

1. **Argparse's type converter still ran.** Setting `action.default = SUPPRESS` (the string `"==SUPPRESS=="`) made argparse try `float("==SUPPRESS==")` for `--lr`. Could be worked around, but ugly.
2. **Stashing the parser on `args`.** I had a brief checkpoint where `args._parser = parser` and `args._argv = argv` were set inside `retrieve_all_args` so `main()` could read them. This *broke pickling* — Stage 3's parser registers `type=float_or_str`, a local function inside `get_model_args`, and Lightning pickles `args` for DDP worker processes and checkpoints. It also dumped the parser representation into `args.json` and `build_manifest.json` via the existing serializers.

Both classes of problem disappeared with a much simpler approach: check whether `--<dest>` (or `--<dest>=...`) appears in `sys.argv`, given the dest names from `vars(args)`. Argparse's standard convention is `--foo_bar` → `dest='foo_bar'`, so this is exactly equivalent to "what dest names did the user type on the CLI?" — without any parser introspection or namespace pollution. `pickle.dumps(args)` works; nothing leaks into manifests; `vars(args)` has zero underscore-prefixed keys.

`detect_cli_keys` is now ~10 lines of pure string matching. The one edge case it doesn't handle is short flags like `-c` for `--config_path`; in practice every BioM3 training arg uses `--<long>` form on the production CLI, and `--config_path` is the only one with a short alias.

### What `load_json_config` gained

Opt-in `track_provenance: bool = False` parameter. When True, returns `(merged_dict, provenance_dict)` where `provenance[key]` is the absolute path of the file that supplied that key's final value (after the full base/current/overwrite merge order). Default behavior (returns plain `dict`) is unchanged — existing callers untouched.

The provenance merge mirrors the value-merge order: `_base_configs` → current file → `_overwrite_configs`, with each later layer overwriting both the value and the provenance entry.

### What `core.run_utils` gained

`_collect_training_env` had been duplicated across all three stages with slightly different prefix lists (Stage 1/2 had `ZE_`, `CCL_`, `ONEAPI_`; Stage 3 didn't). Promoted to `biom3.core.run_utils.collect_training_env`, unified on the richer prefix list, and all three stages now call into it. Same for `resolve_devices_per_node` (see below).

## `gpu_devices` → `devices_per_node` rename (part of `81ef425`)

| Layer | Canonical | Deprecated alias |
|---|---|---|
| CLI flag (3 trainers) | `--devices_per_node` | `--gpu_devices` (still parses) |
| JSON config keys | `"devices_per_node"` | n/a (config file should be migrated) |
| `args.<attr>` | `args.devices_per_node` | not stored — alias is collapsed at parse time |
| `build_manifest.json` outputs block | `"devices_per_node"` | n/a |

The deprecation logic lives in `biom3.core.run_utils.resolve_devices_per_node(args)`:

```python
new_val = getattr(args, "devices_per_node", None)
old_val = getattr(args, "gpu_devices", None)
if new_val is None and old_val is not None:
    logger.warning("--gpu_devices is deprecated; use --devices_per_node")
    new_val = old_val
if new_val is None:
    new_val = default  # 1
args.devices_per_node = int(new_val)
```

Both flags default to `None` (sentinel "unset"); `resolve_devices_per_node` is called once at the end of each stage's `retrieve_all_args` and produces the canonical `args.devices_per_node`. **Local variables and internal references** were renamed too — every internal `gpu_devices` (local var, kwarg, attribute, manifest key) is now `devices_per_node`. The only remaining `gpu_devices` strings in the codebase are the three deprecated-alias `add_argument` declarations, the four lines inside `resolve_devices_per_node` that detect/warn about the alias, and three doc-table notes pointing users at the new name.

The Stage 3 streamlit Training Run Viewer ([src/biom3/app/pages/10_Training_Run_Viewer.py](../../src/biom3/app/pages/10_Training_Run_Viewer.py)) silently skips missing keys, so old manifests on disk just won't display this particular field — acceptable for a deprecation; old runs are not retroactively rewritten.

## Files touched in `81ef425`

62 files, +1490 / -197. Categories:

- **Source (8 files)**: 3 stage trainers, `core/dry_run.py` (new), `core/helpers.py`, `core/run_utils.py`, `Stage3/callbacks.py`, `app/pages/10_Training_Run_Viewer.py`, `benchmarks/Stage3/training.py`.
- **Configs (21 files)**: every JSON in `configs/stage{1,2,3}_training/` and two benchmark configs.
- **HPC scripts (6 files)**: every `scripts/stage{1,2,3}_train_{single,multi}node.sh`.
- **Test fixtures (16 files)**: every `tests/_data/entrypoint_args/training/*.txt` and two `tests/_data/configs/test_stage3_config_v{1,2}.json`.
- **Test code (5 files)**: 3 new per-stage `test_stage{1,2,3}_dry_run.py`, 2 new `test_dry_run.py` + `test_helpers_provenance.py` in core, plus updates to `test_callbacks.py` and `test_stage3_training_runs.py`.
- **Docs (3 files)**: `CLI_reference.md`, `aurora_distributed_training.md`, `stage3_training.md`.

## Verification

- **70 new tests pass** (39 unit + 7 integration + tests touched by the rename).
- **`pytest tests/ --quick`: 670 passed, 151 skipped, 0 failures, 1m43s.**
- **Pickling check**: `pickle.dumps(args)` succeeds — DDP/checkpoint-safe; no underscore-prefixed keys in `vars(args)`.
- **Deprecation alias smoke**: `--gpu_devices 1` → warning logged, `args.devices_per_node = 1`, `args.gpu_devices = 1`.
- **End-to-end dry-run smoke** (Stage 3, single device + multi-node): both render correctly; ZeRO-2 estimator stdout captured into report.

## Known follow-ups (not done in this session)

- A real-config dry-run smoke against `configs/stage3_training/pretrain_scratch_v3.json` on Spark / Polaris / Aurora is the only thing left that no automated test can validate.
- The HPC launcher scripts' env handling for `--devices_per_node` was not test-exercised — only manually inspected. The `NGPU` / `NGPU_PER_NODE` shell vars still carry the old name internally; only the `--<flag>` they pass to the trainer was renamed. Could be cleaned up in a follow-up PR.
- The user mentioned at session start that there were "two items to plan for"; the rename was item one. Item two has not yet surfaced.
- A pre-existing UX nit: running `biom3_train_stage3` with no args raises `TypeError` in `os.path.join(None, ...)` because `--output_root`/`--run_id` default to `None`. Confirmed pre-existing on `addison-dev` (not introduced by this session). Could be fixed with either an `args == [] → retrieve_all_args(["--help"])` guard or by making those two args `required=True`. Not done.
