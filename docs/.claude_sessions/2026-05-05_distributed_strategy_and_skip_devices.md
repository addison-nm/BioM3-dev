# 2026-05-05 — `--distributed_strategy` flag, `--skip_devices` conftest option, dry-run provenance fix

Continuation of [2026-05-05_dry_run_preview_and_devices_per_node_rename.md](2026-05-05_dry_run_preview_and_devices_per_node_rename.md). Branch: `addison-dev`. Two code commits + this note.

- `c3c39aa` feat(stage3): `--distributed_strategy` flag (deepspeed_zero2|ddp) + dry-run provenance fix
- `bee5a77` test(conftest): add `--skip_devices` CLI option
- `<this note>` docs(session): 2026-05-05 distributed_strategy + skip_devices note

## Why

Three small items, addressed together because they all surfaced from the dry-run smoke output and follow-up Spark testing:

1. **Stage 3's Lightning strategy was hardcoded to DeepSpeed Stage 2.** A parallel `ddp_strategy = DDPStrategy(...)` was already constructed at [src/biom3/Stage3/run_PL_training.py:1503-1507](../../src/biom3/Stage3/run_PL_training.py#L1503-L1507) but unselectable without editing source. Operationally we want the choice exposed for Aurora/xccl debugging, smaller-model runs where ZeRO-2's CPU offload is the bottleneck, and CI gates that prefer single-file `.ckpt`.
2. **The dry-run report's CLI provenance attribution was broken when callers invoked `main(args)` directly** (tests, `python -c`). `run_dry_run` was falling back to `sys.argv[1:]` which doesn't reflect the caller's argv — every CLI-set flag was being mis-attributed to "default" in the report's effective-config table. Surfaced during the smoke for the new flag.
3. **`test_io.py` and `test_stage3_run_ProteoScribe_sample.py` parametrize over `["cpu", "cuda", "xpu"]`**, and on Spark the `cpu` variants are slow redundancies you don't want in the loop. Needed a switch.

## `--distributed_strategy` (`c3c39aa`)

CLI surface on `biom3_train_stage3` only:

```text
--distributed_strategy {deepspeed_zero2,ddp}    default: deepspeed_zero2
```

`deepspeed_zero2` keeps every existing run's behavior bit-for-bit — same DeepSpeed config, same sharded ZeRO checkpoint dir at end-of-training. `ddp` selects the already-plumbed `DDPStrategy(static_graph=True, gradient_as_bucket_view=True)` and writes a single-file `last.ckpt` instead.

**`save_model` is unchanged.** The inline comment at the old line 1515 claimed DDP would break `save_model`'s `convert_zero_checkpoint_to_fp32_state_dict` call, but `_convert_or_copy_checkpoint` at [run_PL_training.py:579-593](../../src/biom3/Stage3/run_PL_training.py#L579-L593) already branches on `os.path.isdir(src_ckpt)` — directory → `convert_zero_checkpoint_to_fp32_state_dict`, single file → `shutil.copy2`. Stale comment removed.

Stage 1 / Stage 2 keep their existing dynamic DDP/SingleDevice selection based on `device` and `devices_per_node`. They don't use DeepSpeed and don't need this flag.

`build_manifest.json` outputs block now records `distributed_strategy`, and `deepspeed_stage: "2"` is only emitted when actually using DeepSpeed. `dry_run.infer_strategy` reads `args.distributed_strategy` so the dry-run memory section matches the chosen strategy.

`docs/CLI_reference.md` row added; `docs/stage3_training.md` got a new "Distributed strategy" section explaining the trade-off.

Naming note: `distributed_strategy` (not `trainer_strategy`) was deliberate — keeps visual distance from the existing `--training_strategy` (data strategy: `primary_only` / `combine`). The two flags answer different questions and the names should reflect that at a glance.

## Dry-run CLI provenance fix (also `c3c39aa`)

Same commit, separate concern. The original dry-run feature stashed `_parser` and `_argv` on `args` so `run_dry_run` could detect CLI-set keys; that was reverted earlier as "too complicated" (parser was unpicklable, polluted `args.json`). The replacement defaulted to `sys.argv[1:]` — fine when invoked via the binary entrypoint, broken when tests / `python -c` call `main(args)` with their own argv list.

Restored a *minimal* stash: just `args._argv = argv` (a plain list of strings, picklable, lightweight) and added underscore-key filtering at the three serialization sites that iterate `vars(args)`:

| Site | Filter |
|---|---|
| [src/biom3/core/run_utils.py:227](../../src/biom3/core/run_utils.py#L227) `write_manifest` | `{k: v for k, v in vars(args).items() if not k.startswith("_")}` |
| Stage 1/2/3 `main()` — `args.json` dump | same |
| `core.dry_run.build_args_table` | skip `_`-prefixed keys |

`pickle.dumps(args)` continues to work; no underscore keys leak into manifest, args.json, or dry-run report; provenance attributes correctly under both real-CLI and direct-`main()` paths. Verified by smoke: `--distributed_strategy ddp` shows up as `CLI` in the effective_config table.

## `--skip_devices` conftest option (`bee5a77`)

```bash
pytest tests/ --skip_devices=cpu          # drop cpu variants
pytest tests/ --skip_devices=cpu,xpu      # drop cpu and xpu variants
```

Drops any test variant whose `@pytest.mark.parametrize` values include a string listed in `--skip_devices`, by inspecting `item.callspec.params`. Single conftest change, no test edits needed. Useful on Spark where `test_io.py` and `test_stage3_run_ProteoScribe_sample.py` parametrize over `["cpu", "cuda", "xpu"]` and the cpu variants are slow redundancies when only cuda coverage matters.

Verified: 55 tests went from passing to skipped under `--skip_devices=cpu`; 0 test IDs containing `[cpu` actually ran with the flag set.

## Spark test-failure diagnosis (no code change)

User reported 15 failures in `test_stage3_run_PL_training.py` — all `FileNotFoundError` on tensorboard event files inside `tests/_tmp/outputs/logs/runs/<run_id>/logs/lightning_logs/events.out.tfevents.<ts>...`. Lightning 2.6 catches the deferred async-writer-thread exception in `add_scalar` and rewraps it as the misleading `"you tried to log <int> which is currently not supported"` ValueError; the FileNotFoundError surfaces during the trainer's interrupt-handler call to `logger.finalize`.

Root cause: **two pytest sessions running concurrently against the same checkout.** When process A's `remove_dir(OUTPUTS_DIR)` (called between tests in `test_resume_training` / `test_finetuning` / `test_resume_finetune` / `test_start_phase2_training`) recursively wiped `tests/_tmp/outputs/`, process B's mid-training tensorboard async writer couldn't `io.open(events_file, "ab")` because the parent directory was gone.

Why those tests specifically and not `test_train_from_scratch` / `test_train_from_pretrained_weights`: pure timing. Process B happened to be inside one of the longer training calls when process A's cleanup hit.

User confirmed by re-running serially — all tests now pass. No code change needed.

Test-isolation hardening options were discussed and shelved:
- Per-process `tests/_tmp/outputs/<pid>/` namespacing (cheapest, ~one-line conftest).
- `flock`-based session lock that bails the second concurrent pytest with a clear message (strictest).

Both deferred unless this bites again.

## Files touched in `c3c39aa` (9 files, +100/-16)

- `src/biom3/Stage3/run_PL_training.py` — flag declaration, strategy selection, manifest output, `args._argv` stash, args.json filter
- `src/biom3/core/dry_run.py` — `infer_strategy` reads `args.distributed_strategy`; `run_dry_run` prefers `args._argv` over `sys.argv`; `build_args_table` skips underscore keys
- `src/biom3/core/run_utils.py` — `write_manifest` skips underscore keys
- `src/biom3/Stage{1,2}/run_PL_training.py` — same `args._argv` stash + args.json filter (for parity)
- `tests/core_tests/test_dry_run.py` — three new `infer_strategy` variants for the new flag
- `tests/stage3_tests/test_stage3_dry_run.py` — new ddp-strategy integration test asserting full-replication memory math
- `docs/CLI_reference.md`, `docs/stage3_training.md` — flag row + "Distributed strategy" section

## Files touched in `bee5a77` (1 file)

- `tests/conftest.py` — `--skip_devices` option added to `pytest_addoption`; collection hook drops parametrize variants whose values include a listed device string

## Verification

- **dry-run + core unit tests**: 73/73 pass after the provenance fix and the new `infer_strategy` variants.
- **`pytest tests/ --quick`**: 672 passed, 152 skipped, 0 failures (672 = previous baseline + 2 new tests).
- **Spark full-suite re-run by the user (serially)**: all tests pass.
- **`pickle.dumps(args)`**: still succeeds — DDP/checkpoint-safe.
- **`vars(args)` underscore audit**: zero leaked keys in `args.json`, `dry_run_report.json`, or `build_manifest.json`.
- **`--gpu_devices` deprecated alias** (from the earlier session): still works, still emits the warning.

## Known follow-ups (not done in this session)

- HPC launcher `NGPU` / `NGPU_PER_NODE` shell variables in `scripts/stage{1,2,3}_train_*.sh` still carry the old name internally; only the `--<flag>` they pass to the trainer was renamed in the previous session. Cosmetic, not breaking.
- The user mentioned earlier in the day that `test_stage3_run_PL_training.py` is slow on Spark and asked about speedups (`--epochs 1 --limit_train_batches 5`, swap to `--distributed_strategy ddp` for tests, disable DeepSpeed offload in the test path). All discussed, none applied.
- A real `--distributed_strategy ddp` smoke against `configs/stage3_training/pretrain_scratch_v3.json` on Spark (not just the dry-run preview) is the only remaining thing no automated test can validate. Worth running before declaring DDP fully production-ready.
- The session-start "two items to plan for" list — the `devices_per_node` rename was item 1, `--distributed_strategy` was implicitly item 2 if I'm reading the timeline right. If there's a third, it hasn't surfaced.
