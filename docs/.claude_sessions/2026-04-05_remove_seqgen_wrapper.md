# 2026-04-05 — Remove seqgen_wrapper.sh

## Summary

Removed the thin `scripts/seqgen_wrapper.sh` bash wrapper and updated all
callers to invoke `biom3_ProteoScribe_sample` directly. This aligns BioM3-dev
demos with the pattern already used in BioM3-workflow-demo and
`demos/animate_generation.sh`.

**Pre-session state:** `git checkout 0d198a8`

## Motivation

`seqgen_wrapper.sh` mapped 5 positional arguments to `biom3_ProteoScribe_sample`
CLI flags with no additional logic beyond `mkdir -p`. It obscured the actual
command, had unresolved TODOs, and was not referenced in README, tests, or job
templates. BioM3-workflow-demo already calls the entry point directly in
`pipeline/03_generate.sh`.

## Changes

### Deleted
- `scripts/seqgen_wrapper.sh` — removed entirely.

### Updated
- `demos/SH3/run_gen_seqs_SH3.sh` — replaced wrapper call with direct
  `biom3_ProteoScribe_sample` invocation using long-form flags. Added
  `set -euo pipefail` and `mkdir -p`.
- `demos/SH3/run_gen_seqs_SH3_prompts.sh` — same treatment; loop preserved.
- `docs/sequence_generation_strategies.md` — updated all 5 CLI examples
  (lines 126–157) from short flags (`-i`, `-c`, `-m`, `-o`) to long-form
  (`--input_path`, `--config_path`, `--model_path`, `--output_path`) for
  consistency with README.md.

### Not changed
- `_misc/run_gen_seqs_CM.sh` and `_misc/run_gen_seqs_sample_prompts.sh` still
  reference the deleted wrapper. These are legacy scripts in `_misc/` and were
  intentionally left as-is.
- `docs/.claude_sessions/2026-04-05_job_scripts_and_template_hardening.md`
  mentions the wrapper in a historical context — no update needed.
- `README.md` already used long-form flags in all examples and documents both
  short and long forms in the reference table.

## Notes

- The short flags (`-i`, `-c`, `-m`, `-o`) still work — they are argparse
  aliases defined in the entry point. The docs update is for consistency, not
  correctness.
