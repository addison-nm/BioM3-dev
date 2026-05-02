"""Hand-run smoke scripts for empirically probing how the distributed /
device system is wired up under a given launcher.

These are NOT pytest tests. Each script is a standalone ``python -m`` /
``python <path>`` entry point meant to be invoked under a launcher, e.g.::

    NGPU_PER_NODE=12 NGPU_TOTAL=24 \\
        ./scripts/launchers/aurora_multinode.sh \\
            python tests/_smoke/env_dump.py

Run them in order — each probes a deeper layer than the last:

  1. env_dump.py     - raw os.environ dump (no biom3 imports)
  2. device_probe.py - what torch sees on the hardware (no biom3 imports)
  3. rank_audit.py   - whether biom3.core._dist_env agrees with reality
  4. pg_probe.py     - end-to-end init_process_group + collectives

Each script prints rank-prefixed single-line records (greppable) and a
sorted summary on rank 0. Stop at the first script that contradicts an
assumption baked into the audit — there's no point running 4 if 1 is
already wrong.
"""
