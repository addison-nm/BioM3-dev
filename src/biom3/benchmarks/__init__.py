"""Benchmark harnesses for biom3.

Organized by stage to mirror ``src/biom3/Stage{1,2,3}/``. Each benchmark
records a ``benchmark_type`` in its ``env.json`` so downstream tools
(plots, comparison reports) can dispatch on it.

Implemented:
- ``Stage3.generation`` — parameter sweep for ProteoScribe sampling.
  CLI: ``biom3_benchmark_stage3_generation``.

Planned:
- ``Stage3.training``  — per-epoch / per-step training timing (the
  ``TrainingBenchmarkCallback`` already exists in
  ``biom3.Stage3.callbacks``; a sweep harness would live here).
- ``Stage1.inference``, ``Stage2.inference`` — per-stage embedding cost.
"""
