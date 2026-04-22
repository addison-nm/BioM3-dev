"""Stage 3 (ProteoScribe) benchmarks.

- ``generation``: parameter sweep over (token_strategy, N, P, B), measuring
  per-step time and peak memory.
- ``training``: planned — per-epoch / per-step training timing already
  exists as ``Stage3.callbacks.TrainingBenchmarkCallback`` and can be
  wrapped here once we want a CLI sweep harness for it.
"""
