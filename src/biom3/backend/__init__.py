"""Hardware-backend dispatch for BioM3.

Public entry point is ``biom3.backend.device`` — it detects the active
backend at import time and re-exports the per-backend symbols
(``DIST_BACKEND``, ``resolve_device_for_local_rank``, etc.) into its own
namespace via ``from .{cpu,cuda,xpu} import *``.
"""
