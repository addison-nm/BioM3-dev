"""Thin re-export so Stage 2 callers don't reach across stages."""

from biom3.Stage1.PL_wrapper import PL_Facilitator

__all__ = ["PL_Facilitator"]
