"""Thin re-exports so Stage 2 callers don't reach across stages."""

from biom3.Stage1.preprocess import (
    Facilitator_Dataset,
    Facilitator_DataModule,
)

__all__ = ["Facilitator_Dataset", "Facilitator_DataModule"]
