from __future__ import annotations

_ESMFOLD_MODEL = None


def _get_esmfold_model(device: str | None = None):
    """Lazy-load and cache the ESMFold model."""
    global _ESMFOLD_MODEL
    if _ESMFOLD_MODEL is not None:
        return _ESMFOLD_MODEL

    try:
        import esm
    except ImportError:
        raise ImportError(
            "ESMFold requires fair-esm: pip install fair-esm"
        )
    try:
        import omegaconf  # noqa: F401
    except ImportError:
        raise ImportError(
            "ESMFold requires omegaconf: pip install omegaconf"
        )

    if device is None:
        from biom3.backend.device import get_device
        device = str(get_device())

    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    _ESMFOLD_MODEL = model
    return model


def fold_sequence(
    sequence: str,
    device: str | None = None,
    model=None,
) -> str:
    """Fold a protein sequence into a PDB string using ESMFold.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes, no special tokens).
    device : str, optional
        Torch device. Defaults to biom3.backend.device.get_device().
    model : optional
        Pre-loaded ESMFold model. If None, loads and caches internally.

    Returns
    -------
    str
        PDB-format string of the predicted structure.
    """
    if model is None:
        model = _get_esmfold_model(device)
    with __import__("torch").no_grad():
        pdb_str = model.infer_pdb(sequence)
    return pdb_str


def fold_sequences(
    sequences: list[str],
    device: str | None = None,
    model=None,
) -> list[str]:
    """Fold multiple sequences, returning a list of PDB strings."""
    if model is None:
        model = _get_esmfold_model(device)
    return [fold_sequence(seq, model=model) for seq in sequences]
