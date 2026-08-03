"""Multidomain protein finetuning for ProteoScribe.

A K-domain protein is decoded on K parallel fixed-length canvases, one per domain,
each driven by its own single-family expert decoder, with a trainable cross-domain
coupling letting the canvases condition on each other. The coupling starts at an
additive null, so the composed decoder begins bit-exactly equal to the independent
experts.

This subpackage is self-contained: it reads from stable public surfaces elsewhere
in :mod:`biom3.Stage3` but does not modify them, and the single-domain finetuning
path is free to evolve independently.
"""

from biom3.Stage3.multidomain.audit import (
    AuditFailure,
    assert_additive_null,
    audit_trainable_parameters,
    enforce_audit,
    expert_delta_norms,
)
# Imported for its @register_compose side effect: a record_schema naming
# "map_domains" resolves through the registry, which is only populated once this
# module has been imported.
from biom3.Stage3.multidomain.compose import map_domains
from biom3.Stage3.multidomain.coupling import AllPairsCoupling
from biom3.Stage3.multidomain.data import MultiDomainDataModule
from biom3.Stage3.multidomain.io import (
    MultiDomainSpec,
    build_from_spec,
    build_multidomain_from_checkpoint,
    load_composed_state_dict,
    load_experts,
    state_dict_fingerprint,
)
from biom3.Stage3.multidomain.model import (
    MultiDomainProteoScribe,
    build_multidomain_model,
)
from biom3.Stage3.multidomain.preprocess import make_multidomain_collate_fn

__all__ = [
    "AllPairsCoupling",
    "AuditFailure",
    "MultiDomainDataModule",
    "MultiDomainProteoScribe",
    "MultiDomainSpec",
    "assert_additive_null",
    "audit_trainable_parameters",
    "build_from_spec",
    "build_multidomain_from_checkpoint",
    "build_multidomain_model",
    "enforce_audit",
    "expert_delta_norms",
    "load_composed_state_dict",
    "load_experts",
    "make_multidomain_collate_fn",
    "map_domains",
    "state_dict_fingerprint",
]
