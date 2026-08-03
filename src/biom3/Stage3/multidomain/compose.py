"""Record-schema support for multidomain records.

A multidomain record nests one sub-record per domain::

    {"accession": ..., "sequence": <full-length>, "source": ...,
     "domains": [{"pfam_id": ..., "sequence": ..., "region": [s, e],
                  "fields": {...}}, ...]}

:func:`map_domains` applies an ordinary output spec to each of those
sub-records in turn, so per-domain captions are composed by the *existing*
registered compose functions with no multidomain-specific caption logic. Each
domain draws its own dropout and shuffle because the shared ``rng`` is threaded
through, which is what makes per-epoch caption re-composition vary per domain.

Registering from here rather than from ``core.dataloaders`` keeps the
multidomain path self-contained; ``register_compose`` is public API, so the
registry extends without touching shared code.
"""

import json
import random

from biom3.core.dataloaders import register_compose

# Private, but importing it is strictly better than restating spec-compilation
# semantics here: a second implementation would drift from the schema contract
# that GeneralizedRecordDataset enforces.
from biom3.core.dataloaders.generalized_dataloader import _build_resolver

_RESOLVER_CACHE = {}


def _resolver_for(spec):
    """Compile (and cache) an output spec into a ``resolver(obj, rng)``."""
    try:
        key = json.dumps(spec, sort_keys=True)
    except TypeError:
        # Callable targets are not serializable; compiling is cheap enough.
        return _build_resolver(spec)
    resolver = _RESOLVER_CACHE.get(key)
    if resolver is None:
        resolver = _build_resolver(spec)
        _RESOLVER_CACHE[key] = resolver
    return resolver


@register_compose("map_domains")
def map_domains(obj, args=None, rng=random):
    """Apply an output spec to each domain sub-record, preserving N->C order.

    args keys:
        output        (required) the spec to apply per domain — any form
                      GeneralizedRecordDataset accepts, e.g. {"from": "sequence"}
                      or {"compose": "fields_to_caption", "args": {...}}
        domains_key   record key holding the domain list (default "domains")
        expect_k      require exactly this many domains; raises otherwise

    Returns:
        list of the inner spec's value, one per domain, in record order
    """
    args = args or {}
    if "output" not in args:
        raise ValueError("map_domains requires an 'output' spec in args")

    domains_key = args.get("domains_key", "domains")
    try:
        domains = obj[domains_key]
    except KeyError:
        raise KeyError(
            f"record has no {domains_key!r} key; multidomain records must nest "
            "one sub-record per domain"
        ) from None
    if not isinstance(domains, (list, tuple)):
        raise TypeError(
            f"{domains_key!r} must be a list of domain sub-records, got "
            f"{type(domains).__name__}"
        )

    expect_k = args.get("expect_k")
    if expect_k is not None and len(domains) != expect_k:
        raise ValueError(
            f"record has {len(domains)} domains but expect_k={expect_k}; every "
            "record in a multidomain run must carry the same number of domains"
        )

    resolver = _resolver_for(args["output"])
    return [resolver(domain, rng) for domain in domains]
