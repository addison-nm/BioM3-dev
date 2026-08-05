# Session: Porting the IPEX Gate Fix Upstream, Pinning the Fork, Revalidating

**Date:** 2026-08-05
**Branch:** addison-dev
**Commit:** `0f6786f` — `docs: fix Aurora container weights/data paths and sif scratch dirs`
**Pre-session state:** `git checkout f3a47f4` (`docs: both Aurora images pass the test suite`)
**Companion note:** [2026-08-05_aurora_containers_single_and_multinode.md](./2026-08-05_aurora_containers_single_and_multinode.md)
— this session continues directly from it.

## Goal

`Dockerfile.xpu-oneapi` was carrying a `sed` patch that rewrote the installed lightning
package at build time, to remove an `intel_extension_for_pytorch` gate in `XPUAccelerator`
that has no public torch-2.10 counterpart. The question raised at the end of the previous
session — *is that patch worth porting into the fork itself?* — was answered yes, and this
session did it, then dealt with the consequences.

## What changed

### 1. The gate is gone upstream

`addison-nm/lightning` PR #1, merge commit `5116b36e8`, removes `_IPEX_AVAILABLE` and its
`RequirementCache` import from **both** accelerator copies:

- `src/lightning/pytorch/accelerators/xpu.py`
- `src/lightning/fabric/accelerators/xpu.py`

Both copies matter. The pytorch copy's `__init__` was what raised; the fabric copy's
`num_xpu_devices()` is what `device_parser` consults, and a mismatch between them produced
the earlier `You requested gpu: [0..11] But your machine only has: []`.

`is_available()` and `num_xpu_devices()` now defer to `torch.xpu` directly, which is the
whole point: native `torch.xpu` has not needed IPEX for some time, and the gate was
stranding the accelerator on every torch the oneapi image can actually install.

### 2. The Dockerfile patch became an assertion

The `sed` layer in `Dockerfile.xpu-oneapi` is replaced by a check that the gate has not
come back:

```dockerfile
RUN python -c "\
from lightning.pytorch.accelerators.xpu import XPUAccelerator; \
from lightning.fabric.accelerators.xpu import num_xpu_devices; \
XPUAccelerator(); \
print('XPUAccelerator constructs without IPEX; fabric devices:', num_xpu_devices())"
```

A returning gate would otherwise strand the accelerator silently at runtime rather than at
build time.

### 3. The fork is pinned by commit

```dockerfile
ARG LIGHTNING_REF=5116b36e81d6be8d4c466179f3a47dd528d95c0f
RUN pip install --no-build-isolation \
        "git+https://github.com/addison-nm/lightning.git@${LIGHTNING_REF}"
```

Applied to **both** xpu Dockerfiles. See *The cache lesson* below for why this is not
merely hygiene.

### 4. Comment corrections

Four places claimed the gate was "patched out in the Dockerfile" and now name the upstream
fix instead: `requirements/aurora-oneapi.txt`, `requirements/constraints-oneapi.txt`,
`docker/Dockerfile.xpu-oneapi`, and — a substantive one — `docker/Dockerfile.xpu`, which
asserted IPEX *was required by the fork*. That stopped being true at the merge. IPEX stays
in that image only because it pairs with the torch 2.8 pin the image was validated against.

## The cache lesson

The first rebuild after the merge failed with the *old* error:

```
File ".../lightning/pytorch/accelerators/xpu.py", line 30, in __init__
    raise ModuleNotFoundError(str(_IPEX_AVAILABLE))
```

on a symbol that no longer exists upstream. The build log showed why:

```
=> CACHED [ 9/21] RUN pip install --no-build-isolation "git+https://github.com/addison-nm/lightning.git"
```

An unpinned branch URL gives Docker no cache key that changes when the branch moves, so the
layer kept serving the pre-merge package indefinitely. Nothing else in the file had changed
above it, so nothing invalidated it. **The pin is what fixes this** — `ARG LIGHTNING_REF`
changes the layer's command string, and the reinstall follows.

Generalization worth carrying: any `pip install git+…` without a ref is both
unreproducible *and* effectively permanently cached. The failure mode is not a stale build
that looks stale — it is a build that looks correct and contains old code.

## Revalidation on Aurora

Image `ghcr.io/natural-machine/biom3:xpu-oneapi-cae7e77`, converted to
`biom3_xpu-oneapi-cae7e77.sif`.

| Check | Result |
| --- | --- |
| Test suite | 1134 passed / 162 skipped — matches the pre-merge baseline exactly |
| Rank bootstrap | `GLOBAL_RANK: 0…23`, 24/24 members |
| Throughput, 24 ranks / 2 nodes / cxi | 288 batches in 7m26s = **0.65 it/s, ~496 samples/s** |

The throughput matches the pre-merge CXI figure (0.64 / ~492), so the merged fork behaves
identically to the sed-patched build, and the run was on Slingshot rather than the tcp
fallback.

Expected noise in that log, recorded so it is not re-investigated: `CCL_WARN| MPI was
initialized externally` is the `CCL_ATL_TRANSPORT=mpi` path working as designed;
`could not get local_idx/count from environment variables` resolves via ATL; the
`ProcessGroupXCCL` device warnings are the known `device_id` item; `gocryptfs not found` is
irrelevant.

## Documentation errors this session exposed

Both were in docs that had been *written* the previous session and never *followed* from
scratch.

**Bad bind paths.** `setup_aurora_container.md`'s single-node example used
`BIOM3_WEIGHTS_DIR=/flare/NLDesignProtein/sharepoint/BioM3-data-share/weights`, taken from
the ecosystem `CLAUDE.md` machine table. That path is not populated. Because these are
**apptainer bind sources**, all 24 ranks died at container creation —
`mount source ... doesn't exist` — before any Python ran. The repo's own `weights/` and
`data/` are what work, which is what the multi-node example three sections lower already
said. Fixed, with a note that these are bind sources.

*Still open:* whether `/flare/NLDesignProtein/sharepoint/BioM3-data-share` exists at all
decides whether the ecosystem `CLAUDE.md` Aurora row is wrong, or whether it is a real
share this repo simply doesn't use.

**Apptainer scratch dirs.** The doc told you to put `APPTAINER_CACHEDIR` *and*
`APPTAINER_TMPDIR` on `/flare`. They want opposite things: the cache benefits from Lustre's
persistence across login nodes, but the unpack is millions of small creates/chowns/setxattrs
— the access pattern Lustre handles worst. One layer of the oneAPI base took ~13 minutes.
Doc now splits them: cache on `/flare`, tmp on node-local disk.

## Commits

| | |
| --- | --- |
| `5405989` | `chore: drop the lightning IPEX patch, now fixed upstream` |
| `cae7e77` | `chore: pin the lightning fork to a commit in both xpu images` |
| `6e08f2d` | `docs: drop a stale pointer to the bare-metal multi-node path` |
| `0f6786f` | `docs: fix Aurora container weights/data paths and sif scratch dirs` |

`6e08f2d` and `0f6786f` were unpushed at session end.

Upstream: `addison-nm/lightning` `5116b36e8` (PR #1), on `origin/master`.

## Open items

- **Ecosystem `CLAUDE.md` Aurora path** — pending `ls -d /flare/NLDesignProtein/sharepoint/BioM3-data-share`.
- **Converge `Dockerfile.xpu` on torch 2.10.** Its pin cascade — IPEX, `constraints-xpu.txt`,
  `dpctl==0.19.0` — now exists to satisfy a constraint that no longer applies. Collapsing it
  puts both images on Aurora's actual versions. Invalidates that image's 1134-test
  validation, so it needs its own build-and-test cycle. Not started; not yet authorized.
- **Whether `xpu` should survive at all.** `xpu-oneapi` passes the same suite and runs
  single-node fine. Converging `Dockerfile.xpu` on 2.10 would *not* give it multi-node — it
  still has no Intel MPI. So the real choice is two images or one, and it is a question about
  what to keep validated rather than a technical constraint.
- **`xpu` variant not rebuilt at `cae7e77`.** The pin commit changed `Dockerfile.xpu` too;
  its published image and `.sif` are still from `6d1497e`.
- **`.sif` housekeeping.** `/flare/NLDesignProtein/.biom3-compute-endpoint/containers/` held
  seven images from intermediate commits. **Nothing in this repo references that directory** —
  the Aurora wrappers default to `./biom3_xpu.sif` and take `BIOM3_SIF` explicitly — so
  whatever populates and consumes it is managed elsewhere, and it sits at the shared project
  root rather than under `$USER`. Decision was to keep only `biom3_xpu-oneapi-cae7e77.sif`.
  Note that this leaves the single-node variant with no image there, and that the unsuffixed
  `biom3_xpu-dev.sif` / `biom3_xpu-oneapi.sif` names are the ones most likely to be resolved
  by a service.
- **Carried forward, unchanged:** `device_id` on `init_process_group`; `CCL_WORKER_AFFINITY`
  assuming a 12-rank layout; re-measuring single-node container throughput at ~step 500.

## Process note

Three of this session's failures were mine and all had the same shape: acting on a written
record without checking it against the system. The bad bind path came from the ecosystem
`CLAUDE.md` table; the `APPTAINER_TMPDIR` advice came from our own setup doc; both were
repeated confidently and both cost a full run. The previous session's note warned that
*host values wrong inside the container* was the dominant failure mode — the analogue here
is **documented values that were never executed end-to-end**. A doc example that has not
been run from a clean state is a hypothesis, not a record.

The build cache failure is the more interesting one, because the artifact was wrong while
every visible signal said it was right. What caught it was the assertion layer failing
loudly at build time — which is an argument for keeping such checks even when they look
redundant.
