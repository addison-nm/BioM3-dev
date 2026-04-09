# BioM3-dev — Versioning Maintenance

How and when to bump the `biom3` package version. For the ecosystem-wide picture (all four repos and the cross-repo sync contract), see [BioM3-ecosystem/docs/version_tracking.md](../../docs/version_tracking.md).

## Source of truth

The `biom3` package version is declared in **one place only**:

```
src/biom3/__init__.py
└── __version__ = "0.1.0aN"
```

`pyproject.toml` exposes this value through `setuptools` attribute discovery:

```toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "biom3.__version__"}
```

This means:
- `pip install -e .` reads the version from `__init__.py`.
- `python -c "import biom3; print(biom3.__version__)"` returns the same value.
- There is no separate `VERSION` file. Do not add one — it would create two sources of truth.

## When to bump

While in the `0.x` alpha series:
- **Alpha bump** (`0.1.0a1` → `0.1.0a2`): any change that breaks a public CLI flag, config schema, model interface, or weight format. Bump as soon as the breaking change lands on `main`.
- No bump needed for: bug fixes that preserve interfaces, internal refactors, doc-only changes, test additions.

The `0.x` line is unstable by design. Once interfaces stabilize, we will move to `0.x.0b1` (beta), then `1.0.0`. From `1.0.0` onward, follow standard semver.

## Bump checklist

Run through this list every time you bump:

1. **Edit `src/biom3/__init__.py`** — set `__version__` to the new value.
2. **Reinstall and verify:**
   ```bash
   pip install -e .
   python -c "import biom3; print(biom3.__version__)"
   ```
   Confirm the printed version matches what you just set.
3. **Commit** with a release-style message:
   ```bash
   git add src/biom3/__init__.py
   git commit -m "chore(release): bump to v0.1.0aN"
   ```
4. **Tag the commit:**
   ```bash
   git tag v0.1.0aN
   ```
   The tag must point at the commit that contains the matching `__version__`.
5. **Push (after review):**
   ```bash
   git push origin main
   git push origin v0.1.0aN
   ```
6. **Refresh dependent repos' SYNC_LOG files.** BioM3-dev itself does not have a SYNC_LOG, but every dependent repo does. After the new tag is pushed:
   - In `BioM3-workflow-demo`, `BioM3-workspace-template`, and `BioM3-data-share`, pull the latest BioM3-dev, run a smoke test, then add a new row to that repo's `SYNC_LOG.md` pairing the new BioM3-dev hash with the dependent repo's commit hash.
   - See [BioM3-ecosystem/docs/version_tracking.md §How to keep things in sync](../../docs/version_tracking.md#how-to-keep-things-in-sync) for the full ritual.
7. **Verify with `/version-check`** (from the ecosystem root). All rows should report `OK` after the dust settles.

## Recovering from drift

**Symptom:** the highest git tag does not match `__version__`, or a tag exists at a commit that does not contain the matching `__version__`.

**Real example (April 2026):** the `v0.1.0a2` tag existed at commit `5d05460`, but `src/biom3/__init__.py` still said `0.1.0a1` for two further commits. The fix was to bump `__version__` to `0.1.0a2` (without retagging), accept the temporary HEAD-vs-tag drift, and address the drift at the next natural bump.

**Lesson:** the `__version__` bump and the `git tag` must happen on the same commit. If you tag without bumping (or bump without tagging), `/version-check` will flag the inconsistency on the next audit, and you will have to choose one of these recoveries:

| Drift type | Recovery |
|---|---|
| Tag exists, `__version__` is older | Bump `__version__` to match the tag, commit, push. Do *not* retag. The next bump will be `0.1.0a(N+1)` and will tag a fresh commit. |
| `__version__` is newer than the highest tag | Tag the commit where `__version__` was bumped (use `git log --follow src/biom3/__init__.py` to find it), then push the tag. |
| Tag points at the wrong commit | Almost always safer to leave the wrong tag in place and bump to a new version, rather than rewriting tag history (which breaks anyone who already pulled the bad tag). |

The general rule: **never delete a tag that has been pushed**. Pushed tags are public history. Move forward with a new bump instead.

## See also

- [BioM3-ecosystem/docs/version_tracking.md](../../docs/version_tracking.md) — ecosystem-wide versioning conventions.
- `pyproject.toml` — where the dynamic version is declared.
- `src/biom3/__init__.py` — the single source of truth.
