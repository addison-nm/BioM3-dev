#!/usr/bin/env python3
"""Verify a BioM3 weights bundle against its MANIFEST.json.

Deliberately stdlib-only and importable without torch or biom3, so a freshly
pulled bundle can be checked before anything is installed.

Usage:

    python3 scripts/weights_bundle/verify_bundle.py <bundle_dir>
    python3 scripts/weights_bundle/verify_bundle.py <bundle_dir> --quick
"""

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path, chunk_size=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle_dir, quick=False):
    """Return a list of human-readable problems; empty means the bundle is good."""
    manifest_path = os.path.join(bundle_dir, "MANIFEST.json")
    if not os.path.isfile(manifest_path):
        return [f"{manifest_path}: missing"]

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    problems = []
    listed = set()
    for record in manifest.get("files", []):
        rel = record["path"]
        listed.add(rel)
        path = os.path.join(bundle_dir, rel)
        if not os.path.isfile(path):
            problems.append(f"{rel}: missing")
            continue
        size = os.path.getsize(path)
        if size != record["bytes"]:
            problems.append(f"{rel}: {size} bytes, manifest says {record['bytes']}")
            continue
        if quick:
            continue
        digest = sha256_file(path)
        if digest != record["sha256"]:
            problems.append(
                f"{rel}: sha256 {digest}, manifest says {record['sha256']}"
            )

    for root, _, names in os.walk(bundle_dir):
        for name in names:
            rel = os.path.relpath(os.path.join(root, name), bundle_dir)
            if rel != "MANIFEST.json" and rel not in listed:
                problems.append(f"{rel}: present but not in manifest")

    if manifest.get("incomplete"):
        problems.append(
            "manifest marks this bundle incomplete (built with --skip_llms); "
            "Stage 1 will not run"
        )
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify a BioM3 weights bundle against its MANIFEST.json."
    )
    parser.add_argument("bundle_dir", type=str)
    parser.add_argument("--quick", action="store_true",
                        help="Check presence and size only, skipping sha256.")
    args = parser.parse_args(argv)

    problems = verify_bundle(args.bundle_dir, quick=args.quick)
    if problems:
        print(f"FAILED: {len(problems)} problem(s) in {args.bundle_dir}", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    mode = "size" if args.quick else "sha256"
    print(f"OK: {args.bundle_dir} verified ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
