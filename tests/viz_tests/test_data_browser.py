"""Tests for the app's data-directory browser helper.

The webapp's file picker is used by every page that reads from disk;
``list_files`` must follow symlinked directories because
``scripts/sync_weights.sh`` populates ``weights/`` with per-entry symlinks
into a shared data directory (and worktree workflows depend on it).
"""

from pathlib import Path

from biom3.app._data_browser import list_files


def test_list_files_follows_directory_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "inner").mkdir()
    (real / "inner" / "a.pt").write_bytes(b"x")
    (real / "inner" / "b.txt").write_text("hi")

    browse_root = tmp_path / "browse"
    browse_root.mkdir()
    (browse_root / "linked").symlink_to(real, target_is_directory=True)

    files = list_files(browse_root, extensions=[".pt"], recursive=True)
    names = sorted(f.name for f in files)
    assert names == ["a.pt"]


def test_list_files_recursive_includes_nested(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    (root / "top.pt").write_bytes(b"x")
    (root / "a" / "mid.pt").write_bytes(b"x")
    (root / "a" / "b" / "deep.pt").write_bytes(b"x")

    files = list_files(root, extensions=[".pt"], recursive=True)
    assert sorted(f.name for f in files) == ["deep.pt", "mid.pt", "top.pt"]


def test_list_files_non_recursive_root_only(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "top.pt").write_bytes(b"x")
    (root / "a" / "nested.pt").write_bytes(b"x")

    files = list_files(root, extensions=[".pt"], recursive=False)
    assert [f.name for f in files] == ["top.pt"]


def test_list_files_extensions_filter(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.pt").write_bytes(b"x")
    (root / "b.txt").write_text("x")
    (root / "c.PDB").write_text("x")

    pts = list_files(root, extensions=[".pt"])
    assert [f.name for f in pts] == ["a.pt"]

    pdbs = list_files(root, extensions=[".pdb"])
    assert [f.name for f in pdbs] == ["c.PDB"]


def test_list_files_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert list_files(tmp_path / "does_not_exist") == []
