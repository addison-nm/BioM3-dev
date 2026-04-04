import os
import pytest
import py3Dmol

from biom3.viz.viewer import _read_pdb, view_pdb, view_overlay, highlight_residues, color_by_values, to_html, save_html


TMPDIR = "tests/_tmp"


class TestReadPdb:
    def test_read_from_file(self, mini_pdb_file):
        content = _read_pdb(mini_pdb_file)
        assert "ATOM" in content
        assert "ALA" in content

    def test_read_from_string(self, mini_pdb):
        content = _read_pdb(mini_pdb)
        assert content == mini_pdb


class TestViewPdb:
    def test_returns_view(self, mini_pdb):
        v = view_pdb(mini_pdb)
        assert isinstance(v, py3Dmol.view)

    def test_from_file(self, mini_pdb_file):
        v = view_pdb(mini_pdb_file)
        assert isinstance(v, py3Dmol.view)


class TestViewOverlay:
    def test_overlay_two_structures(self, mini_pdb, mini_pdb_shifted):
        v = view_overlay([mini_pdb, mini_pdb_shifted], labels=["A", "B"])
        assert isinstance(v, py3Dmol.view)


class TestHighlightResidues:
    def test_highlight(self, mini_pdb):
        v = view_pdb(mini_pdb)
        result = highlight_residues(v, [1, 3, 5], color="yellow")
        assert result is v


class TestColorByValues:
    def test_color_by_values(self, mini_pdb):
        v = view_pdb(mini_pdb)
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = color_by_values(v, values)
        assert result is v

    def test_color_with_nans(self, mini_pdb):
        import math
        v = view_pdb(mini_pdb)
        values = [float("nan"), 0.5, float("nan"), 0.8, 1.0]
        result = color_by_values(v, values)
        assert result is v


class TestHtmlExport:
    def test_to_html_contains_script(self, mini_pdb):
        v = view_pdb(mini_pdb)
        html = to_html(v, title="Test")
        assert "3Dmol" in html
        assert "<title>Test</title>" in html
        assert "<!DOCTYPE html>" in html

    def test_save_html(self, mini_pdb):
        v = view_pdb(mini_pdb)
        os.makedirs(TMPDIR, exist_ok=True)
        out_path = os.path.join(TMPDIR, "test_viewer.html")
        save_html(v, out_path, title="Saved Test")
        assert os.path.isfile(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "3Dmol" in content
