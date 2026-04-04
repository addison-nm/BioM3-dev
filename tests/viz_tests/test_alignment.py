import pytest
import numpy as np

from biom3.viz.alignment import superimpose, superimpose_and_view, AlignmentResult


class TestSuperimpose:
    def test_identical_structures_zero_rmsd(self, mini_pdb):
        result = superimpose(mini_pdb, mini_pdb)
        assert isinstance(result, AlignmentResult)
        assert result.rmsd == pytest.approx(0.0, abs=1e-6)
        assert result.n_atoms == 5

    def test_shifted_structures_positive_rmsd(self, mini_pdb, mini_pdb_shifted):
        result = superimpose(mini_pdb, mini_pdb_shifted)
        assert isinstance(result, AlignmentResult)
        # After optimal superimposition, RMSD should be very small
        # (pure translation is perfectly correctable)
        assert result.rmsd == pytest.approx(0.0, abs=1e-3)
        assert result.n_atoms == 5

    def test_result_fields(self, mini_pdb, mini_pdb_shifted):
        result = superimpose(mini_pdb, mini_pdb_shifted)
        assert result.rotation.shape == (3, 3)
        assert result.translation.shape == (3,)
        assert "ATOM" in result.fixed_pdb
        assert "ATOM" in result.moving_pdb


class TestSuperimposeAndView:
    def test_returns_view_and_result(self, mini_pdb, mini_pdb_shifted):
        import py3Dmol
        view, result = superimpose_and_view(mini_pdb, mini_pdb_shifted)
        assert isinstance(view, py3Dmol.view)
        assert isinstance(result, AlignmentResult)


class TestBlastSequence:
    @pytest.mark.network
    def test_blast_returns_results(self):
        from biom3.viz.alignment import blast_sequence
        results = blast_sequence("MKTLLILAVL", max_hits=3)
        assert isinstance(results, list)
        if results:
            assert hasattr(results[0], "percent_identity")
