import pytest
from unittest.mock import patch, MagicMock
import sys

import biom3.viz.folding as folding_mod


class TestFoldSequenceImportErrors:
    def test_missing_esm_raises(self):
        folding_mod._ESMFOLD_MODEL = None
        with patch.dict(sys.modules, {"esm": None}):
            with pytest.raises(ImportError, match="fair-esm"):
                folding_mod._get_esmfold_model(device="cpu")

    def test_missing_omegaconf_raises(self):
        folding_mod._ESMFOLD_MODEL = None
        mock_esm = MagicMock()
        with patch.dict(sys.modules, {"esm": mock_esm, "omegaconf": None}):
            with pytest.raises(ImportError, match="omegaconf"):
                folding_mod._get_esmfold_model(device="cpu")


class TestModelCaching:
    def test_cached_model_reused(self):
        sentinel = MagicMock()
        folding_mod._ESMFOLD_MODEL = sentinel
        result = folding_mod._get_esmfold_model()
        assert result is sentinel
        folding_mod._ESMFOLD_MODEL = None  # cleanup


@pytest.mark.use_gpu
class TestFoldSequenceGPU:
    def test_fold_returns_pdb_string(self):
        pdb = folding_mod.fold_sequence("AAAAAA")
        assert isinstance(pdb, str)
        assert "ATOM" in pdb
