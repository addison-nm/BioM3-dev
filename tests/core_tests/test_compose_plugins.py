"""Tests for external compose-function plugins (Option A loader) and the
example ``pencl_caption`` plugin that ships under ``caption_plugins/``.
"""

import random
from pathlib import Path

import pytest

from biom3.core.dataloaders import compose_functions as cf
from biom3.core.dataloaders import load_compose_plugins

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PATH = REPO_ROOT / "caption_plugins" / "pencl_caption.py"

# A Pfam-style record carrying labels that are out-of-vocabulary for PenCL
# (gene_ontology, family_name, family_description) alongside in-vocab fields.
PFAM_RECORD = {
    "sequence": "MRSLAILTTLLAGHAFA",
    "source": "pfam",
    "fields": {
        "family_name": "SH3 domain",
        "family_description": "SH3 (Src homology 3) domains mediate signalling.",
        "protein_name": "Neutrophil cytosolic factor 2",
        "gene_ontology": "cytoplasm, NADPH oxidase complex, phagocytosis",
        "similarity": "Belongs to the NCF2 family",
        "lineage": "Eukaryota, Metazoa, Chordata, Craniata, Vertebrata",
    },
}

SUPPLEMENTAL_RECORD = {
    "sequence": "MSLEQKKGADII",
    "source": "supplemental",
    "fields": {
        "protein_name": "SH3 domain",
        "lineage": "cellular organisms; Eukaryota; Opisthokonta; Fungi",
        "sh3_paralog_name": "SLA1",
        "paralog_function": "Cytoskeletal protein binding protein",
    },
}


@pytest.fixture(scope="module")
def pencl_plugin_loaded():
    assert PLUGIN_PATH.is_file(), f"missing example plugin at {PLUGIN_PATH}"
    load_compose_plugins([str(PLUGIN_PATH)])
    return cf.get_compose_function("pencl_caption")


class TestLoader:

    def test_plugin_registers_function(self, pencl_plugin_loaded):
        assert "pencl_caption" in cf.list_compose_functions()

    def test_load_is_idempotent(self, pencl_plugin_loaded):
        # A second load of the same file must not re-run the module body and
        # trigger a duplicate-registration ValueError.
        load_compose_plugins([str(PLUGIN_PATH)])
        assert "pencl_caption" in cf.list_compose_functions()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_compose_plugins(["./caption_plugins/does_not_exist.py"])


class TestPenCLCaption:

    def test_in_vocab_labels_only(self, pencl_plugin_loaded):
        cap = pencl_plugin_loaded(PFAM_RECORD, {}, random.Random(0))
        assert "GENE ONTOLOGY" not in cap
        assert "FAMILY DESCRIPTION" not in cap
        assert "FAMILY NAME:" not in cap  # singular is OOD for PenCL

    def test_family_fusion(self, pencl_plugin_loaded):
        cap = pencl_plugin_loaded(PFAM_RECORD, {}, random.Random(0))
        assert "FAMILY NAMES: Family names are SH3 domain" in cap

    def test_lineage_natural_language(self, pencl_plugin_loaded):
        cap = pencl_plugin_loaded(PFAM_RECORD, {}, random.Random(0))
        assert "LINEAGE: The organism lineage is Eukaryota, Metazoa" in cap

    def test_lineage_drops_cellular_organisms_root(self, pencl_plugin_loaded):
        cap = pencl_plugin_loaded(SUPPLEMENTAL_RECORD, {}, random.Random(0))
        assert "The organism lineage is Eukaryota, Opisthokonta, Fungi" in cap
        assert "cellular organisms" not in cap

    def test_protein_name_leads_and_family_last(self, pencl_plugin_loaded):
        cap = pencl_plugin_loaded(PFAM_RECORD, {}, random.Random(0))
        assert cap.startswith("PROTEIN NAME: Neutrophil cytosolic factor 2")
        assert cap.index("PROTEIN NAME") < cap.index("LINEAGE") < cap.index("FAMILY NAMES")

    def test_dropout_can_remove_optional_fields(self, pencl_plugin_loaded):
        # protein_name is protected (0.0); similarity forced out (1.0).
        args = {"dropout_rates": {"protein_name": 0.0, "similarity": 1.0}}
        cap = pencl_plugin_loaded(PFAM_RECORD, args, random.Random(1))
        assert "PROTEIN NAME" in cap
        assert "SIMILARITY" not in cap

    def test_max_item_chars_drops_overlength_field(self, pencl_plugin_loaded):
        record = {
            "fields": {
                "protein_name": "Kinase",
                "function": "x" * 5000,  # over any sane per-field cap
                "similarity": "Belongs to the SRC family",
            }
        }
        cap = pencl_plugin_loaded(record, {"max_item_chars": 585}, random.Random(0))
        assert "PROTEIN NAME: Kinase" in cap
        assert "FUNCTION" not in cap  # dropped for length, tail preserved
        assert "SIMILARITY: Belongs to the SRC family" in cap

    def test_protein_name_exempt_from_cap(self, pencl_plugin_loaded):
        record = {"fields": {"protein_name": "N" * 1000}}
        cap = pencl_plugin_loaded(record, {"max_item_chars": 100}, random.Random(0))
        assert cap.startswith("PROTEIN NAME: " + "N" * 1000)
