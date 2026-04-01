"""Tests for enrich module: annotation columns and caption composition."""

import pandas as pd
import pytest

from biom3.dbio.enrich import (
    ANNOTATION_FIELDS,
    ANNOTATION_COLUMNS,
    enrich_dataframe,
    compose_caption,
    extract_annotations,
)


MOCK_UNIPROT_ENTRY = {
    "primaryAccession": "A0A001",
    "proteinDescription": {
        "recommendedName": {
            "fullName": {"value": "SH3 domain-containing kinase"}
        }
    },
    "comments": [
        {
            "commentType": "FUNCTION",
            "texts": [{"value": "Involved in signal transduction"}],
        },
        {
            "commentType": "CATALYTIC ACTIVITY",
            "reaction": {"name": "ATP + protein = ADP + phosphoprotein"},
        },
        {
            "commentType": "SUBCELLULAR LOCATION",
            "subcellularLocations": [
                {"location": {"value": "Cytoplasm"}},
                {"location": {"value": "Nucleus"}},
            ],
        },
    ],
    "uniProtKBCrossReferences": [
        {
            "database": "GO",
            "properties": [{"key": "GoTerm", "value": "F:protein kinase activity"}],
        },
    ],
    "organism": {
        "lineage": ["Eukaryota", "Metazoa", "Chordata", "Mammalia"],
    },
}


class TestExtractAnnotations:

    def test_parses_fields(self):
        annotations = extract_annotations(MOCK_UNIPROT_ENTRY)
        assert annotations["annot_protein_name"] == "SH3 domain-containing kinase"
        assert annotations["annot_function"] == "Involved in signal transduction"
        assert "Reaction=" in annotations["annot_catalytic_activity"]
        assert annotations["annot_subcellular_location"] == "Cytoplasm, Nucleus"
        assert annotations["annot_gene_ontology"] == "protein kinase activity"
        assert "Eukaryota" in annotations["annot_lineage"]

    def test_missing_fields_omitted(self):
        annotations = extract_annotations(MOCK_UNIPROT_ENTRY)
        assert "annot_cofactor" not in annotations
        assert "annot_pathway" not in annotations


class TestEnrichDataframe:

    def test_adds_annotation_columns(self):
        df = pd.DataFrame({
            "primary_Accession": ["A0A001", "A0A002"],
            "protein_sequence": ["ALYG", "EETH"],
            "family_name": ["SH3 domain", "SH3 domain"],
            "family_description": ["Binds proline", "Binds proline"],
            "[final]text_caption": ["old caption", "old caption"],
            "pfam_label": ["PF00018", "PF00018"],
        })
        uniprot_data = {"A0A001": MOCK_UNIPROT_ENTRY}
        result = enrich_dataframe(df, uniprot_data=uniprot_data)

        # Annotation columns should be present
        for col in ANNOTATION_COLUMNS:
            assert col in result.columns

        # A0A001 was enriched
        assert result.loc[0, "annot_protein_name"] == "SH3 domain-containing kinase"
        # A0A002 was not — should be NA
        assert pd.isna(result.loc[1, "annot_protein_name"])

        # Family columns copied
        assert result.loc[0, "annot_family_name"] == "SH3 domain"

    def test_does_not_compose_caption(self):
        """enrich_dataframe should NOT modify [final]text_caption."""
        df = pd.DataFrame({
            "primary_Accession": ["A0A001"],
            "protein_sequence": ["ALYG"],
            "family_name": ["SH3"],
            "family_description": ["desc"],
            "[final]text_caption": ["original caption"],
            "pfam_label": ["PF00018"],
        })
        result = enrich_dataframe(df, uniprot_data={"A0A001": MOCK_UNIPROT_ENTRY})
        assert result.loc[0, "[final]text_caption"] == "original caption"


class TestComposeCaption:

    def test_composes_from_columns(self):
        df = pd.DataFrame({
            "primary_Accession": ["X"],
            "annot_family_name": ["SH3 domain"],
            "annot_family_description": ["Binds proline-rich ligands"],
            "annot_protein_name": ["Tyrosine kinase"],
            "annot_function": ["Signal transduction"],
            "annot_lineage": ["The organism lineage is Eukaryota, Mammalia"],
        })
        result = compose_caption(df)
        caption = result.loc[0, "[final]text_caption"]

        assert caption.startswith("FAMILY NAME: SH3 domain.")
        assert "PROTEIN NAME: Tyrosine kinase." in caption
        assert "FUNCTION: Signal transduction." in caption
        assert "LINEAGE: The organism lineage is" in caption

    def test_skips_missing_fields(self):
        df = pd.DataFrame({
            "primary_Accession": ["X"],
            "annot_protein_name": ["MyProtein"],
        })
        result = compose_caption(df)
        caption = result.loc[0, "[final]text_caption"]
        assert caption == "PROTEIN NAME: MyProtein."
        assert "FAMILY NAME" not in caption

    def test_custom_field_order(self):
        df = pd.DataFrame({
            "primary_Accession": ["X"],
            "annot_function": ["Does stuff"],
            "annot_protein_name": ["MyProtein"],
        })
        custom_fields = [
            ("FUNCTION", "annot_function"),
            ("PROTEIN NAME", "annot_protein_name"),
        ]
        result = compose_caption(df, fields=custom_fields)
        caption = result.loc[0, "[final]text_caption"]
        assert caption == "FUNCTION: Does stuff. PROTEIN NAME: MyProtein."

    def test_empty_row(self):
        df = pd.DataFrame({
            "primary_Accession": ["X"],
        })
        result = compose_caption(df)
        assert result.loc[0, "[final]text_caption"] == ""
