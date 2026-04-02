import pytest

from biom3.dbio.caption import (
    CaptionSpec,
    compose_row_caption,
    strip_pubmed_refs,
    strip_evidence_tags,
)


class TestStripPubmedRefs:

    def test_removes_single_ref(self):
        text = "Does something (PubMed:12345)."
        assert strip_pubmed_refs(text) == "Does something."

    def test_removes_multiple_refs(self):
        text = "Known role (PubMed:12345, PubMed:67890)."
        assert strip_pubmed_refs(text) == "Known role."

    def test_cleans_double_periods(self):
        text = "Sentence one (PubMed:123). Sentence two."
        result = strip_pubmed_refs(text)
        assert ".." not in result

    def test_no_refs_unchanged(self):
        text = "Plain text with no references."
        assert strip_pubmed_refs(text) == text


class TestStripEvidenceTags:

    def test_removes_eco_tag(self):
        text = "Transcription activation. {ECO:0000305}."
        assert "{ECO:" not in strip_evidence_tags(text)

    def test_no_tags_unchanged(self):
        text = "Plain text."
        assert strip_evidence_tags(text) == text


class TestCaptionSpec:

    def test_default_spec_uses_annotation_fields(self):
        spec = CaptionSpec()
        assert len(spec.fields) > 0
        assert spec.fields[0][1] == "annot_family_name"

    def test_custom_fields(self):
        spec = CaptionSpec(fields=[("Name", "annot_protein_name")])
        assert len(spec.fields) == 1


class TestComposeRowCaption:

    def test_empty_annotations(self):
        spec = CaptionSpec(fields=[("NAME", "annot_protein_name")])
        assert compose_row_caption({}, spec) == ""

    def test_single_field(self):
        spec = CaptionSpec(
            fields=[("PROTEIN NAME", "annot_protein_name")],
            family_names_label=None,
        )
        result = compose_row_caption({"annot_protein_name": "Kinase"}, spec)
        assert result == "PROTEIN NAME: Kinase."

    def test_multiple_fields(self):
        spec = CaptionSpec(
            fields=[
                ("NAME", "annot_protein_name"),
                ("FUNCTION", "annot_function"),
            ],
            family_names_label=None,
        )
        annotations = {
            "annot_protein_name": "Kinase",
            "annot_function": "Phosphorylates substrates",
        }
        result = compose_row_caption(annotations, spec)
        assert "NAME: Kinase" in result
        assert "FUNCTION: Phosphorylates substrates" in result

    def test_skips_empty_fields(self):
        spec = CaptionSpec(
            fields=[
                ("NAME", "annot_protein_name"),
                ("FUNCTION", "annot_function"),
            ],
            family_names_label=None,
        )
        result = compose_row_caption({"annot_protein_name": "Kinase"}, spec)
        assert "FUNCTION" not in result

    def test_custom_separator(self):
        spec = CaptionSpec(
            fields=[
                ("NAME", "annot_protein_name"),
                ("FUNCTION", "annot_function"),
            ],
            separator="; ",
            family_names_label=None,
        )
        annotations = {
            "annot_protein_name": "Kinase",
            "annot_function": "Does stuff",
        }
        result = compose_row_caption(annotations, spec)
        assert "; " in result

    def test_custom_template(self):
        spec = CaptionSpec(
            fields=[("name", "annot_protein_name")],
            field_template="{label}={value}",
            family_names_label=None,
        )
        result = compose_row_caption({"annot_protein_name": "Kinase"}, spec)
        assert result.startswith("name=Kinase")

    def test_family_names_appended(self):
        spec = CaptionSpec(
            fields=[("NAME", "annot_protein_name")],
            family_names_label="FAMILY NAMES",
            family_names_template="Family names are {names}",
        )
        result = compose_row_caption(
            {"annot_protein_name": "Kinase"},
            spec,
            pfam_family_names=["SH3 domain", "Cutinase"],
        )
        assert "FAMILY NAMES: Family names are SH3 domain, Cutinase" in result

    def test_family_names_omitted_when_label_none(self):
        spec = CaptionSpec(
            fields=[("NAME", "annot_protein_name")],
            family_names_label=None,
        )
        result = compose_row_caption(
            {"annot_protein_name": "Kinase"},
            spec,
            pfam_family_names=["SH3 domain"],
        )
        assert "FAMILY" not in result

    def test_pubmed_stripped(self):
        spec = CaptionSpec(
            fields=[("FUNCTION", "annot_function")],
            strip_pubmed=True,
            family_names_label=None,
        )
        result = compose_row_caption(
            {"annot_function": "Does something (PubMed:12345)."},
            spec,
        )
        assert "PubMed" not in result

    def test_pubmed_preserved_when_disabled(self):
        spec = CaptionSpec(
            fields=[("FUNCTION", "annot_function")],
            strip_pubmed=False,
            strip_evidence=False,
            family_names_label=None,
        )
        result = compose_row_caption(
            {"annot_function": "Does something (PubMed:12345)."},
            spec,
        )
        assert "PubMed:12345" in result

    def test_trailing_period_control(self):
        spec = CaptionSpec(
            fields=[("NAME", "annot_protein_name")],
            trailing_period=False,
            family_names_label=None,
        )
        result = compose_row_caption({"annot_protein_name": "Kinase"}, spec)
        assert not result.endswith(".")

    def test_pfam_style_lowercase(self):
        spec = CaptionSpec(
            fields=[
                ("Protein name", "family_name"),
                ("Family description", "family_description"),
            ],
            strip_pubmed=False,
            strip_evidence=False,
            family_names_label=None,
            trailing_period=False,
        )
        result = compose_row_caption(
            {"family_name": "SH3 domain", "family_description": "Binds stuff"},
            spec,
        )
        assert result.startswith("Protein name: SH3 domain")
        assert "Family description: Binds stuff" in result
