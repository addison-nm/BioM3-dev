import gzip
import json
import os

import pytest

from biom3.dbio_v2.parsers import ParseError, SchemaError
from biom3.dbio_v2.parsers.uniprot_dat import (
    UniProtRecord,
    iter_records,
    parse_description,
    parse_entry,
)

DATDIR = os.path.join(os.path.dirname(__file__), "..", "..", "_data", "dbio_v2")
SPROT = os.path.join(DATDIR, "uniprot_sprot_mini.dat")
TREMBL = os.path.join(DATDIR, "uniprot_trembl_mini.dat")
SPROT_EXPECTATIONS = os.path.join(
    DATDIR, "parser_expectations", "uniprot_sprot_mini",
)

PROMISED = (
    "entry_name", "status", "accessions", "primary_accession", "dates",
    "description", "gene_names", "organism", "organelle", "lineage",
    "taxon_id", "hosts", "references", "comments", "cross_references",
    "features", "keywords", "protein_existence", "sequence",
    "sequence_length", "molecular_weight", "crc64", "unparsed",
)


@pytest.fixture(scope="module")
def sprot():
    return list(iter_records(SPROT))


@pytest.fixture(scope="module")
def trembl():
    return list(iter_records(TREMBL))


class TestStructuralInvariants:

    @pytest.mark.parametrize("fixture,status,n", [
        ("sprot", "Reviewed", 100),
        ("trembl", "Unreviewed", 100),
    ])
    def test_counts_types_and_status(self, request, fixture, status, n):
        recs = request.getfixturevalue(fixture)
        assert len(recs) == n
        assert all(isinstance(r, UniProtRecord) for r in recs)
        assert {r.status for r in recs} == {status}
        assert all(r.record_type == "uniprot" for r in recs)

    def test_sequence_length_and_crc64_validated(self, sprot, trembl):
        for r in sprot + trembl:
            assert len(r.sequence) == r.sequence_length
            assert len(r.crc64) == 16

    def test_promised_schema_is_exact(self, sprot):
        assert UniProtRecord._promised_fields() == PROMISED
        assert set(sprot[0].as_dict()) == set(PROMISED)


class TestSwissProtEntry:
    """First Swiss-Prot entry: 001R_FRG3G / Q6GZX4 (Reviewed, viral)."""

    def test_id_line(self, sprot):
        r = sprot[0]
        assert r.entry_name == "001R_FRG3G"
        assert r.status == "Reviewed"
        assert r.sequence_length == 256
        assert r.molecular_weight == 29735
        assert r.crc64 == "B4840739BF7D4121"

    def test_accessions(self, sprot):
        assert sprot[0].accessions == ["Q6GZX4"]
        assert sprot[0].primary_accession == "Q6GZX4"

    def test_taxonomy(self, sprot):
        r = sprot[0]
        assert r.taxon_id == 654924
        assert r.lineage[:3] == ["Viruses", "Varidnaviria", "Bamfordvirae"]
        assert not r.lineage[-1].endswith(".")

    def test_description_shape_c(self, sprot):
        d = sprot[0].description
        assert d["rec_name"] == {
            "full": "Putative transcription factor 001R",
            "short": [], "ec_numbers": [], "evidence": [],
        }
        assert d["sub_name"] is None
        assert d["alt_names"] == []
        assert d["includes"] == [] and d["contains"] == []
        assert d["flags"] == []

    def test_cross_references_pfam(self, sprot):
        assert sprot[0].cross_references["Pfam"] == [["PF04947", "Pox_VLTF3", "1"]]

    def test_comment_block_keeps_evidence_tag_and_period(self, sprot):
        c = sprot[0].comments[0]
        assert c["topic"] == "FUNCTION"
        assert c["text"] == "Transcription activation. {ECO:0000305}."

    def test_terminator_strips_semicolons_only(self, sprot):
        """Semicolons (field delimiters) are stripped; trailing periods
        (often part of the data, e.g. abbreviation initials in `RA`) are
        preserved."""
        r = sprot[0]
        # Periods preserved as written in the source:
        assert r.organism == "Frog virus 3 (isolate Goorha) (FV-3)."
        assert r.dates[0] == "28-JUN-2011, integrated into UniProtKB/Swiss-Prot."
        assert r.hosts[0] == "NCBI_TaxID=30343; Dryophytes versicolor (chameleon treefrog)."
        ref = r.references[0]
        assert ref["RP"] == "NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA]."
        assert ref["RL"] == "Virology 323:70-84(2004)."
        # Semicolons (UniProt field-terminator) are stripped:
        assert r.gene_names == ["ORFNames=FV3-001R"]
        assert ref["RX"] == "PubMed=15165820; DOI=10.1016/j.virol.2004.02.019"
        assert ref["RA"] == "Tan W.G., Barkman T.J., Gregory Chinchar V., Essani K."
        # RT keeps the source's wrapping quote and the title's internal
        # period; only the trailing `;` is stripped:
        assert ref["RT"].startswith('"') and ref["RT"].endswith('."')

    def test_feature_qualifiers(self, sprot):
        f = sprot[0].features[0]
        assert f["type"] == "CHAIN"
        assert f["location"] == "1..256"
        assert f["qualifiers"]["note"] == "Putative transcription factor 001R"
        assert f["qualifiers"]["id"] == "PRO_0000410512"

    def test_keywords_pe_sequence(self, sprot):
        r = sprot[0]
        assert r.keywords[:2] == ["Activator", "Reference proteome"]
        assert r.protein_existence == "4: Predicted"
        assert r.sequence.startswith("MAFSAEDVLKEYDRRRRMEA")
        assert len(r.sequence) == 256

    def test_no_unparsed_codes(self, sprot):
        assert sprot[0].unparsed == {}


class TestTremblEntry:
    """First TrEMBL entry: F5A684_9EURY (Unreviewed, Fragment, ARBA evidence)."""

    def test_status_and_ids(self, trembl):
        r = trembl[0]
        assert r.entry_name == "F5A684_9EURY"
        assert r.status == "Unreviewed"
        assert r.primary_accession == "F5A684"
        assert r.sequence_length == 188
        assert r.crc64 == "54156C551B90A034"

    def test_description_shape_c_with_ec_and_fragment(self, trembl):
        d = trembl[0].description
        assert d["rec_name"]["full"] == "catalase"
        assert d["rec_name"]["ec_numbers"] == ["1.11.1.6"]
        assert "ECO:0000256|ARBA:ARBA00012314" in d["rec_name"]["evidence"]
        assert d["flags"] == ["Fragment"]
        assert d["sub_name"] is None and d["alt_names"] == []

    def test_cofactor_comment_and_pfam(self, trembl):
        r = trembl[0]
        assert r.comments[0]["topic"] == "COFACTOR"
        assert "ECO:0000256" in r.comments[0]["text"]
        assert r.cross_references["Pfam"] == [["PF00199", "Catalase", "1"]]

    def test_keywords_keep_evidence(self, trembl):
        assert all("ECO:0000256" in k for k in trembl[0].keywords[:1])


class TestFaithfulCapture:

    def test_copyright_footer_is_not_a_comment(self, sprot, trembl):
        for r in sprot + trembl:
            for c in r.comments:
                assert "Copyrighted" not in c["topic"]
                assert not c["topic"].startswith("---")


class TestFailLoud:

    def _first_entry_text(self, path):
        out = []
        for line in open(path):
            out.append(line)
            if line.rstrip("\n") == "//":
                break
        return "".join(out)

    def test_tampered_residue_raises_crc64(self, tmp_path):
        src = open(SPROT).read()
        i = src.index("\nSQ   ")
        j = src.index("\n     ", i) + 6
        bad = src[:j] + ("A" if src[j] != "A" else "C") + src[j + 1:]
        p = tmp_path / "bad.dat"
        p.write_text(bad)
        with pytest.raises(ParseError, match="CRC64"):
            list(iter_records(str(p)))

    def test_truncated_gzip_raises(self, tmp_path):
        raw = gzip.compress(open(SPROT, "rb").read())
        p = tmp_path / "trunc.dat.gz"
        p.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(ParseError):
            list(iter_records(str(p)))

    def test_missing_terminator_raises(self, tmp_path):
        entry = self._first_entry_text(SPROT)
        assert entry.endswith("//\n")
        p = tmp_path / "noterm.dat"
        p.write_text(entry[: -len("//\n")])
        with pytest.raises(ParseError, match="mid-entry"):
            list(iter_records(str(p)))

    def test_file_not_starting_with_id_raises(self, tmp_path):
        p = tmp_path / "junk.dat"
        p.write_text("garbage line\n" + self._first_entry_text(SPROT))
        with pytest.raises(ParseError, match="expected an ID line"):
            list(iter_records(str(p)))

    def test_length_mismatch_raises(self, tmp_path):
        src = self._first_entry_text(SPROT)
        bad = src.replace("SEQUENCE   256 AA;", "SEQUENCE   255 AA;")
        p = tmp_path / "len.dat"
        p.write_text(bad)
        with pytest.raises(ParseError, match="length"):
            list(iter_records(str(p)))


class TestJsonExpectations:
    """Per-entry comparison against hand-written JSON expectations under
    tests/_data/dbio_v2/parser_expectations/uniprot_sprot_mini/."""

    @pytest.mark.parametrize("idx", range(8))
    def test_record_matches_expectation(self, sprot, idx):
        with open(os.path.join(SPROT_EXPECTATIONS, f"sprot_exp_{idx}.json")) as f:
            expected = json.load(f)
        actual = sprot[idx].as_dict()
        assert actual == expected, (
            f"#{idx} ({sprot[idx].entry_name}) mismatch; "
            f"differing keys: "
            f"{[k for k in expected if expected[k] != actual.get(k)]}"
        )


class TestParseDescriptionUnit:
    """Focused unit tests for the DE-block grammar parser."""

    def test_empty_de_block(self):
        d = parse_description([])
        assert d == {"rec_name": None, "sub_name": None, "alt_names": [],
                     "includes": [], "contains": [], "flags": []}

    def test_subname_branch(self):
        # TrEMBL replaces RecName with SubName; parser must route accordingly.
        lines = ["SubName: Full=Hypothetical protein {ECO:0000313|EMBL:ABC.1};"]
        d = parse_description(lines)
        assert d["rec_name"] is None
        assert d["sub_name"] == {
            "full": "Hypothetical protein", "short": [], "ec_numbers": [],
            "evidence": ["ECO:0000313|EMBL:ABC.1"],
        }

    def test_short_and_ec_continuations(self):
        lines = [
            "RecName: Full=Protein X;",
            "         Short=PX;",
            "         EC=1.2.3.4;",
        ]
        rec = parse_description(lines)["rec_name"]
        assert rec["full"] == "Protein X"
        assert rec["short"] == ["PX"]
        assert rec["ec_numbers"] == ["1.2.3.4"]

    def test_evidence_deduplicated(self):
        # Same ECO tag on both Full= and EC= must not appear twice.
        lines = [
            "RecName: Full=catalase {ECO:0000256|ARBA:A1};",
            "         EC=1.11.1.6 {ECO:0000256|ARBA:A1};",
        ]
        ev = parse_description(lines)["rec_name"]["evidence"]
        assert ev == ["ECO:0000256|ARBA:A1"]

    def test_multiple_flags_one_line(self):
        d = parse_description(["Flags: Precursor; Fragment;"])
        assert d["flags"] == ["Precursor", "Fragment"]

    def test_polymorphic_altname_allergen(self):
        d = parse_description([
            "RecName: Full=Major protein;",
            "AltName: Allergen=Foo i 1 {ECO:0000305};",
        ])
        assert d["alt_names"] == [{"allergen": "Foo i 1", "evidence": ["ECO:0000305"]}]


class TestParseEntryUnit:

    def test_parse_entry_round_trips_first_record(self, sprot):
        text = []
        for line in open(SPROT):
            text.append(line)
            if line.rstrip("\n") == "//":
                break
        rec = parse_entry(text)
        assert isinstance(rec, UniProtRecord)
        assert rec.primary_accession == sprot[0].primary_accession
        assert rec.sequence == sprot[0].sequence

    def test_parse_entry_emits_exact_schema(self, sprot):
        with pytest.raises(SchemaError):
            UniProtRecord(entry_name="x")
