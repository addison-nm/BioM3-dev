"""Fetch protein structures from RCSB / AlphaFoldDB and parse BLAST hit IDs."""

from __future__ import annotations

import io
import re
import urllib.error
import urllib.request


_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

_PDB_ID_RE = re.compile(r"[0-9][0-9A-Za-z]{3}")
_UNIPROT_RE = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")


def fetch_rcsb_pdb(pdb_id: str, timeout: float = 30.0) -> str:
    """Download a PDB file from the RCSB and return its text content.

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier (case-insensitive).
    timeout : float
        HTTP timeout in seconds.

    Returns
    -------
    str
        PDB file content.
    """
    pdb_id = pdb_id.strip().upper()
    if not _PDB_ID_RE.fullmatch(pdb_id):
        raise ValueError(
            f"Invalid PDB ID {pdb_id!r}: must be 4 alphanumeric chars starting with a digit"
        )
    url = _RCSB_URL.format(pdb_id=pdb_id)
    req = urllib.request.Request(url, headers={"User-Agent": "biom3"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RCSB fetch failed for {pdb_id}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"RCSB fetch failed for {pdb_id}: {e.reason}") from e


def fetch_alphafold(uniprot_id: str, timeout: float = 30.0) -> str:
    """Download an AlphaFoldDB predicted structure by UniProt accession.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession (e.g. ``P12345``).
    timeout : float
        HTTP timeout in seconds.

    Returns
    -------
    str
        PDB file content of the AlphaFold model (v4, fragment F1).
    """
    uniprot_id = uniprot_id.strip().upper()
    if not _UNIPROT_RE.fullmatch(uniprot_id):
        raise ValueError(f"Invalid UniProt accession: {uniprot_id!r}")
    url = _ALPHAFOLD_URL.format(uniprot_id=uniprot_id)
    req = urllib.request.Request(url, headers={"User-Agent": "biom3"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"AlphaFoldDB has no model for {uniprot_id}"
            ) from e
        raise RuntimeError(f"AlphaFoldDB fetch failed for {uniprot_id}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"AlphaFoldDB fetch failed for {uniprot_id}: {e.reason}") from e


def parse_pdb_id(hit_id: str, hit_def: str = "") -> str | None:
    """Extract a 4-character PDB ID from a BLAST hit's id/definition, if present.

    Handles common NCBI formats:
      - ``pdb|4HHB|A``
      - ``pdb|4HHB``
      - ``4HHB_A`` (chain-specific)
      - ``gi|12345|pdb|4HHB|A``
    """
    for text in (hit_id, hit_def):
        if not text:
            continue
        m = re.search(r"pdb\|([0-9][0-9A-Za-z]{3})", text)
        if m:
            return m.group(1).upper()
        m = re.match(r"^([0-9][0-9A-Za-z]{3})_[0-9A-Za-z]+(?:\b|$)", text)
        if m:
            return m.group(1).upper()
    return None


def parse_uniprot_id(hit_id: str, hit_def: str = "") -> str | None:
    """Extract a UniProt accession from a BLAST hit's id/definition, if present.

    Handles NCBI ``sp|ACCESSION|NAME`` and ``tr|ACCESSION|NAME`` formats.
    """
    for text in (hit_id, hit_def):
        if not text:
            continue
        m = re.search(r"(?:sp|tr)\|([A-Z0-9]{6,10})", text)
        if m and _UNIPROT_RE.fullmatch(m.group(1)):
            return m.group(1).upper()
    return None


def pdb_to_sequence(pdb_str: str) -> str:
    """Return the one-letter amino-acid sequence of the longest peptide in a PDB.

    Walks all chains and returns the longest peptide sequence found. Non-standard
    residues are skipped by BioPython's ``PPBuilder``.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import PPBuilder

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("q", io.StringIO(pdb_str))
    ppb = PPBuilder()
    longest = ""
    for pp in ppb.build_peptides(structure):
        s = str(pp.get_sequence())
        if len(s) > len(longest):
            longest = s
    return longest
