from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Superimposer, Select
from Bio.PDB.Polypeptide import is_aa
from Bio.Blast import NCBIWWW, NCBIXML

from biom3.viz import viewer


@dataclass
class AlignmentResult:
    rmsd: float
    n_atoms: int
    rotation: np.ndarray
    translation: np.ndarray
    fixed_pdb: str
    moving_pdb: str


@dataclass
class BlastResult:
    hit_id: str
    hit_def: str
    e_value: float
    score: float
    identities: int
    positives: int
    align_length: int
    query_seq: str
    hit_seq: str
    midline: str
    percent_identity: float


def _parse_pdb(pdb: str | Path, parser_id: str = "struct") -> "Bio.PDB.Structure.Structure":
    parser = PDBParser(QUIET=True)
    if isinstance(pdb, Path):
        return parser.get_structure(parser_id, str(pdb))
    if "\n" not in pdb:
        p = Path(pdb)
        if p.is_file():
            return parser.get_structure(parser_id, str(p))
    handle = StringIO(pdb)
    return parser.get_structure(parser_id, handle)


def _get_ca_atoms(structure, atom_name: str = "CA") -> list:
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True) and atom_name in residue:
                    atoms.append(residue[atom_name])
        break  # first model only
    return atoms


def _structure_to_pdb_string(structure) -> str:
    io = PDBIO()
    io.set_structure(structure)
    out = StringIO()
    io.save(out)
    return out.getvalue()


def superimpose(
    fixed: str | Path,
    moving: str | Path,
    atom_name: str = "CA",
) -> AlignmentResult:
    """Superimpose two PDB structures on C-alpha atoms using BioPython.

    Pairs atoms by sequential index, truncating to the shorter chain.
    Applies the rotation/translation to the moving structure in-place
    and returns both as PDB strings.
    """
    fixed_struct = _parse_pdb(fixed, "fixed")
    moving_struct = _parse_pdb(moving, "moving")

    fixed_atoms = _get_ca_atoms(fixed_struct, atom_name)
    moving_atoms = _get_ca_atoms(moving_struct, atom_name)

    n = min(len(fixed_atoms), len(moving_atoms))
    if n == 0:
        raise ValueError("No matching atoms found for superimposition")

    sup = Superimposer()
    sup.set_atoms(fixed_atoms[:n], moving_atoms[:n])

    # Apply transformation to all atoms in the moving structure
    all_moving_atoms = list(moving_struct.get_atoms())
    sup.apply(all_moving_atoms)

    return AlignmentResult(
        rmsd=sup.rms,
        n_atoms=n,
        rotation=sup.rotran[0],
        translation=sup.rotran[1],
        fixed_pdb=_structure_to_pdb_string(fixed_struct),
        moving_pdb=_structure_to_pdb_string(moving_struct),
    )


def superimpose_and_view(
    fixed: str | Path,
    moving: str | Path,
    labels: list[str] | None = None,
    atom_name: str = "CA",
    width: int = 800,
    height: int = 600,
) -> tuple:
    """Superimpose two structures and display the overlay.

    Returns (py3Dmol.view, AlignmentResult).
    """
    result = superimpose(fixed, moving, atom_name=atom_name)
    if labels is None:
        labels = ["Fixed", "Moving"]
    view = viewer.view_overlay(
        [result.fixed_pdb, result.moving_pdb],
        labels=labels,
        width=width,
        height=height,
    )
    return view, result


def blast_sequence(
    sequence: str,
    database: str = "nr",
    program: str = "blastp",
    e_value: float = 10.0,
    max_hits: int = 10,
) -> list[BlastResult]:
    """Run remote BLAST on a protein sequence and return parsed results.

    Uses Bio.Blast.NCBIWWW.qblast for remote BLAST against NCBI.
    """
    result_handle = NCBIWWW.qblast(
        program,
        database,
        sequence,
        expect=e_value,
        hitlist_size=max_hits,
    )
    blast_records = NCBIXML.parse(result_handle)

    results = []
    for record in blast_records:
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                pct_identity = (hsp.identities / hsp.align_length * 100) if hsp.align_length else 0.0
                results.append(BlastResult(
                    hit_id=alignment.hit_id,
                    hit_def=alignment.hit_def,
                    e_value=hsp.expect,
                    score=hsp.score,
                    identities=hsp.identities,
                    positives=hsp.positives,
                    align_length=hsp.align_length,
                    query_seq=hsp.query,
                    hit_seq=hsp.sbjct,
                    midline=hsp.match,
                    percent_identity=pct_identity,
                ))

    results.sort(key=lambda r: r.e_value)
    return results
