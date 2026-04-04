import pytest


# Minimal PDB with 5 alanine residues in a straight line along the z-axis.
MINI_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       0.000   0.000   3.800  1.00  0.00           N
ATOM      6  CA  ALA A   2       1.458   0.000   3.800  1.00  0.00           C
ATOM      7  C   ALA A   2       2.009   1.420   3.800  1.00  0.00           C
ATOM      8  O   ALA A   2       1.246   2.390   3.800  1.00  0.00           O
ATOM      9  N   ALA A   3       0.000   0.000   7.600  1.00  0.00           N
ATOM     10  CA  ALA A   3       1.458   0.000   7.600  1.00  0.00           C
ATOM     11  C   ALA A   3       2.009   1.420   7.600  1.00  0.00           C
ATOM     12  O   ALA A   3       1.246   2.390   7.600  1.00  0.00           O
ATOM     13  N   ALA A   4       0.000   0.000  11.400  1.00  0.00           N
ATOM     14  CA  ALA A   4       1.458   0.000  11.400  1.00  0.00           C
ATOM     15  C   ALA A   4       2.009   1.420  11.400  1.00  0.00           C
ATOM     16  O   ALA A   4       1.246   2.390  11.400  1.00  0.00           O
ATOM     17  N   ALA A   5       0.000   0.000  15.200  1.00  0.00           N
ATOM     18  CA  ALA A   5       1.458   0.000  15.200  1.00  0.00           C
ATOM     19  C   ALA A   5       2.009   1.420  15.200  1.00  0.00           C
ATOM     20  O   ALA A   5       1.246   2.390  15.200  1.00  0.00           O
END
"""

# Second PDB: same structure but shifted +2.0 along x-axis.
MINI_PDB_SHIFTED = """\
ATOM      1  N   ALA A   1       2.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       3.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       4.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       2.000   0.000   3.800  1.00  0.00           N
ATOM      6  CA  ALA A   2       3.458   0.000   3.800  1.00  0.00           C
ATOM      7  C   ALA A   2       4.009   1.420   3.800  1.00  0.00           C
ATOM      8  O   ALA A   2       3.246   2.390   3.800  1.00  0.00           O
ATOM      9  N   ALA A   3       2.000   0.000   7.600  1.00  0.00           N
ATOM     10  CA  ALA A   3       3.458   0.000   7.600  1.00  0.00           C
ATOM     11  C   ALA A   3       4.009   1.420   7.600  1.00  0.00           C
ATOM     12  O   ALA A   3       3.246   2.390   7.600  1.00  0.00           O
ATOM     13  N   ALA A   4       2.000   0.000  11.400  1.00  0.00           N
ATOM     14  CA  ALA A   4       3.458   0.000  11.400  1.00  0.00           C
ATOM     15  C   ALA A   4       4.009   1.420  11.400  1.00  0.00           C
ATOM     16  O   ALA A   4       3.246   2.390  11.400  1.00  0.00           O
ATOM     17  N   ALA A   5       2.000   0.000  15.200  1.00  0.00           N
ATOM     18  CA  ALA A   5       3.458   0.000  15.200  1.00  0.00           C
ATOM     19  C   ALA A   5       4.009   1.420  15.200  1.00  0.00           C
ATOM     20  O   ALA A   5       3.246   2.390  15.200  1.00  0.00           O
END
"""


@pytest.fixture
def mini_pdb():
    return MINI_PDB


@pytest.fixture
def mini_pdb_shifted():
    return MINI_PDB_SHIFTED


@pytest.fixture
def mini_pdb_file(tmp_path):
    p = tmp_path / "mini.pdb"
    p.write_text(MINI_PDB)
    return p
