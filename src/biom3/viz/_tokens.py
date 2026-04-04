from biom3.Stage3.animation_tools import AA_COLORS

TOKENS: list[str] = [
    '-', '<START>', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '<END>', '<PAD>',
    'X', 'U', 'Z', 'B', 'O',
]

TOKEN_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(TOKENS)}

MASK_IDX: int = 0

STANDARD_AA: set[str] = set("ACDEFGHIKLMNPQRSTVWY")

SPECIAL_TOKENS: set[str] = {'-', '<START>', '<END>', '<PAD>'}


def token_idx_to_char(idx: int) -> str:
    return TOKENS[idx]


def sequence_from_token_indices(indices) -> str:
    """Convert token index array to a clean amino acid string (no special tokens)."""
    chars = []
    for idx in indices:
        tok = TOKENS[int(idx)]
        if tok not in SPECIAL_TOKENS:
            chars.append(tok)
    return "".join(chars)
