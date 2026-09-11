"""save_animation_frames writes a faithful, self-describing trajectory record.

Downstream interactive viewers (e.g. nm-portal's generation animation) read
these JSON files, so the contract is pinned here: one record per animated
(prompt, replica) pair, carrying the token vocabulary and
frames[step][position] token indices, with index 0 being the still-masked
state. The frames are the realized sampled path, not an argmax reconstruction.
"""
import json

import numpy as np

from biom3.Stage3.run_ProteoScribe_sample import (
    resolve_animate_prompts,
    save_animation_frames,
)
from biom3.Stage3.sampling_analysis import TokenProbRow


def test_writes_one_record_per_pair_with_realized_path(tmp_path):
    tokens = ['-', '<START>', 'A', 'C', '<END>', '<PAD>']
    # Two diffusion steps; position 1 resolves from the mask (0) to a residue.
    frames = {
        (0, 0): [np.array([1, 0, 4]), np.array([1, 2, 4])],
        (2, 1): [np.array([1, 0, 4]), np.array([1, 3, 4])],
    }

    written = save_animation_frames(frames, tokens, str(tmp_path))

    assert sorted(p.rsplit('/', 1)[-1] for p in written) == [
        'prompt_0_replica_0.json', 'prompt_2_replica_1.json']

    rec = json.loads((tmp_path / 'prompt_0_replica_0.json').read_text())
    assert rec['prompt_index'] == 0
    assert rec['replica_index'] == 0
    assert rec['tokens'] == tokens
    assert rec['frames'] == [[1, 0, 4], [1, 2, 4]]
    # Plain Python ints, so the record is JSON-native (no numpy scalars).
    assert all(isinstance(v, int) for step in rec['frames'] for v in step)
    # Nothing recorded, nothing written: the field only appears when asked for.
    assert 'confidence' not in rec


def test_confidence_series_runs_from_the_step_a_residue_was_placed(tmp_path):
    tokens = ['-', '<START>', 'A', 'C', '<END>', '<PAD>']
    # Position 0 is a pre-revealed <START>; 1 holds 'A' from step 0; 2 holds 'C'
    # from step 2; 3 holds <PAD> from step 1.
    frames = [
        np.array([1, 2, 0, 0]),
        np.array([1, 2, 0, 5]),
        np.array([1, 2, 3, 5]),
    ]
    token_probs = TokenProbRow(
        values=np.array([
            [0.9, 0.8, 0.1, 0.2],
            [0.9, 0.75, 0.15, 0.6],
            [0.9, 0.7, 0.55, 0.65],
        ], dtype=np.float16),
        placed_at=np.array([-1, 0, 2, 1], dtype=np.int32),
    )

    save_animation_frames(
        {(0, 0): frames}, tokens, str(tmp_path),
        confidence={(0, 0): token_probs},
    )

    rec = json.loads((tmp_path / 'prompt_0_replica_0.json').read_text())
    assert rec['confidence'] == [
        None,                 # <START>: structural, and never sampled
        [0.8, 0.75, 0.7],     # placed at step 0, so one value per step
        [0.55],               # placed at the last step, so a single value
        None,                 # <PAD>: not an amino acid
    ]


def test_empty_selection_writes_nothing(tmp_path):
    assert save_animation_frames({}, ['-', 'A'], str(tmp_path)) == []
    assert list(tmp_path.iterdir()) == []


def test_resolve_animate_prompts_clamps_out_of_range():
    # A dispatcher expands "first N" to 0..N-1 without knowing the exact prompt
    # count, so out-of-range indices must cap to what exists, not raise minutes
    # into the Stage 3 job.
    assert resolve_animate_prompts([0, 1, 2, 3, 4], 3) == {0, 1, 2}
    # Nothing valid -> animation off, rather than a crash.
    assert resolve_animate_prompts([7, 8], 3) is None
    # The unambiguous cases are unchanged.
    assert resolve_animate_prompts('all', 2) == {0, 1}
    assert resolve_animate_prompts(None, 5) is None
    assert resolve_animate_prompts([1], 3) == {1}
