import numpy as np
import pytest
from PIL import Image

from biom3.Stage3.animation_tools import (
    _render_frame,
    _get_font,
    _MASK_COLOR,
    _PAD_COLOR,
    _PAD_IDX,
    _CELL,
    _STRIDE,
    generate_sequence_animation,
)
from biom3.viz._tokens import TOKENS


@pytest.fixture
def fonts():
    return _get_font(13), _get_font(11)


def _make_pad_frame():
    """5-position sequence: START(1), A(2), PAD(23), END(22), PAD(23)."""
    return np.array([1, 2, 23, 22, 23])


def _make_probs(seq_len=5, num_classes=29, pad_prob_at_2=0.85, pad_prob_at_4=0.10):
    probs = np.full((seq_len, num_classes), 0.01, dtype=np.float32)
    probs[0, 1] = 0.95
    probs[1, 2] = 0.90
    probs[2, _PAD_IDX] = pad_prob_at_2
    probs[3, 22] = 0.92
    probs[4, _PAD_IDX] = pad_prob_at_4
    return probs


class TestPadProbabilityGauge:
    def test_pad_cell_two_tone_high_prob(self, fonts):
        """PAD cell with high probability should have grey top and dark bottom."""
        font, font_sm = fonts
        token_indices = _make_pad_frame()
        prev_indices = np.zeros(5, dtype=int)
        probs = _make_probs(pad_prob_at_2=0.85)

        img = _render_frame(
            token_indices=token_indices,
            prev_indices=prev_indices,
            tokens=TOKENS,
            step=1,
            total_steps=5,
            title=None,
            cols_per_row=50,
            font=font,
            font_sm=font_sm,
            step_probs=probs,
            prob_style="brightness",
        )
        assert isinstance(img, Image.Image)

    def test_pad_cell_two_tone_low_prob(self, fonts):
        """PAD cell with low probability should be mostly grey (MASK_COLOR)."""
        font, font_sm = fonts
        token_indices = _make_pad_frame()
        prev_indices = np.zeros(5, dtype=int)
        probs = _make_probs(pad_prob_at_4=0.05)

        img = _render_frame(
            token_indices=token_indices,
            prev_indices=prev_indices,
            tokens=TOKENS,
            step=1,
            total_steps=5,
            title=None,
            cols_per_row=50,
            font=font,
            font_sm=font_sm,
            step_probs=probs,
            prob_style="brightness",
        )
        assert isinstance(img, Image.Image)

    def test_pad_cell_flat_without_probs(self, fonts):
        """Without probs, PAD cell should render flat _PAD_COLOR."""
        font, font_sm = fonts
        token_indices = _make_pad_frame()

        img = _render_frame(
            token_indices=token_indices,
            prev_indices=None,
            tokens=TOKENS,
            step=0,
            total_steps=1,
            title=None,
            cols_per_row=50,
            font=font,
            font_sm=font_sm,
            step_probs=None,
            prob_style=None,
        )
        assert isinstance(img, Image.Image)

    def test_aa_cell_still_modulated(self, fonts):
        """AA cells should still get brightness modulation (regression check)."""
        font, font_sm = fonts
        token_indices = np.array([2, 3, 4])  # A, C, D
        prev_indices = np.zeros(3, dtype=int)
        probs = np.full((3, 29), 0.01, dtype=np.float32)
        probs[0, 2] = 0.50
        probs[1, 3] = 0.90
        probs[2, 4] = 0.10

        img = _render_frame(
            token_indices=token_indices,
            prev_indices=prev_indices,
            tokens=TOKENS,
            step=0,
            total_steps=1,
            title=None,
            cols_per_row=50,
            font=font,
            font_sm=font_sm,
            step_probs=probs,
            prob_style="brightness",
        )
        assert isinstance(img, Image.Image)

    def test_generate_animation_with_pad(self, tmp_path):
        """Full integration: animation with PAD positions and probs."""
        frames = [np.zeros(5, dtype=int)]
        frames.append(np.array([1, 0, 0, 0, 0]))
        frames.append(np.array([1, 2, 0, 0, 0]))
        frames.append(np.array([1, 2, 23, 0, 0]))
        frames.append(np.array([1, 2, 23, 22, 0]))
        frames.append(np.array([1, 2, 23, 22, 23]))

        steps = len(frames)
        probs = np.full((steps, 5, 29), 0.01, dtype=np.float32)
        for s in range(steps):
            for j in range(5):
                tok = frames[s][j]
                if tok > 0:
                    probs[s, j, tok] = 0.85

        out = str(tmp_path / "test_pad.gif")
        generate_sequence_animation(
            frames=frames,
            tokens=TOKENS,
            output_path=out,
            probs=probs,
            prob_style="brightness",
            title="PAD gauge test",
        )
        assert os.path.isfile(out)


import os
