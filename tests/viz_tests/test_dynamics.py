import os
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

from biom3.viz.dynamics import plot_probability_dynamics, plot_probability_dynamics_from_file
from biom3.viz._tokens import TOKENS


def _make_probs(steps=10, seq_len=20, num_classes=29):
    rng = np.random.default_rng(42)
    raw = rng.random((steps, seq_len, num_classes)).astype(np.float32)
    raw /= raw.sum(axis=-1, keepdims=True)
    return raw


def _make_frames(steps=10, seq_len=20, pad_positions=None):
    """Build synthetic frames.  Positions in *pad_positions* get <PAD> (23)."""
    rng = np.random.default_rng(42)
    frames = []
    current = np.zeros(seq_len, dtype=int)
    current[0] = 1   # START
    current[-1] = 22  # END
    frames.append(current.copy())
    for s in range(1, steps):
        pos = rng.integers(1, seq_len - 1)
        current[pos] = rng.integers(2, 22)
        frames.append(current.copy())
    if pad_positions is not None:
        for f in frames:
            for p in pad_positions:
                f[p] = 23
    return frames


class TestPlotProbabilityDynamics:
    def test_returns_figure(self):
        probs = _make_probs()
        fig = plot_probability_dynamics(probs, TOKENS)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 2
        plt_module = matplotlib.pyplot
        plt_module.close(fig)

    def test_with_frames(self):
        probs = _make_probs()
        frames = _make_frames()
        fig = plot_probability_dynamics(probs, TOKENS, frames=frames)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_save_to_file(self, tmp_path):
        probs = _make_probs()
        out = str(tmp_path / "dynamics.png")
        fig = plot_probability_dynamics(probs, TOKENS, output_path=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0
        matplotlib.pyplot.close(fig)

    def test_custom_figsize(self):
        probs = _make_probs()
        fig = plot_probability_dynamics(probs, TOKENS, figsize=(10, 5))
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_hide_pad(self):
        """Positions ending as PAD should be excluded from heatmap."""
        probs = _make_probs()
        frames = _make_frames(pad_positions=[17, 18, 19])
        fig = plot_probability_dynamics(
            probs, TOKENS, frames=frames, hide_pad=True,
        )
        ax_heatmap = fig.axes[0]
        im_data = ax_heatmap.images[0].get_array()
        # 3 PAD positions removed → 17 columns in transposed heatmap
        assert im_data.shape[0] == 17
        matplotlib.pyplot.close(fig)

    def test_blank_unmasked(self):
        """Steps after unmasking should be masked (white) in the heatmap."""
        probs = _make_probs()
        frames = _make_frames()
        fig = plot_probability_dynamics(
            probs, TOKENS, frames=frames, blank_unmasked=True,
        )
        ax_heatmap = fig.axes[0]
        im_data = ax_heatmap.images[0].get_array()
        assert hasattr(im_data, 'mask')
        assert im_data.mask.any()
        matplotlib.pyplot.close(fig)

    def test_hide_pad_and_blank_unmasked(self):
        """Both options combined."""
        probs = _make_probs()
        frames = _make_frames(pad_positions=[18, 19])
        fig = plot_probability_dynamics(
            probs, TOKENS, frames=frames, hide_pad=True, blank_unmasked=True,
        )
        ax_heatmap = fig.axes[0]
        im_data = ax_heatmap.images[0].get_array()
        assert im_data.shape[0] == 18
        assert hasattr(im_data, 'mask') and im_data.mask.any()
        matplotlib.pyplot.close(fig)

    def test_hide_pad_no_frames_is_noop(self):
        """hide_pad without frames should still produce a valid figure."""
        probs = _make_probs()
        fig = plot_probability_dynamics(probs, TOKENS, hide_pad=True)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)


class TestPlotProbabilityDynamicsFromFile:
    def test_load_and_plot(self, tmp_path):
        probs = _make_probs()
        tokens = np.array(TOKENS, dtype=object)
        npz_path = str(tmp_path / "test_probs.npz")
        np.savez_compressed(npz_path, probs=probs, tokens=tokens)

        fig = plot_probability_dynamics_from_file(npz_path)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 2
        matplotlib.pyplot.close(fig)

    def test_load_and_save(self, tmp_path):
        probs = _make_probs()
        tokens = np.array(TOKENS, dtype=object)
        npz_path = str(tmp_path / "test_probs.npz")
        np.savez_compressed(npz_path, probs=probs, tokens=tokens)

        out = str(tmp_path / "dynamics_from_file.png")
        fig = plot_probability_dynamics_from_file(npz_path, output_path=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0
        matplotlib.pyplot.close(fig)
