import numpy as np
import pytest

from biom3.viz.unmasking import (
    extract_unmasking_order,
    extract_unmasking_order_from_sampling_path,
    unmasking_order_to_normalized,
)


class TestExtractUnmaskingOrder:
    def test_sequential_unmasking(self):
        """5-position sequence, one position unmasked per step."""
        # Token indices: 0 = masked, 2+ = amino acids, 1 = START
        seq_len = 7  # START + 5 AA + END
        frames = []
        # Step 0: START(1) and END(22) present, rest masked
        frame0 = np.array([1, 0, 0, 0, 0, 0, 22])
        frames.append(frame0)
        # Steps 1-5: unmask one AA position each step
        for step in range(1, 6):
            frame = frames[-1].copy()
            frame[step] = step + 1  # unmask position `step` with token idx step+1
            frames.append(frame)

        order = extract_unmasking_order(frames)
        assert order.shape == (seq_len,)
        # START at position 0 was unmasked at step 0
        assert order[0] == 0
        # AA positions 1-5 unmasked at steps 1-5
        for i in range(1, 6):
            assert order[i] == i
        # END at position 6 was unmasked at step 0
        assert order[6] == 0

    def test_all_unmasked_at_once(self):
        """All positions unmasked from the start."""
        frame = np.array([1, 2, 3, 4, 22])
        frames = [frame, frame.copy()]
        order = extract_unmasking_order(frames)
        assert np.all(order == 0)


class TestExtractFromSamplingPath:
    def test_permutation(self):
        path = np.array([3, 0, 4, 1, 2])
        order = extract_unmasking_order_from_sampling_path(path)
        np.testing.assert_array_equal(order, [3, 0, 4, 1, 2])


class TestNormalize:
    def test_basic_normalization(self):
        order = np.array([0, 1, 2, 3, 4])
        result = unmasking_order_to_normalized(order)
        assert result[0] == pytest.approx(0.0)
        assert result[4] == pytest.approx(1.0)
        assert result[2] == pytest.approx(0.5)

    def test_negative_values_become_nan(self):
        order = np.array([-1, 0, 1, 2, -1])
        result = unmasking_order_to_normalized(order)
        assert np.isnan(result[0])
        assert np.isnan(result[4])
        assert not np.isnan(result[1])

    def test_all_same_value(self):
        order = np.array([5, 5, 5])
        result = unmasking_order_to_normalized(order)
        np.testing.assert_array_equal(result, [0.5, 0.5, 0.5])
