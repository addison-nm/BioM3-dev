"""Tests for --scale_learning_rate parsing (true/false/linear/sqrt) and the
resulting learning-rate scaling factor. CPU-only, no weights.
"""

from pathlib import Path

import pytest

from biom3.Stage3.run_PL_training import parse_lr_scaling
from biom3.Stage3.run_ProteoScribe_finetuning import parse_arguments

# Self-contained minimal config (record_schema + scale_learning_rate, no
# compose_plugins) so parsing does not load external plugin files. Absolute path
# -> CWD-independent.
TEST_CONFIG = str(
    Path(__file__).resolve().parents[1] / "_data" / "configs" / "test_lr_scaling_config.json"
)


class TestParseLrScaling:

    @pytest.mark.parametrize("value,expected", [
        (True, 'linear'),
        (False, None),
        ('true', 'linear'),
        ('True', 'linear'),
        ('false', None),
        ('False', None),
        ('linear', 'linear'),
        ('sqrt', 'sqrt'),
        ('SQRT', 'sqrt'),
        ('none', None),
    ])
    def test_valid_values(self, value, expected):
        assert parse_lr_scaling(value) == expected

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_lr_scaling('quadratic')


class TestEntrypointParsing:
    """The real finetuning entrypoint parser must accept 'sqrt' end-to-end
    (config value, and a CLI override) — it previously raised via str_to_bool."""

    def test_config_value_sqrt(self):
        args = parse_arguments(['--config_path', TEST_CONFIG])
        assert args.scale_learning_rate == 'sqrt'

    def test_cli_overrides_to_linear(self):
        args = parse_arguments(
            ['--config_path', TEST_CONFIG, '--scale_learning_rate', 'true'])
        assert args.scale_learning_rate == 'linear'

    def test_cli_overrides_to_disabled(self):
        args = parse_arguments(
            ['--config_path', TEST_CONFIG, '--scale_learning_rate', 'false'])
        assert args.scale_learning_rate is None


class TestScalingFactor:
    """Mirror the factor logic in train_model for the production geometry."""

    def _factor(self, mode, num_nodes, devices_per_node):
        if not mode:
            return 1.0
        n = num_nodes * devices_per_node
        return n ** 0.5 if mode == 'sqrt' else n

    def test_sqrt_factor_16x12(self):
        f = self._factor('sqrt', 16, 12)
        assert f == pytest.approx(192 ** 0.5)
        assert 1e-4 * f == pytest.approx(1.3856e-3, rel=1e-3)

    def test_linear_factor_16x12(self):
        assert self._factor('linear', 16, 12) == 192

    def test_disabled_factor(self):
        assert self._factor(None, 16, 12) == 1.0
