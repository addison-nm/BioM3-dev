"""Stochastic text-prompt construction.

RandomizedPromptConstructor is instantiated with a per-key retention
probability dictionary and a shuffle flag. Its build() method takes one or
more fragment dicts (each mapping a key to a fragment string such as
"LABEL: DESC"), drops fragments according to their probability, optionally
shuffles the survivors, and joins them into a final prompt string.

This operates on fragments that are already assembled; it is independent of
dataset/caption building on the database side.

Usage:
    from biom3.core.prompt_assembly import RandomizedPromptConstructor

    rpc = RandomizedPromptConstructor({"name": 1.0, "function": 0.5}, shuffle=True)

    rpc.build({"name": "PROTEIN_NAME: SH3", "function": "FUNCTION: osmosensing"})
    # e.g. "FUNCTION: osmosensing. PROTEIN_NAME: SH3."

    rpc.build([frags_a, frags_b])  # -> ["prompt_a", "prompt_b"]
"""

import random


class RandomizedPromptConstructor:
    """Builds randomized text prompts from keyed fragments.

    Args:
        probs: dict mapping a key to its retention probability in [0, 1].
            Probabilities are applied per fragment at build time; a fragment
            key absent from probs is always retained (probability 1.0), and a
            probs key absent from a given fragment dict is simply ignored.
        shuffle: if True, retained fragments are shuffled before joining.
        separator: string inserted between retained fragments.
        trailing_period: if True, ensure a non-empty prompt ends with a period.
    """

    def __init__(self, probs=None, shuffle=False,
                 separator=". ", trailing_period=True):
        probs = dict(probs) if probs else {}
        for key, p in probs.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"retention probability for {key!r} must be in [0, 1], got {p}"
                )
        self.probs = probs
        self._shuffle = bool(shuffle)
        self.separator = separator
        self.trailing_period = trailing_period

    def shuffle(self, b):
        """Set whether retained fragments are shuffled before joining."""
        self._shuffle = bool(b)
        return self

    def build(self, fragments, rng=None):
        """Construct prompt(s) from one or more fragment dicts.

        Args:
            fragments: a single fragment dict, or a list/tuple of fragment
                dicts. Each dict maps a key to a fragment string used verbatim.
            rng: optional random.Random instance for reproducibility. Defaults
                to the module-level random generator.

        Returns:
            A prompt string for a single dict, or a list of prompt strings for
            a list/tuple of dicts. An all-dropped fragment dict yields "".
        """
        if rng is None:
            rng = random
        if isinstance(fragments, dict):
            return self._build_one(fragments, rng)
        return [self._build_one(frag, rng) for frag in fragments]

    def _build_one(self, fragments, rng):
        kept = [
            value for key, value in fragments.items()
            if rng.random() < self.probs.get(key, 1.0)
        ]
        if self._shuffle:
            rng.shuffle(kept)

        prompt = self.separator.join(kept)
        if self.trailing_period and prompt and not prompt.endswith("."):
            prompt += "."
        return prompt

    def __call__(self, fragments, rng=None):
        return self.build(fragments, rng=rng)
