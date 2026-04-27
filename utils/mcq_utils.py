"""MCQ prompt construction helpers.

Pure functions used by ``ClassificationDataset`` to build N-choice multiple-choice
questions with a prescribed correct-option position and a seeded distractor order.
Keeping this dataset-agnostic lets category-based and category-free MCQ variants
share a single implementation.
"""

from __future__ import annotations

import random
from typing import Sequence


def letter_for(pos: int) -> str:
    """0 -> 'A', 1 -> 'B', ..."""
    return chr(ord("A") + pos)


def build_mcq_prompt(
    *,
    question: str,
    correct: str,
    pool: Sequence[str],
    n_choices: int,
    correct_pos: int,
    shuffle_seed: str,
    letter_sep: str = ". ",
) -> tuple[str, str]:
    """Return ``(prompt_text, correct_letter)`` for an N-choice MCQ.

    Arguments:
        question: The question stem rendered on the first line.
        correct: The correct class name (inserted at ``correct_pos``).
        pool: Candidate distractor class names (``correct`` already excluded).
        n_choices: Final MCQ width ``k``. The caller must ensure
            ``len(pool) >= n_choices - 1`` or cap ``n_choices`` accordingly.
        correct_pos: Position of the correct answer, ``0 <= correct_pos < n_choices``.
        shuffle_seed: Deterministic seed (e.g. wnid or offset) for distractor order.
        letter_sep: Separator between letter and option text. Default ``". "``.

    Returns:
        Tuple of the rendered prompt and the correct option letter.
    """
    if n_choices < 2:
        raise ValueError(f"n_choices must be >= 2, got {n_choices}")
    if not (0 <= correct_pos < n_choices):
        raise ValueError(
            f"correct_pos {correct_pos} out of range for n_choices={n_choices}"
        )
    if len(pool) < n_choices - 1:
        raise ValueError(
            f"pool has {len(pool)} distractors, need at least {n_choices - 1}"
        )

    shuffled = list(pool)
    random.Random(shuffle_seed).shuffle(shuffled)
    choices = shuffled[: n_choices - 1]
    choices.insert(correct_pos, correct)

    lines = [question]
    for i, choice in enumerate(choices):
        lines.append(f"{letter_for(i)}{letter_sep}{choice}")
    return "\n".join(lines), letter_for(correct_pos)
