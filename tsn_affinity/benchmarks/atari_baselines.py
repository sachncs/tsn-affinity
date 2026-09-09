"""Atari benchmark constants.

This module is the single source of truth for the human and random
baseline scores used to compute human-normalized scores. The values come
from the Arcade Learning Environment (ALE) and the published DQN paper:

    - Mnih et al. (2015). "Human-level control through deep reinforcement
      learning." Nature 518, 529-533. https://doi.org/10.1038/nature14236
    - The Arcade Learning Environment (ALE) README, baseline tables:
      https://github.com/mgbellemare/Arcade-Learning-Environment

The numbers below match the ALE random/human baselines that ship with
``ale-py`` (the Python bindings of the Atari Learning Environment).
Treat them as the official scores reported by ALE for the
``NoFrameskip-v4`` environment family.
"""

from __future__ import annotations

from typing import TypedDict

__all__ = ["ATARI_BASELINES", "ATARI_GAMES", "AtariBaselines"]


class AtariBaselines(TypedDict):
    """Human and random baseline scores for one Atari game."""

    human: float
    random: float


# Atari games used in the TSN-Affinity paper.
ATARI_GAMES: list[str] = [
    "Breakout",  # Task 0
    "Alien",  # Task 1
    "Atlantis",  # Task 2
    "Boxing",  # Task 3
    "Centipede",  # Task 4
]

# Sources: ALE README baseline tables (NoFrameskip-v4) and Mnih et al. 2015.
# Cite as: Mnih et al. (2015). "Human-level control through deep
# reinforcement learning." Nature 518, 529-533.
ATARI_BASELINES: dict[str, AtariBaselines] = {
    "Breakout": {"human": 31.8, "random": 1.7},
    "Alien": {"human": 6875.4, "random": 227.8},
    "Atlantis": {"human": 29028.1, "random": 12850.0},
    "Boxing": {"human": 71.8, "random": 0.1},
    "Centipede": {"human": 11963.2, "random": 2090.9},
}


def get_atari_baselines(game: str) -> AtariBaselines:
    """Return the human and random baselines for a single Atari game.

    Args:
        game: Bare game name (for example ``"Breakout"``).

    Returns:
        ``AtariBaselines`` mapping with ``"human"`` and ``"random"``.

    Raises:
        KeyError: When ``game`` is not in :data:`ATARI_BASELINES`.
    """
    if game not in ATARI_BASELINES:
        raise KeyError(
            f"No Atari baselines registered for {game!r}. "
            f"Known games: {sorted(ATARI_BASELINES)}."
        )
    return ATARI_BASELINES[game]
