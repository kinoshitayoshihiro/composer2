"""Probability conditioning helpers for groove sampling."""

from typing import Literal, Sequence
import math
import numpy as np


def apply_kick_pattern_bias(
    probs: np.ndarray,
    idx_to_state: Sequence[str],
    pos_quarter: int,
    policy: Literal["four_on_floor", "sparse", None] | None,
    *,
    bias_kick: float = 1.5,
    bias_other: float = 1.0,
) -> np.ndarray:
    """Apply kick pattern bias to probability distribution.

    Parameters
    ----------
    probs:
        Probability array over states.
    idx_to_state:
        Sequence of state labels corresponding to ``probs``.
    pos_quarter:
        Quarter position within the bar (0..3).  Currently unused but kept for
        future extensions and API stability.
    policy:
        Bias policy. ``four_on_floor`` boosts kick probability while
        suppressing others. ``sparse`` dampens kick probability.
    bias_kick:
        Logit boost applied to kick states in ``four_on_floor`` mode.
    bias_other:
        Logit penalty applied to non-kick states in ``four_on_floor`` mode.
    """

    if policy is None:
        return probs

    probs = probs.copy()
    if policy == "four_on_floor":
        boost = math.exp(bias_kick)
        suppress = math.exp(-bias_other)
        for i, lbl in enumerate(idx_to_state):
            if isinstance(lbl, str) and lbl == "kick":
                probs[i] *= boost
            else:
                probs[i] *= suppress
    elif policy == "sparse":
        for i, lbl in enumerate(idx_to_state):
            if isinstance(lbl, str) and lbl == "kick":
                probs[i] *= 0.4
    return probs


def apply_velocity_bias(
    probs: np.ndarray, level: Literal["soft", "hard", None] | None
) -> np.ndarray:
    """Apply a simple velocity bias by reweighting probabilities.

    This implementation uses Gaussian-like weights over the probability index
    to approximate velocity bins. ``soft`` emphasises lower bins whereas
    ``hard`` emphasises higher bins.
    """

    if level is None:
        return probs

    probs = probs.copy()
    n = len(probs)
    if n == 0:
        return probs
    x = np.linspace(-1.0, 1.0, n)
    if level == "soft":
        weights = np.exp(-((x + 0.5) ** 2) * 4)
    else:  # "hard"
        weights = np.exp(-((x - 0.5) ** 2) * 4)
    return probs * weights


def apply_style_bias(
    probs: np.ndarray,
    idx_to_state: Sequence[str],
    style: str | None,
) -> np.ndarray:
    """Very lightweight style-specific probability tweaks.

    ``lofi`` dampens cymbal-heavy choices while ``funk`` slightly boosts
    hi-hats. Styles not recognised leave ``probs`` untouched.
    """

    if not style:
        return probs
    probs = probs.copy()
    if style == "lofi":
        for i, lbl in enumerate(idx_to_state):
            if isinstance(lbl, str) and lbl in {"crash", "ride", "splash", "china"}:
                probs[i] *= 0.5
    elif style == "funk":
        for i, lbl in enumerate(idx_to_state):
            if isinstance(lbl, str) and lbl.startswith("hh_") and lbl != "hh_pedal":
                probs[i] *= 1.2
    return probs


def apply_feel_bias(
    probs: np.ndarray,
    idx_to_state: Sequence[str],
    pos_quarter: int,
    feel: str | None,
) -> np.ndarray:
    """Adjust probabilities to approximate musical "feel" such as swing."""

    if not feel or feel == "straight":
        return probs
    probs = probs.copy()
    if feel == "swing" and pos_quarter % 2 == 1:
        for i, lbl in enumerate(idx_to_state):
            if isinstance(lbl, str) and lbl.startswith("hh_") and lbl != "hh_pedal":
                probs[i] *= 1.2
    elif feel == "laidback" and pos_quarter % 2 == 0:
        for i, lbl in enumerate(idx_to_state):
            if isinstance(lbl, str) and lbl.startswith("hh_") and lbl != "hh_pedal":
                probs[i] *= 0.8
    return probs

