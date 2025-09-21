from __future__ import annotations

import torch

__all__ = ["denorm_duv"]


def denorm_duv(vel_reg: torch.Tensor, dur_reg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Denormalize velocity and duration predictions.

    Parameters
    ----------
    vel_reg:
        Normalized velocity in ``[0, 1]``.
    dur_reg:
        Log-duration produced by ``log1p(duration)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        MIDI velocity ``[0, 127]`` and duration in original units.
    """

    if not isinstance(vel_reg, torch.Tensor):
        vel_reg = torch.tensor(vel_reg)
    if not isinstance(dur_reg, torch.Tensor):
        dur_reg = torch.tensor(dur_reg)

    vel = vel_reg.mul(127.0).round().clamp(0, 127)
    dur = torch.expm1(dur_reg)
    return vel, dur
