"""Optional Cython extensions for Modular Composer."""

try:
    from .humanize import (
        apply_envelope,
        apply_swing,
        apply_velocity_histogram,
        humanize_velocities,
        timing_correct_part,
    )
except Exception:  # pragma: no cover
    apply_swing = humanize_velocities = apply_envelope = timing_correct_part = None
    apply_velocity_histogram = None

try:
    from .generators import (
        apply_velocity_curve,
        postprocess_kick_lock,
        velocity_random_walk,
    )
except Exception:  # pragma: no cover
    postprocess_kick_lock = velocity_random_walk = apply_velocity_curve = None

__all__ = [
    'apply_swing',
    'humanize_velocities',
    'apply_envelope',
    'apply_velocity_histogram',
    'timing_correct_part',
    'postprocess_kick_lock',
    'velocity_random_walk',
    'apply_velocity_curve',
]
