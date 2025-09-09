from collections.abc import Callable
from typing import Any

Part = Any

apply_swing: Callable[[Part, float, int], None] | None
humanize_velocities: Callable[[Part, int, bool, bool, str, int], None] | None
apply_envelope: Callable[[Part, int, int, float], None] | None
apply_velocity_histogram: Callable[[Part, dict[int, float]], None] | None
timing_correct_part: Callable[[Part], None] | None
__all__: list[str]
