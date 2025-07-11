from typing import Any

Part = Any

def humanize_velocities(
    part_stream: Part,
    amount: int = ...,
    use_expr_cc11: bool = ...,
    use_aftertouch: bool = ...,
    expr_curve: str = ...,
    kick_leak_jitter: int = ...,
) -> None: ...

def apply_envelope(part: Part, start: int, end: int, scale: float) -> None: ...

def apply_swing(part: Part, ratio: float, subdiv: int) -> None: ...

def apply_velocity_histogram(part: Part, histogram: dict[int, float]) -> None: ...
