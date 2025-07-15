"""Utility functions for handling vocal rest metrics."""


def get_rest_windows(
    vocal_metrics: dict | None, min_dur: float = 0.5
) -> list[tuple[float, float]]:
    """Return [start, end] tuples for rests of at least ``min_dur`` beats."""
    if not vocal_metrics:
        return []
    return [(s, s + d) for s, d in vocal_metrics.get("rests", []) if d >= min_dur]
