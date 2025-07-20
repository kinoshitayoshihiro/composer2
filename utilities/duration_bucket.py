from __future__ import annotations

EDGES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def to_bucket(q_len: float) -> int:
    """Return duration bucket index for ``q_len`` quarterLength."""
    for i, edge in enumerate(EDGES):
        if q_len <= edge:
            return i
    return len(EDGES)


__all__ = ["to_bucket", "EDGES"]
