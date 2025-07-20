"""Utilities for mapping note durations to discrete buckets."""

# Upper bounds for each duration bucket in quarterLength units.  With ``n``
# boundaries there are ``n + 1`` buckets.  Durations below ``_BOUNDS[0]`` map to
# bucket ``0`` while values exceeding the last bound fall into the final bucket.
_BOUNDS = [0.25, 0.5, 1.5, 3.0, 6.0, 12.0, 24.0]


def to_bucket(qlen: float) -> int:
    for i, b in enumerate(_BOUNDS):
        if qlen < b:
            return i
    return len(_BOUNDS)


def bucket_duration(seconds: float, tempo: float) -> int:
    qlen = seconds * (tempo / 60.0)
    return to_bucket(qlen)


__all__ = ["to_bucket", "bucket_duration"]
