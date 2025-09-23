from __future__ import annotations

import struct
from typing import Sequence, Tuple

__all__ = ["murmur32", "hash_ctx", "murmur32_worker"]


def murmur32(data: bytes, seed: int = 0) -> int:
    """Return the MurmurHash3 32-bit hash of *data*.

    This pure-Python implementation is adapted for small payloads and is
    sufficient for testing purposes.  It matches the unsigned 32-bit output of
    the reference algorithm.
    """

    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    length = len(data)
    h1 = seed & 0xFFFFFFFF
    rounded_end = length & ~0x3

    for i in range(0, rounded_end, 4):
        k1 = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24)
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF

    k1 = 0
    tail_index = rounded_end
    tail_size = length & 0x3
    if tail_size == 3:
        k1 ^= data[tail_index + 2] << 16
    if tail_size >= 2:
        k1 ^= data[tail_index + 1] << 8
    if tail_size >= 1:
        k1 ^= data[tail_index]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= h1 >> 16

    return h1 & 0xFFFFFFFF


def hash_ctx(
    context_events: Sequence[int], aux: Tuple[int, int, int] | None = None
) -> int:
    """Hash a sequence of context events together with auxiliary IDs."""

    aux_vals: Sequence[int]
    if aux is None:
        aux_vals = ()
    else:
        aux_vals = aux
    # NOTE: ``H`` packs 16-bit unsigned values, matching the historic hash
    # format.  TODO: switch to 32-bit packing if future models emit IDs above
    # 65535 so the hash remains stable.
    data = struct.pack(
        f"<{len(context_events) + len(aux_vals)}H",
        *list(context_events) + list(aux_vals),
    )
    return murmur32(data) & 0xFFFFFFFF


def murmur32_worker(queue, data: bytes = b"abc") -> None:
    """Helper suitable for multiprocessing pickling tests."""

    queue.put(murmur32(data))
