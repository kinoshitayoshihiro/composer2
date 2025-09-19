from __future__ import annotations

"""Lightweight CSV helpers shared across composer2 scripts."""

from typing import Iterable

import numpy as np
import pandas as pd


def coerce_columns(
    df: pd.DataFrame,
    *,
    float32: Iterable[str] | None = None,
    int32: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Cast selected DataFrame columns to ``np.float32``/``np.int32`` in-place."""

    float_set = set(float32 or ())
    int_set = set(int32 or ())
    cols = set(df.columns)
    for column in float_set & cols:
        df[column] = df[column].astype(np.float32, copy=False)
    for column in int_set & cols:
        df[column] = df[column].astype(np.int32, copy=False)
    return df


__all__ = ["coerce_columns"]
