from pathlib import Path

import pandas as pd

from utilities.merge_artic_features import (
    merge_artic_features,
    split_artic_features,
)


def _write(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def test_round_trip_merge_split(tmp_path: Path) -> None:
    vel = pd.DataFrame({
        "track_id": [0],
        "onset": [0.0],
        "pitch": [60],
        "velocity": [90],
    })
    dur = pd.DataFrame({
        "track_id": [0],
        "onset": [0.0],
        "pitch": [60],
        "duration": [0.5],
    })
    ped = pd.DataFrame({
        "track_id": [0],
        "onset": [0.0],
        "pitch": [60],
        "pedal_state": [1],
    })
    v_path = _write(vel, tmp_path / "velocity.csv")
    d_path = _write(dur, tmp_path / "duration.csv")
    p_path = _write(ped, tmp_path / "pedal.csv")

    merged = merge_artic_features([v_path, d_path], [p_path])
    out_v = tmp_path / "v_out.csv"
    out_d = tmp_path / "d_out.csv"
    out_p = tmp_path / "p_out.csv"
    split_artic_features(
        merged,
        {out_v: ["velocity", "duration"], out_d: []},
        {out_p: ["pedal_state"]},
    )
    assert pd.read_csv(out_v).equals(pd.read_csv(v_path).merge(pd.read_csv(d_path), on=["track_id", "onset", "pitch"]))
    assert pd.read_csv(out_p).equals(pd.read_csv(p_path))


def test_collision_raises(tmp_path: Path) -> None:
    a = pd.DataFrame({
        "track_id": [0],
        "onset": [0.0],
        "pitch": [60],
        "velocity": [90],
    })
    b = pd.DataFrame({
        "track_id": [0],
        "onset": [0.0],
        "pitch": [60],
        "velocity": [91],
    })
    path_a = _write(a, tmp_path / "a.csv")
    path_b = _write(b, tmp_path / "b.csv")
    try:
        merge_artic_features([path_a, path_b], [])
    except ValueError:
        pass
    else:  # pragma: no cover
        assert False, "expected ValueError"


def test_rounding(tmp_path: Path) -> None:
    vel = pd.DataFrame({
        "track_id": [0],
        "onset": [0.123456789],
        "pitch": [60],
        "velocity": [90],
    })
    path_v = _write(vel, tmp_path / "v.csv")
    df = merge_artic_features([path_v], [], float_rnd=1e-5)
    assert df["onset"].iloc[0] == round(0.123456789, 5)
