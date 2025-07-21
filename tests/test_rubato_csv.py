import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pretty_midi")
import pretty_midi

from utilities import rubato_csv


@pytest.fixture
def beats_4() -> np.ndarray:
    return np.array([0.0, 0.5, 1.0, 1.5], dtype=float)


def test_beat_index(monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray) -> None:
    pm = pretty_midi.PrettyMIDI()

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    df = rubato_csv.extract_tempo_curve(pm, score_bpm=120, use_librosa=False)
    assert df["beat"].dtype == np.int32
    assert np.array_equal(df["beat"].values, np.arange(len(df), dtype=np.int32))


def test_track_id_dtype(monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray) -> None:
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.extend([pretty_midi.Instrument(0), pretty_midi.Instrument(0)])

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    df = rubato_csv.extract_tempo_curve(
        pm, score_bpm=120, track_ids=[1], use_librosa=False
    )
    assert df["track_id"].dtype == np.int32
    assert np.all(df["track_id"].values == 1)


def test_tempo_factor_dtype(
    monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray
) -> None:
    pm = pretty_midi.PrettyMIDI()

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    df = rubato_csv.extract_tempo_curve(pm, score_bpm=120, use_librosa=False)
    assert df["tempo_factor"].dtype == np.float32


def test_csv_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray
) -> None:
    pm = pretty_midi.PrettyMIDI()

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    out = tmp_path / "out.csv"
    path = rubato_csv.extract_tempo_curve(
        pm, score_bpm=120, return_df=False, out_path=out, use_librosa=False
    )
    assert path == out
    df = pd.read_csv(path)
    assert not df.empty


def test_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray
) -> None:
    pm = pretty_midi.PrettyMIDI()

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    out = tmp_path / "out.json"
    path = rubato_csv.extract_tempo_curve(
        pm,
        score_bpm=120,
        return_df=False,
        out_path=out,
        as_json=True,
        use_librosa=False,
    )
    assert path == out
    df = pd.read_json(path)
    assert not df.empty


def test_out_path_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray
) -> None:
    pm = pretty_midi.PrettyMIDI()

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    nested = tmp_path / "nested" / "out.csv"
    path = rubato_csv.extract_tempo_curve(
        pm,
        score_bpm=120,
        return_df=False,
        out_path=nested,
        use_librosa=False,
    )
    assert path == nested
    assert path.exists()


def test_bad_track_id(monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray) -> None:
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(0))

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    with pytest.raises(ValueError, match="invalid track_ids"):
        rubato_csv.extract_tempo_curve(
            pm,
            score_bpm=120,
            track_ids=[1],
            use_librosa=False,
        )


def test_negative_track_id(
    monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray
) -> None:
    pm = pretty_midi.PrettyMIDI()

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    df = rubato_csv.extract_tempo_curve(
        pm,
        score_bpm=120,
        track_ids=[-2],
        use_librosa=False,
    )
    assert np.all(df["track_id"].values == 2)


def test_eighth_note_unit(monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray) -> None:
    pm = pretty_midi.PrettyMIDI()

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    df = rubato_csv.extract_tempo_curve(
        pm, score_bpm=120, beat_unit=8, use_librosa=False
    )
    assert np.allclose(df["time_sec"].values, [0.0, 0.5, 1.0, 1.5])
    assert np.array_equal(df["beat"].values, [0, 2, 4, 6])
    # eighth-note grid has step 2 between beats
    assert np.all(np.diff(df["beat"].values) == 2)
    assert np.allclose(df["tempo_factor"].values, -0.5)


def test_scipy_missing(
    beats_4: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pm = pretty_midi.PrettyMIDI()

    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)
    monkeypatch.setattr(rubato_csv, "savgol_filter", None)
    with caplog.at_level(logging.WARNING):
        df = rubato_csv.extract_tempo_curve(
            pm, score_bpm=120, smoothing_window_beats=2, use_librosa=False
        )
    assert "SciPy not available" in caplog.text
    assert len(df) == 4


def test_librosa_missing(monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray) -> None:
    pm = pretty_midi.PrettyMIDI()

    def beats_stub(
        _: pretty_midi.PrettyMIDI, __: float, *, use_librosa: bool = True
    ) -> np.ndarray:
        if rubato_csv.librosa is None or not use_librosa:
            raise RuntimeError("librosa is required for beat tracking")
        return np.array([0.0, 0.5, 1.0, 1.5], dtype=float)

    monkeypatch.setattr(rubato_csv, "_detect_beats", beats_stub)
    monkeypatch.setattr(rubato_csv, "librosa", None)

    with pytest.raises(RuntimeError, match="librosa is required"):
        rubato_csv.extract_tempo_curve(pm, score_bpm=120)


def test_librosa_disabled(monkeypatch: pytest.MonkeyPatch, beats_4: np.ndarray) -> None:
    pm = pretty_midi.PrettyMIDI()

    def no_beats(
        _: pretty_midi.PrettyMIDI, __: float, *, use_librosa: bool = True
    ) -> np.ndarray:
        assert use_librosa is False
        raise RuntimeError("librosa disabled; cannot track beats")

    monkeypatch.setattr(rubato_csv, "_detect_beats", no_beats)

    with pytest.raises(RuntimeError, match="disabled"):
        rubato_csv.extract_tempo_curve(pm, score_bpm=120, use_librosa=False)


def test_smoothing_effect(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pm = pretty_midi.PrettyMIDI()

    def variable_beats(
        _: pretty_midi.PrettyMIDI, __: float, *, use_librosa: bool = True
    ) -> np.ndarray:
        return np.array([0.0, 0.55, 1.0, 1.55], dtype=float)

    if rubato_csv.savgol_filter is None:
        pytest.skip("SciPy not available")

    monkeypatch.setattr(rubato_csv, "_detect_beats", variable_beats)

    df_plain = rubato_csv.extract_tempo_curve(
        pm, score_bpm=120, smoothing_window_beats=0, use_librosa=False
    )

    monkeypatch.setattr(
        rubato_csv,
        "savgol_filter",
        lambda arr, w, p, mode="nearest": np.zeros_like(arr),
    )

    with caplog.at_level(logging.WARNING):
        df_smooth = rubato_csv.extract_tempo_curve(
            pm,
            score_bpm=120,
            smoothing_window_beats=1,
            polyorder=5,
            use_librosa=False,
        )
    assert "clamping polyorder" in caplog.text

    assert df_plain["tempo_factor"].std() > df_smooth["tempo_factor"].std()


def test_polyorder_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    midi = pretty_midi.PrettyMIDI()
    midi_path = tmp_path / "test.mid"
    midi.write(str(midi_path))

    out = tmp_path / "out.csv"
    captured = {}

    import argparse

    real_parse = argparse.ArgumentParser.parse_args

    def capture(self: argparse.ArgumentParser, args=None, namespace=None):
        res = real_parse(self, args, namespace)
        captured.update(vars(res))
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rubato_csv",
            str(midi_path),
            "--poly",
            "3",
            "--out",
            str(out),
            "--json",
            "--overwrite",
            "--no-librosa",
        ],
    )
    with pytest.raises(SystemExit):
        rubato_csv._cli()
    assert captured.get("polyorder") == 3
    assert captured.get("as_json") is True


def test_cli_overwrite_exit(tmp_path: Path) -> None:
    midi = tmp_path / "test.mid"
    pretty_midi.PrettyMIDI().write(str(midi))
    out = tmp_path / "out.csv"
    out.write_text("x")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "utilities.rubato_csv",
            str(midi),
            "--out",
            str(out),
            "--no-librosa",
        ],
        capture_output=True,
    )
    assert result.returncode == 1


def test_extraction_summary_respects_log_level(
    beats_4: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pm = pretty_midi.PrettyMIDI()
    monkeypatch.setattr(rubato_csv, "_detect_beats", lambda *_a, **_k: beats_4)

    rubato_csv.configure_logging(logging.INFO)
    with caplog.at_level(logging.INFO):
        rubato_csv.extract_tempo_curve(pm, score_bpm=120, use_librosa=False)
    assert "Extraction finished" in caplog.text

    caplog.clear()
    rubato_csv.configure_logging(logging.WARNING)
    with caplog.at_level(logging.WARNING):
        rubato_csv.extract_tempo_curve(pm, score_bpm=120, use_librosa=False)
    assert "Extraction finished" not in caplog.text
