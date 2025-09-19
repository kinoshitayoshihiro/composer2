import argparse
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from scripts import predict_duv


class _StubModel:
    has_vel_head = True
    has_dur_head = True
    requires_duv_feats = False

    def __init__(self) -> None:
        core = argparse.Namespace(
            use_bar_beat=False,
            section_emb=None,
            mood_emb=None,
            vel_bucket_emb=None,
            dur_bucket_emb=None,
            d_model=0,
            max_len=0,
        )
        self.core = core
        self.d_model = 0
        self.max_len = 0
        self.heads = {"vel_reg": True, "dur_reg": True}

    @staticmethod
    def load(path: str) -> "_StubModel":
        return _StubModel()

    def to(self, device: object) -> "_StubModel":
        return self

    def eval(self) -> "_StubModel":
        return self

    def __call__(self, tensor: object) -> np.ndarray:
        return np.zeros((0, 0), dtype=np.float32)


class _StubPrettyMIDI:
    class Instrument:
        def __init__(self, program: int = 0) -> None:
            self.program = program
            self.notes: list[_StubPrettyMIDI.Note] = []

    class Note:
        def __init__(self, velocity: int, pitch: int, start: float, end: float) -> None:
            self.velocity = velocity
            self.pitch = pitch
            self.start = start
            self.end = end

    def __init__(self) -> None:
        self.instruments: list[_StubPrettyMIDI.Instrument] = []

    def write(self, path: str) -> None:
        Path(path).write_bytes(b"")


@pytest.fixture(autouse=True)
def _patch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict_duv.MLVelocityModel, "load", staticmethod(_StubModel.load))
    monkeypatch.setattr(predict_duv, "_duv_sequence_predict", _duv_stub)
    monkeypatch.setattr(predict_duv, "load_stats_and_normalize", lambda *a, **k: (np.zeros((0, 0), dtype=np.float32), []))
    monkeypatch.setattr(predict_duv, "_load_duration_model", lambda *_a, **_k: _StubModel())
    monkeypatch.setattr(predict_duv.pm, "PrettyMIDI", _StubPrettyMIDI)
    monkeypatch.setattr(predict_duv.pm, "Instrument", _StubPrettyMIDI.Instrument)
    monkeypatch.setattr(predict_duv.pm, "Note", _StubPrettyMIDI.Note)


def _duv_stub(df: pd.DataFrame, *_args, **_kwargs) -> dict[str, np.ndarray]:
    n = len(df)
    velocity = np.full(n, 64, dtype=np.float32)
    duration = np.full(n, 0.5, dtype=np.float32)
    mask = np.ones(n, dtype=bool)
    return {
        "velocity": velocity,
        "velocity_mask": mask,
        "duration": duration,
        "duration_mask": mask,
    }


def test_filter_program_missing_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "notes.csv"
    df = pd.DataFrame(
        {
            "pitch": [60, 62],
            "velocity": [40, 50],
            "duration": [0.5, 0.5],
            "position": [0, 1],
            "bar": [0, 0],
        }
    )
    df.to_csv(csv_path, index=False)

    stats = ([], np.array([], dtype=np.float32), np.array([], dtype=np.float32), {})
    monkeypatch.setattr(predict_duv, "_load_stats", lambda *a, **k: stats)

    args = argparse.Namespace(
        csv=csv_path,
        ckpt=tmp_path / "model.ckpt",
        out=tmp_path / "out.mid",
        batch=2,
        device="cpu",
        stats_json=tmp_path / "stats.json",
        num_workers=0,
        vel_smooth=1,
        smooth_pred_only=True,
        dur_quant=None,
        filter_program="program == 0",
        verbose=True,
        limit=0,
    )

    rc = predict_duv.run(args)
    assert rc == 0
    result_df, _ = predict_duv.load_duv_dataframe(
        csv_path,
        feature_columns=[],
        filter_expr="program == 0",
    )
    assert "program" in result_df.columns
    assert result_df.empty
