import argparse
import json
import warnings
from importlib import reload
from pathlib import Path
from typing import Dict

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pm = pytest.importorskip("pretty_midi")

from scripts import eval_duv, predict_duv


class _DummyDUV(torch.nn.Module):
    def __init__(self, max_len: int = 4) -> None:
        super().__init__()
        self.requires_duv_feats = True
        self.has_vel_head = True
        self.has_dur_head = True
        self.core = self
        self.max_len = max_len

    def forward(self, feats: Dict[str, torch.Tensor], *, mask: torch.Tensor | None = None):
        assert isinstance(feats, dict)
        assert mask is not None
        for key in ("pitch", "position", "velocity", "duration"):
            assert key in feats
            assert feats[key].shape == (1, self.max_len)
        length = int(mask.sum().item())
        vel = torch.linspace(0.0, 1.0, self.max_len, device=mask.device).unsqueeze(0)
        dur = torch.log1p(torch.arange(self.max_len, dtype=torch.float32, device=mask.device)).unsqueeze(0)
        if length < self.max_len:
            vel[:, length:] = 0.0
            dur[:, length:] = 0.0
        return {"velocity": vel, "duration": dur}


def test_duv_sequence_predict_builds_mask() -> None:
    df = pd.DataFrame(
        {
            "track_id": [0, 0, 0],
            "bar": [0, 0, 0],
            "position": [0, 1, 2],
            "pitch": [60, 62, 64],
            "velocity": [0.5, 0.4, 0.3],
            "duration": [0.1, 0.2, 0.3],
        }
    )
    model = _DummyDUV(max_len=4)
    preds = eval_duv._duv_sequence_predict(df, model, torch.device("cpu"))
    assert preds is not None
    assert preds["velocity_mask"].tolist() == [True, True, True]
    assert preds["duration_mask"].tolist() == [True, True, True]
    assert preds["velocity"].shape == (len(df),)
    assert preds["duration"].shape == (len(df),)
    expected_vel = [1.0, 42.0, 85.0]
    np.testing.assert_allclose(preds["velocity"][:3], expected_vel, rtol=0, atol=1)
    np.testing.assert_allclose(preds["duration"][:3], [0.0, 1.0, 2.0], rtol=0, atol=1e-6)


def test_duv_sequence_predict_warns_without_bar() -> None:
    df = pd.DataFrame(
        {
            "track_id": [0, 0],
            "position": [0, 1],
            "pitch": [60, 62],
            "velocity": [0.5, 0.6],
            "duration": [0.2, 0.3],
        }
    )
    model = _DummyDUV(max_len=2)
    with pytest.warns(RuntimeWarning, match="bar segmentation"):
        eval_duv._duv_sequence_predict(df, model, torch.device("cpu"))


def test_duv_sequence_predict_verbose_optional_summary(capsys: pytest.CaptureFixture[str]) -> None:
    df = pd.DataFrame(
        {
            "track_id": [0, 0],
            "bar": [0, 0],
            "position": [0, 1],
            "pitch": [60, 62],
            "velocity": [0.4, 0.5],
            "duration": [0.2, 0.3],
        }
    )
    model = _DummyDUV(max_len=2)
    model.use_bar_beat = True
    model.section_emb = object()
    model.mood_emb = object()
    model.vel_bucket_emb = object()
    model.dur_bucket_emb = object()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval_duv._duv_sequence_predict(df, model, torch.device("cpu"), verbose=True)
    captured = capsys.readouterr()
    assert "duv_optional_features" in captured.err
    assert "zero_fill" in captured.err


class _StubModel:
    requires_duv_feats = True

    def to(self, device):
        return self

    def eval(self):
        return self


class _EvalStubModel:
    requires_duv_feats = True
    has_vel_head = True
    has_dur_head = False

    def __init__(self, max_len: int = 4) -> None:
        self.core = self
        self.d_model = 0
        self.max_len = max_len

    def to(self, _device):
        return self

    def eval(self):
        return self


def _write_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "track_id": [0, 0, 0],
            "bar": [0, 0, 0],
            "position": [0, 1, 2],
            "pitch": [60, 62, 64],
            "velocity": [30, 10, 30],
            "duration": [0.5, 0.5, 0.5],
            "start": [0.0, 0.5, 1.0],
        }
    )
    df.to_csv(path, index=False)


def _stub_stats(*_args, **_kwargs):
    return ([], np.array([], dtype=np.float32), np.array([], dtype=np.float32), {})


def _stub_duv_preds() -> dict[str, np.ndarray]:
    return {
        "velocity": np.array([100.0, 50.0, 80.0], dtype=np.float32),
        "velocity_mask": np.array([True, False, True]),
        "duration": np.zeros(3, dtype=np.float32),
        "duration_mask": np.zeros(3, dtype=bool),
    }


def _run_predict(tmp_path: Path, smooth_pred_only: bool) -> pm.PrettyMIDI:
    csv_path = tmp_path / "notes.csv"
    _write_csv(csv_path)
    args = argparse.Namespace(
        csv=csv_path,
        ckpt=tmp_path / "model.ckpt",
        out=tmp_path / ("out_true.mid" if smooth_pred_only else "out_false.mid"),
        batch=2,
        device="cpu",
        stats_json=tmp_path / "stats.json",
        num_workers=0,
        vel_smooth=3,
        smooth_pred_only=smooth_pred_only,
        dur_quant=None,
        filter_program=None,
        verbose=False,
        limit=0,
    )
    args.stats_json.write_text(json.dumps({"feat_cols": [], "mean": [], "std": []}))

    predict_duv._load_stats = _stub_stats  # type: ignore[attr-defined]
    predict_duv._duv_sequence_predict = (  # type: ignore[attr-defined]
        lambda _df, _model, _dev, **_kwargs: _stub_duv_preds()
    )
    predict_duv.MLVelocityModel.load = lambda _path: _StubModel()  # type: ignore[assignment]
    try:
        predict_duv.run(args)
    finally:
        reload(predict_duv)
    return pm.PrettyMIDI(str(args.out))


def test_predict_duv_filter_program_resets_index(tmp_path: Path) -> None:
    csv_path = tmp_path / "notes.csv"
    _write_csv(csv_path)
    args = argparse.Namespace(
        csv=csv_path,
        ckpt=tmp_path / "model.ckpt",
        out=tmp_path / "out_filter.mid",
        batch=2,
        device="cpu",
        stats_json=tmp_path / "stats.json",
        num_workers=0,
        vel_smooth=1,
        smooth_pred_only=True,
        dur_quant=None,
        filter_program="position > 0",
        verbose=False,
        limit=0,
    )
    args.stats_json.write_text(json.dumps({"feat_cols": [], "mean": [], "std": []}))

    captured: list[pd.DataFrame] = []

    def _duv_stub(df: pd.DataFrame, *_args, **_kwargs):  # type: ignore[override]
        captured.append(df.copy())
        n = len(df)
        values = np.linspace(10.0, 120.0, num=n, dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        mask = np.ones(n, dtype=bool)
        return {
            "velocity": values,
            "velocity_mask": mask,
            "duration": np.zeros(n, dtype=np.float32),
            "duration_mask": np.zeros(n, dtype=bool),
        }

    predict_duv._load_stats = _stub_stats  # type: ignore[attr-defined]
    predict_duv._duv_sequence_predict = _duv_stub  # type: ignore[attr-defined]
    predict_duv.MLVelocityModel.load = lambda _path: _StubModel()  # type: ignore[assignment]
    try:
        predict_duv.run(args)
    finally:
        reload(predict_duv)

    assert captured, "DUV predictor was not invoked"
    assert captured[0].index.tolist() == list(range(len(captured[0]))), "DataFrame index was not reset"
    midi = pm.PrettyMIDI(str(args.out))
    assert len(midi.instruments[0].notes) == len(captured[0])


def test_eval_duv_filter_and_limit(tmp_path: Path) -> None:
    csv_path = tmp_path / "notes.csv"
    df = pd.DataFrame(
        {
            "track_id": [0] * 6,
            "file": ["stub.mid"] * 6,
            "bar": [0, 0, 0, 1, 1, 1],
            "position": [0, 1, 2, 3, 4, 5],
            "pitch": [60, 61, 62, 63, 64, 65],
            "velocity": [40, 50, 60, 70, 80, 90],
            "duration": [0.5] * 6,
        }
    )
    df.to_csv(csv_path, index=False)
    args = argparse.Namespace(
        csv=csv_path,
        ckpt=tmp_path / "model.ckpt",
        stats_json=tmp_path / "stats.json",
        batch=2,
        device="cpu",
        dur_quant=None,
        num_workers=0,
        verbose=False,
        limit=5,
        filter_program="position >= 1",
        out_json=tmp_path / "metrics.json",
    )
    args.stats_json.write_text(json.dumps({"feat_cols": [], "mean": [], "std": []}))

    captured: list[pd.DataFrame] = []

    def _duv_stub(df_in: pd.DataFrame, *_args, **_kwargs):  # type: ignore[override]
        captured.append(df_in.copy())
        n = len(df_in)
        values = np.linspace(10.0, 110.0, num=n, dtype=np.float32) if n else np.zeros(0, dtype=np.float32)
        mask = np.ones(n, dtype=bool)
        return {
            "velocity": values,
            "velocity_mask": mask,
            "duration": np.zeros(n, dtype=np.float32),
            "duration_mask": np.zeros(n, dtype=bool),
        }

    eval_duv._load_stats = _stub_stats  # type: ignore[attr-defined]
    eval_duv._duv_sequence_predict = _duv_stub  # type: ignore[attr-defined]
    eval_duv.MLVelocityModel.load = lambda _path: _EvalStubModel(max_len=4)  # type: ignore[assignment]
    try:
        rc = eval_duv.run(args)
    finally:
        reload(eval_duv)

    assert rc == 0
    assert captured, "DUV predictor was not invoked"
    first = captured[0]
    assert first.index.tolist() == list(range(len(first)))
    assert len(first) <= args.limit
    metrics_path = args.out_json
    data = json.loads(metrics_path.read_text().strip())
    assert "velocity_mae" in data


def test_eval_duv_logs_rows_and_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = tmp_path / "notes.csv"
    df = pd.DataFrame(
        {
            "track_id": [0, 0, 0],
            "bar": [0, 0, 0],
            "position": [0, 1, 2],
            "pitch": [60, 61, 62],
            "velocity": [40, 50, 60],
            "duration": [0.5, 0.5, 0.5],
        }
    )
    df.to_csv(csv_path, index=False)
    args = argparse.Namespace(
        csv=csv_path,
        ckpt=tmp_path / "model.ckpt",
        stats_json=tmp_path / "stats.json",
        batch=2,
        device="cpu",
        dur_quant=None,
        num_workers=0,
        verbose=False,
        limit=2,
        filter_program=None,
        out_json=None,
    )
    args.stats_json.write_text(json.dumps({"feat_cols": [], "mean": [], "std": []}))

    def _duv_stub(df_in: pd.DataFrame, *_args, **_kwargs):  # type: ignore[override]
        n = len(df_in)
        return {
            "velocity": np.full(n, 64.0, dtype=np.float32),
            "velocity_mask": np.ones(n, dtype=bool),
            "duration": np.zeros(n, dtype=np.float32),
            "duration_mask": np.zeros(n, dtype=bool),
        }

    eval_duv._load_stats = _stub_stats  # type: ignore[attr-defined]
    eval_duv._duv_sequence_predict = _duv_stub  # type: ignore[attr-defined]
    eval_duv.MLVelocityModel.load = lambda _path: _EvalStubModel(max_len=4)  # type: ignore[assignment]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rc = eval_duv.run(args)
    finally:
        reload(eval_duv)

    assert rc == 0
    outerr = capsys.readouterr()
    assert "'rows': 2" in outerr.err
    assert "'limit': 2" in outerr.err
    assert outerr.out.strip(), "metrics JSON was not printed"


def test_eval_duv_constant_predictions_null_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "notes.csv"
    df = pd.DataFrame(
        {
            "track_id": [0, 0, 0],
            "bar": [0, 0, 0],
            "position": [0, 1, 2],
            "pitch": [60, 61, 62],
            "velocity": [50, 55, 60],
            "duration": [0.5, 0.6, 0.7],
        }
    )
    df.to_csv(csv_path, index=False)
    args = argparse.Namespace(
        csv=csv_path,
        ckpt=tmp_path / "model.ckpt",
        stats_json=tmp_path / "stats.json",
        batch=2,
        device="cpu",
        dur_quant=None,
        num_workers=0,
        verbose=False,
        limit=0,
        filter_program=None,
        out_json=None,
    )
    args.stats_json.write_text(json.dumps({"feat_cols": [], "mean": [], "std": []}))

    def _duv_stub(df_in: pd.DataFrame, *_args, **_kwargs):  # type: ignore[override]
        n = len(df_in)
        return {
            "velocity": np.full(n, 80.0, dtype=np.float32),
            "velocity_mask": np.ones(n, dtype=bool),
            "duration": np.zeros(n, dtype=np.float32),
            "duration_mask": np.zeros(n, dtype=bool),
        }

    eval_duv._load_stats = _stub_stats  # type: ignore[attr-defined]
    eval_duv._duv_sequence_predict = _duv_stub  # type: ignore[attr-defined]
    eval_duv.MLVelocityModel.load = lambda _path: _EvalStubModel(max_len=4)  # type: ignore[assignment]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rc = eval_duv.run(args)
    finally:
        reload(eval_duv)

    assert rc == 0
    assert not any(
        getattr(warning.category, "__name__", "") == "ConstantInputWarning" for warning in caught
    ), "ConstantInputWarning was emitted"
    outerr = capsys.readouterr()
    metrics = json.loads(outerr.out.strip())
    assert metrics["velocity_pearson"] is None
    assert metrics["velocity_spearman"] is None
    assert metrics.get("velocity_stats_note") == "constant_prediction"


def test_predict_duv_smooth_pred_only(tmp_path: Path) -> None:
    midi = _run_predict(tmp_path, smooth_pred_only=True)
    velocities = [n.velocity for n in midi.instruments[0].notes]
    assert velocities == [100, 10, 80]


def test_predict_duv_smooth_all_when_disabled(tmp_path: Path) -> None:
    midi = _run_predict(tmp_path, smooth_pred_only=False)
    velocities = [n.velocity for n in midi.instruments[0].notes]
    assert velocities == [100, 80, 80]
