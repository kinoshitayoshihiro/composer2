from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest


def _install_numpy_stub() -> None:
    module = sys.modules.get("numpy")
    if module is None:
        module = types.ModuleType("numpy")
        sys.modules["numpy"] = module
    module.float32 = "float32"
    module.int32 = "int32"
    module.ndarray = object
    module.zeros = lambda *a, **k: None  # type: ignore[assignment]
    module.empty_like = lambda *a, **k: None  # type: ignore[assignment]
    module.concatenate = lambda seq, axis=0: seq  # type: ignore[assignment]
    module.clip = lambda *a, **k: None  # type: ignore[assignment]
    module.pad = lambda *a, **k: None  # type: ignore[assignment]
    module.median = lambda *a, **k: None  # type: ignore[assignment]
    module.maximum = lambda *a, **k: None  # type: ignore[assignment]
    module.round = lambda *a, **k: None  # type: ignore[assignment]
    try:
        from importlib.machinery import ModuleSpec

        module.__spec__ = ModuleSpec("numpy", loader=None)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback for limited importlib
        module.__spec__ = None  # type: ignore[attr-defined]


class _DummyDataFrame:
    def __init__(self, columns: tuple[str, ...], rows: list[dict[str, object]] | None = None):
        self.columns = list(columns)
        self._rows = rows or []

    @property
    def empty(self) -> bool:
        return not self._rows

    def reset_index(self, drop: bool = True):  # noqa: D401 - simple stub
        return self

    def query(self, expr: str, engine: str = "python"):
        raise UndefinedVariableError("name 'program' is not defined")

    def get(self, key: str, default=None):
        return default

    def __len__(self) -> int:
        return len(self._rows)


class UndefinedVariableError(NameError):
    pass


def _install_pandas_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    pandas_module = sys.modules.get("pandas")
    if pandas_module is None:
        pandas_module = types.ModuleType("pandas")
        sys.modules["pandas"] = pandas_module

    def read_csv(*args, nrows=None, **kwargs):
        if nrows == 0:
            return _DummyDataFrame(("pitch",))
        return _DummyDataFrame(("pitch",), rows=[{"pitch": 60}])

    pandas_module.read_csv = read_csv  # type: ignore[assignment]
    pandas_module.factorize = lambda values: ([], [])  # type: ignore[assignment]

    errors_module = sys.modules.get("pandas.errors")
    if errors_module is None:
        errors_module = types.ModuleType("pandas.errors")
        sys.modules["pandas.errors"] = errors_module
    errors_module.UndefinedVariableError = UndefinedVariableError
    pandas_module.errors = errors_module  # type: ignore[assignment]

    try:
        from importlib.machinery import ModuleSpec

        pandas_module.__spec__ = ModuleSpec("pandas", loader=None)  # type: ignore[attr-defined]
        errors_module.__spec__ = ModuleSpec("pandas.errors", loader=None)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback for limited importlib
        pandas_module.__spec__ = None  # type: ignore[attr-defined]
        errors_module.__spec__ = None  # type: ignore[attr-defined]


def _install_pretty_midi_stub() -> None:
    module = sys.modules.get("pretty_midi")
    if module is None:
        module = types.ModuleType("pretty_midi")
        sys.modules["pretty_midi"] = module

    class _Note:  # pragma: no cover - never used, placeholder for import
        def __init__(self, **kwargs):
            pass

    class _Instrument:  # pragma: no cover - never used, placeholder for import
        def __init__(self, program=0):
            self.notes = []

        def append(self, value):
            self.notes.append(value)

    class _PrettyMIDI:  # pragma: no cover - never used, placeholder for import
        def __init__(self):
            self.instruments = []

        def write(self, path):
            pass

        def get_tempo_changes(self):
            return [], []

    module.Note = _Note
    module.Instrument = _Instrument
    module.PrettyMIDI = _PrettyMIDI

    try:
        from importlib.machinery import ModuleSpec

        module.__spec__ = ModuleSpec("pretty_midi", loader=None)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback for limited importlib
        module.__spec__ = None  # type: ignore[attr-defined]


def _install_torch_stub() -> None:
    module = sys.modules.get("torch")
    if module is None:
        module = types.ModuleType("torch")
        sys.modules["torch"] = module

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    module.no_grad = lambda: _NoGrad()  # type: ignore[assignment]
    module.from_numpy = lambda *a, **k: None  # type: ignore[assignment]

    try:
        from importlib.machinery import ModuleSpec

        module.__spec__ = ModuleSpec("torch", loader=None)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback for limited importlib
        module.__spec__ = None  # type: ignore[attr-defined]

    utils_module = sys.modules.get("torch.utils")
    if utils_module is None:
        utils_module = types.ModuleType("torch.utils")
        sys.modules["torch.utils"] = utils_module

    data_module = sys.modules.get("torch.utils.data")
    if data_module is None:
        data_module = types.ModuleType("torch.utils.data")
        sys.modules["torch.utils.data"] = data_module

    class _DummyDataLoader:  # pragma: no cover - placeholder
        def __init__(self, *args, **kwargs):
            pass

    class _DummyTensorDataset:  # pragma: no cover - placeholder
        def __init__(self, *args, **kwargs):
            pass

    data_module.DataLoader = _DummyDataLoader  # type: ignore[attr-defined]
    data_module.TensorDataset = _DummyTensorDataset  # type: ignore[attr-defined]
    utils_module.data = data_module  # type: ignore[attr-defined]

    try:
        from importlib.machinery import ModuleSpec

        utils_module.__spec__ = ModuleSpec("torch.utils", loader=None)  # type: ignore[attr-defined]
        data_module.__spec__ = ModuleSpec("torch.utils.data", loader=None)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback for limited importlib
        utils_module.__spec__ = None  # type: ignore[attr-defined]
        data_module.__spec__ = None  # type: ignore[attr-defined]


def _prepare_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_numpy_stub()
    _install_pretty_midi_stub()
    _install_torch_stub()
    _install_pandas_stub(monkeypatch)
    _install_utilities_stubs()


def _install_utilities_stubs() -> None:
    utilities_module = sys.modules.get("utilities")
    if utilities_module is None:
        utilities_module = types.ModuleType("utilities")
        utilities_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["utilities"] = utilities_module

    def _set_spec(name: str) -> None:
        try:
            from importlib.machinery import ModuleSpec

            sys.modules[name].__spec__ = ModuleSpec(name, loader=None)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback for limited importlib
            sys.modules[name].__spec__ = None  # type: ignore[attr-defined]

    csv_module = types.ModuleType("utilities.csv_io")
    csv_module.coerce_columns = lambda df, **kwargs: df  # type: ignore[assignment]
    sys.modules["utilities.csv_io"] = csv_module
    _set_spec("utilities.csv_io")

    duv_module = types.ModuleType("utilities.duv_infer")
    duv_module.CSV_FLOAT32_COLUMNS = set()
    duv_module.CSV_INT32_COLUMNS = set()
    duv_module.OPTIONAL_COLUMNS = set()
    duv_module.OPTIONAL_FLOAT32_COLUMNS = set()
    duv_module.OPTIONAL_INT32_COLUMNS = set()
    duv_module.REQUIRED_COLUMNS = set()
    duv_module.duv_sequence_predict = lambda *a, **k: None  # type: ignore[assignment]
    duv_module.duv_verbose = lambda verbose: verbose  # type: ignore[assignment]
    sys.modules["utilities.duv_infer"] = duv_module
    _set_spec("utilities.duv_infer")

    velocity_module = types.ModuleType("utilities.ml_velocity")

    class _DummyVelocityModel:
        requires_duv_feats = False

        @staticmethod
        def load(path):
            return _DummyVelocityModel()

        def to(self, device):
            return self

        def eval(self):
            return self

    velocity_module.MLVelocityModel = _DummyVelocityModel
    sys.modules["utilities.ml_velocity"] = velocity_module
    _set_spec("utilities.ml_velocity")

    eval_module = types.ModuleType("scripts.eval_duv")
    eval_module._ensure_int = lambda value, default=0: default  # type: ignore[attr-defined]
    eval_module._duration_predict = lambda *a, **k: (None, None)  # type: ignore[attr-defined]
    eval_module._get_device = lambda device: device  # type: ignore[attr-defined]
    eval_module._load_duration_model = lambda ckpt: None  # type: ignore[attr-defined]
    eval_module._load_stats = lambda *a, **k: ([], None, None, None)  # type: ignore[attr-defined]
    eval_module._parse_quant = lambda *a, **k: 0  # type: ignore[attr-defined]
    eval_module.load_stats_and_normalize = lambda *a, **k: (None, None)  # type: ignore[attr-defined]
    sys.modules["scripts.eval_duv"] = eval_module
    try:
        from importlib.machinery import ModuleSpec

        eval_module.__spec__ = ModuleSpec("scripts.eval_duv", loader=None)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback for limited importlib
        eval_module.__spec__ = None  # type: ignore[attr-defined]


def test_filter_program_missing_column(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare_stubs(monkeypatch)

    from scripts import predict_duv

    monkeypatch.setattr(predict_duv, "_load_stats", lambda *a, **k: ([], None, None, None))

    args = argparse.Namespace(
        stats_json=tmp_path / "stats.json",
        ckpt=tmp_path / "model.ckpt",
        csv=tmp_path / "notes.csv",
        out=tmp_path / "out.mid",
        batch=1,
        device="cpu",
        vel_smooth=1,
        smooth_pred_only=True,
        dur_quant=None,
        filter_program="program == 0",
        limit=0,
        verbose=False,
    )

    with pytest.raises(SystemExit) as excinfo:
        predict_duv.run(args)

    message = str(excinfo.value)
    assert "--filter-program" in message
    assert "program" in message
