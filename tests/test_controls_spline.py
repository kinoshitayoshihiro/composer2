import pytest

pd = pytest.importorskip("pandas")

from ml.controls_spline import fit_controls, infer_controls


def test_fit_infer(tmp_path):
    df = pd.DataFrame({"bend": [0, 1, 2], "cc11": [10, 20, 30]})
    notes = tmp_path / "notes.parquet"
    df.to_parquet(notes)
    model = tmp_path / "model.json"
    fit_controls(notes, targets=["bend", "cc11"], out_path=model)
    out = tmp_path / "pred.parquet"
    infer_controls(model, out_path=out)
    assert out.exists()
