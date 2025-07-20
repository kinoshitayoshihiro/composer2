from pathlib import Path

import pytest

from utilities.ml_articulation import MLArticulationModel

torch = pytest.importorskip("torch")


def test_schema_roundtrip(tmp_path: Path) -> None:
    model = MLArticulationModel(9)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)
    loaded = MLArticulationModel.load(ckpt, Path("articulation_schema.yaml"))
    assert loaded.fc.weight.shape[0] == 9
