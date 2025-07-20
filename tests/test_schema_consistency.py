from pathlib import Path

import pytest

from utilities.ml_articulation import MLArticulationModel

torch = pytest.importorskip("torch")


def test_schema_consistency(tmp_path: Path) -> None:
    model = MLArticulationModel(7)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)
    schema = Path("articulation_schema.yaml")
    with pytest.raises(AssertionError):
        MLArticulationModel.load(ckpt, schema)
