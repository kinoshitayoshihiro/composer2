import pytest

torch = pytest.importorskip("torch")
from ml_models.pedal_model import PedalModel


def test_forward_shape() -> None:
    model = PedalModel()
    x = torch.randn(2, 8, 13)
    out = model(x)
    assert out.shape == (2, 8)


def test_loss() -> None:
    model = PedalModel(class_weight=2.0)
    x = torch.randn(1, 4, 13)
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    pred = model(x)
    loss = model.loss(pred, target)
    assert loss > 0
