import pytest

from utilities.ml_articulation import MLArticulationModel

torch = pytest.importorskip("torch")


def test_model_mask() -> None:
    model = MLArticulationModel(3)
    batch = 1
    length = 4
    pitch = torch.zeros(batch, length, dtype=torch.long)
    bucket = torch.zeros_like(pitch)
    pedal = torch.zeros_like(pitch)
    vel = torch.zeros(batch, length)
    qlen = torch.zeros(batch, length)
    labels = torch.zeros(batch, length, dtype=torch.long)
    pad_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    _ = model.forward(pitch, bucket, pedal, vel, qlen, labels=labels, pad_mask=pad_mask)
    decoded = model.decode(pitch, bucket, pedal, vel, qlen, pad_mask=pad_mask)
    assert len(decoded[0]) == 3
