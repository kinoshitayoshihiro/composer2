import pytest

from ml_models import NoteFeatureEmbedder

torch = pytest.importorskip("torch")


def test_feature_embedder_dim() -> None:
    emb = NoteFeatureEmbedder()
    pitch = torch.zeros(2, 3, dtype=torch.long)
    bucket = torch.zeros(2, 3, dtype=torch.long)
    pedal = torch.zeros(2, 3, dtype=torch.long)
    vel = torch.zeros(2, 3)
    qlen = torch.zeros(2, 3)
    out = emb(pitch, bucket, pedal, vel, qlen)
    assert out.shape[-1] == 16 + 4 + 2 + 2
