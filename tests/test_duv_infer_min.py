import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from utilities.duv_infer import duv_sequence_predict


class _SpyDUV(torch.nn.Module):
    def __init__(
        self,
        *,
        max_len: int = 4,
        use_bar_beat: bool = True,
        section: bool = True,
        mood: bool = True,
        vel_bucket: bool = True,
        dur_bucket: bool = True,
    ) -> None:
        super().__init__()
        self.requires_duv_feats = True
        self.has_vel_head = True
        self.has_dur_head = True
        self.max_len = max_len
        self.core = self
        self.use_bar_beat = use_bar_beat
        self.section_emb = object() if section else None
        self.mood_emb = object() if mood else None
        self.vel_bucket_emb = object() if vel_bucket else None
        self.dur_bucket_emb = object() if dur_bucket else None
        self.captured: dict[str, torch.Tensor] | None = None
        self.captured_mask: torch.Tensor | None = None

    def forward(self, feats: dict[str, torch.Tensor], *, mask: torch.Tensor | None = None):
        assert mask is not None
        self.captured = {k: v.detach().cpu() for k, v in feats.items()}
        self.captured_mask = mask.detach().cpu()
        vel = torch.zeros(1, self.max_len, dtype=torch.float32, device=mask.device)
        dur = torch.zeros(1, self.max_len, dtype=torch.float32, device=mask.device)
        return {"velocity": vel, "duration": dur}


def test_duv_sequence_predict_zero_fill_required_only() -> None:
    df = pd.DataFrame(
        {
            "track_id": [0, 0],
            "bar": [0, 0],
            "position": [0, 5],
            "pitch": [60, 200],
            "velocity": [0.5, 0.6],
            "duration": [0.1, 0.2],
        }
    )
    model = _SpyDUV(max_len=4)
    preds = duv_sequence_predict(df, model, torch.device("cpu"), verbose=False)
    assert preds is not None
    assert preds["velocity"].shape == (len(df),)
    assert preds["velocity_mask"].tolist() == [True, True]
    assert preds["duration_mask"].tolist() == [True, True]
    feats = model.captured
    assert feats is not None
    assert torch.all(feats["bar_phase"] == 0)
    assert torch.all(feats["beat_phase"] == 0)
    assert torch.all(feats["section"] == 0)
    assert torch.all(feats["mood"] == 0)
    assert torch.all(feats["vel_bucket"] == 0)
    assert torch.all(feats["dur_bucket"] == 0)
    assert feats["pitch"][0, 1].item() == 127
    assert feats["pitch_class"][0, 1].item() == 7
    assert feats["position"][0, 1].item() == model.max_len - 1


def test_duv_sequence_predict_optional_passthrough() -> None:
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "bar": [2, 2, 2],
            "position": [0, 1, 2],
            "pitch": [10, 20, 30],
            "velocity": [0.1, 0.2, 0.3],
            "duration": [0.2, 0.3, 0.4],
            "bar_phase": [0.1, 0.2, 0.3],
            "beat_phase": [0.4, 0.5, 0.6],
            "section": [3, 4, 5],
            "vel_bucket": [2, 3, 4],
        }
    )
    model = _SpyDUV(max_len=4, mood=True, dur_bucket=True)
    preds = duv_sequence_predict(df, model, torch.device("cpu"), verbose=False)
    assert preds is not None
    feats = model.captured
    assert feats is not None
    np.testing.assert_allclose(feats["bar_phase"][0, :3].numpy(), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(feats["beat_phase"][0, :3].numpy(), [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(feats["section"][0, :3].numpy(), [3, 4, 5])
    np.testing.assert_array_equal(feats["vel_bucket"][0, :3].numpy(), [2, 3, 4])
    assert torch.all(feats["mood"] == 0)
    assert torch.all(feats["dur_bucket"] == 0)
    assert torch.all(model.captured_mask[0, :3])
    assert not model.captured_mask[0, 3]
