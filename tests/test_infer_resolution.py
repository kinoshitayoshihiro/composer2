from utilities import groove_sampler


def test_infer_resolution_beats_per_bar_int() -> None:
    beats = [0.0, 1.0, 2.0]
    res = groove_sampler.infer_resolution(beats, beats_per_bar=8)
    assert res == 8


def test_infer_resolution_beats_per_bar_half() -> None:
    beats = [0.0, 0.5, 1.0]
    res = groove_sampler.infer_resolution(beats, beats_per_bar=8)
    assert res == 16

