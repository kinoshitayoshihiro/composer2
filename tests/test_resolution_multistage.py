from utilities import groove_sampler

def test_multistage_resolution() -> None:
    beats = [0.0, 0.25, 0.5, 0.75, 1.0]
    res = groove_sampler.infer_multistage_resolution(beats)
    assert res['coarse'] <= res['fine']
    assert res['coarse'] > 0 and res['fine'] > 0
