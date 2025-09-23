import random

from scripts import sample_phrase as sample_phrase


def test_resolve_steps_per_bar_priority():
    state = {"steps_per_bar": 3, "meta": {"steps_per_bar": 9}}
    hparams = {"steps_per_bar": 6}
    assert sample_phrase._resolve_steps_per_bar(state, hparams) == 6
    assert sample_phrase._resolve_steps_per_bar(state, None) == 3
    assert sample_phrase._resolve_steps_per_bar({"meta": {"steps_per_bar": 5}}, None) == 5
    assert sample_phrase._resolve_steps_per_bar({}, None) == 4


def test_compute_step_limits_and_fallback_events():
    max_steps, bar_limit = sample_phrase._compute_step_limits(
        length=64, model_max_len=8, bars=1, steps_per_bar=4
    )
    assert max_steps == 4
    assert bar_limit == 4
    random.seed(42)
    events = sample_phrase._generate_fallback_events(
        max_steps=max_steps, pitch_min=36, pitch_max=48
    )
    assert len(events) == max_steps
    assert all(36 <= ev["pitch"] <= 48 for ev in events)
