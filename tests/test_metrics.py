import pytest

from eval import metrics


@pytest.mark.eval
def test_swing_score() -> None:
    shuffle = [
        {"offset": 0.0, "velocity": 100},
        {"offset": 2 / 3, "velocity": 100},
    ] * 2
    assert metrics.swing_score(shuffle) == pytest.approx(0.0, abs=0.01)
    swing60 = [
        {"offset": 0.0, "velocity": 100},
        {"offset": 0.6, "velocity": 100},
    ] * 2
    assert metrics.swing_score(swing60) > 0.5
