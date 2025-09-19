import pytest

pytest.importorskip("pretty_midi")

import pretty_midi

from tools.ujam_bridge.utils import _tick_to_time


def test_tick_to_time_rounds_and_clamps():
    pm = pretty_midi.PrettyMIDI()
    assert _tick_to_time(pm, 150.4) == pytest.approx(pm.tick_to_time(150))
    assert _tick_to_time(pm, -3.2) == pytest.approx(pm.tick_to_time(0))
    assert _tick_to_time(pm, float("nan")) == pytest.approx(pm.tick_to_time(0))
