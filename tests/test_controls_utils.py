import pytest

pretty_midi = pytest.importorskip("pretty_midi")

from utilities.controls_utils import (
    apply_post_bend_policy,
    synthesize_vibrato,
)
from utilities import pb_math


def test_vibrato_cycles():
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    bends = synthesize_vibrato(inst, depth_semitones=2.0, rate_hz=5.0)
    apply_post_bend_policy(inst, bends, policy="replace")
    vals = [b.pitch for b in inst.pitch_bends]
    crossings = sum(1 for a, b in zip(vals, vals[1:]) if (a < 0 <= b) or (a > 0 >= b))
    assert 8 <= crossings <= 12


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("skip", [1000]),
        ("add", [pb_math.PB_MAX]),
        ("replace", [8000]),
    ],
)
def test_post_bend_policy(policy, expected):
    inst = pretty_midi.Instrument(program=0)
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=1000, time=0.0))
    bends = [pretty_midi.PitchBend(pitch=8000, time=0.0)]
    apply_post_bend_policy(inst, bends, policy=policy)
    assert [b.pitch for b in inst.pitch_bends] == expected
