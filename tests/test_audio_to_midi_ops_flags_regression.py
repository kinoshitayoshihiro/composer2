import math
from pathlib import Path

import numpy as np
import pretty_midi

from utilities.apply_controls import apply_controls, ControlCurve


def _dummy_wav(sr: int = 16000, secs: float = 1.0) -> tuple[np.ndarray, int]:
    """Return a constant-amplitude mono waveform and its sample rate."""

    return np.ones(int(sr * secs), dtype=float), sr


def test_beats_domain_tempo_map_regression(tmp_path: Path) -> None:
    audio, sr = _dummy_wav(16000, 1.0)
    # Ensure the clip endpoint is represented so tempo folding extends to 2 beats.
    times_sec = np.array([0.0, 0.5, 1.0])
    values = [0.0, 127.0, 127.0]
    bpm0 = 120.0
    times_beats = (times_sec * bpm0 / 60.0).tolist()
    curve = ControlCurve(times_beats, values, domain="beats", sample_rate_hz=2.0)

    pm = pretty_midi.PrettyMIDI()
    curves_by_ch = {0: {"cc11": curve}}
    tempo_map = [(0, 120.0), (1, 60.0)]

    apply_controls(
        pm,
        curves_by_ch,
        bend_range_semitones=2.0,
        sample_rate_hz={"cc": 2.0},
        tempo_map=tempo_map,
    )
    inst0 = [i for i in pm.instruments if i.name == "channel0"][0]
    last_t = max(cc.time for cc in inst0.control_changes if cc.number == 11)
    assert math.isclose(last_t, 1.5, rel_tol=0.1)


def test_rpn_order_is_stable(tmp_path: Path) -> None:
    pm = pretty_midi.PrettyMIDI()
    curves_by_ch: dict[int, dict[str, ControlCurve]] = {
        0: {"cc11": ControlCurve([0.0, 0.0], [0, 0], domain="time", sample_rate_hz=1.0)}
    }
    pb_inst = pretty_midi.Instrument(program=0, name="channel0")
    pb_inst.pitch_bends.append(pretty_midi.PitchBend(pitch=100, time=0.01))
    pm.instruments.append(pb_inst)

    apply_controls(pm, curves_by_ch, bend_range_semitones=2.0, write_rpn=True)
    inst0 = pm.instruments[0]
    ccs = inst0.control_changes[:4]
    assert ccs == sorted(ccs, key=lambda c: (c.time, c.number, c.value))
    nums = [c.number for c in ccs]
    assert nums == [101, 100, 6, 38]


def test_total_max_events_includes_rpn(tmp_path: Path) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="channel0")
    for i in range(100):
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=i * 10, time=i / 1000.0))
    pm.instruments.append(inst)

    curves_by_ch: dict[int, dict[str, ControlCurve]] = {0: {}}
    apply_controls(
        pm,
        curves_by_ch,
        bend_range_semitones=2.0,
        sample_rate_hz={"bend": 100.0},
        max_events={"bend": 100},
        total_max_events=40,
        write_rpn=True,
    )
    total = sum(len(i.control_changes) + len(i.pitch_bends) for i in pm.instruments)
    assert total <= 40
