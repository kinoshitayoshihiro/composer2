import sitecustomize  # noqa: F401  # ensure tempo patch

import pytest

pretty_midi = pytest.importorskip("pretty_midi")
PrettyMIDI = pretty_midi.PrettyMIDI


def test_tempo_changes() -> None:
    pm = PrettyMIDI()
    tempi = [120, 140]
    times = [0.0, 30.0]
    # PrettyMIDI lacks ``set_tempo_changes`` in older versions. Emulate the
    # behaviour by writing to ``_tick_scales`` directly.
    pm._tick_scales = []
    tick = 0.0
    last_time = times[0]
    last_scale = 60.0 / (float(tempi[0]) * pm.resolution)
    for i, (bpm, start) in enumerate(zip(tempi, times)):
        if i > 0:
            tick += (start - last_time) / last_scale
            last_scale = 60.0 / (float(bpm) * pm.resolution)
            last_time = start
        pm._tick_scales.append((int(round(tick)), last_scale))
    pm._update_tick_to_time(int(round(tick)) + 1)

    times, bpms = pm.get_tempo_changes()
    times_list = times.tolist() if hasattr(times, "tolist") else list(times)
    assert len(bpms) == 2
    assert times_list == pytest.approx([0.0, 30.0])
    assert bpms[0] == pytest.approx(120)
    assert bpms[1] == pytest.approx(140)
