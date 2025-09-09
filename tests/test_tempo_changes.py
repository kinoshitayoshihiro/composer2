import sitecustomize  # noqa: F401  # ensure tempo patch
from pretty_midi import PrettyMIDI


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

    assert len(pm.get_tempo_changes()[0]) == 2
    assert pm.get_tempo_changes()[0][0] == 120
    assert pm.get_tempo_changes()[0][1] == 140
