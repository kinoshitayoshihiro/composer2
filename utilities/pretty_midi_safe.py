from __future__ import annotations

import pretty_midi as _pm


def new_pm(
    path: str | bytes | None = None,
    tempo: float | None = None,
    resolution: int = 480,
) -> _pm.PrettyMIDI:
    if path is None:
        pm = _pm.PrettyMIDI(initial_tempo=tempo or 120.0, resolution=resolution)
    else:
        pm = _pm.PrettyMIDI(path)
        if tempo is not None:
            if hasattr(pm, "write_tempo_changes"):
                pm.write_tempo_changes([0.0], [tempo])
            else:
                pm._tick_scales = [(0, 60.0 / (tempo * pm.resolution))]
    pm.resolution = resolution
    return pm
