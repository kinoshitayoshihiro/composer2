"""Project-local stub loaded automatically by Python at start-up.
Silences heavy optional dependencies in minimal CI environments."""

import importlib.util
import os
from pathlib import Path

try:
    import pytest_asyncio.plugin as _pa_plugin
    _pa_plugin.pytest_collectstart = lambda *a, **k: None  # type: ignore
except Exception:
    pass

# Patch pytest-asyncio to avoid AttributeError on newer pytest versions
try:
    import pytest_asyncio.plugin as _pa_plugin  # type: ignore

    def _safe_collectstart(collector):
        pyobject = getattr(collector, "obj", None)
        if pyobject is not None:
            setattr(
                pyobject,
                "__pytest_asyncio_scoped_event_loop",
                _pa_plugin.scoped_event_loop,
            )

    _pa_plugin.pytest_collectstart = _safe_collectstart  # type: ignore[attr-defined]
except Exception:
    pass

try:
    import pretty_midi  # type: ignore
except Exception:
    pretty_midi = None

if os.getenv("COMPOSER_CI_STUBS") == "1":
    path = Path(__file__).resolve().parent / "utilities" / "stub_utils.py"
    spec = importlib.util.spec_from_file_location("_stub_utils", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    module.install_stubs(module.STUB_MODULES, force_names=["music21", "scipy"])

if pretty_midi is not None and not hasattr(pretty_midi.PrettyMIDI, "set_tempo_changes"):
    def _set_tempo_changes(self, tempi, times):
        if len(tempi) != len(times):
            raise ValueError("tempi and times must have the same length")
        tick_scales = []
        tick = 0.0
        last_time = times[0] if times else 0.0
        last_scale = 60.0 / (float(tempi[0]) * self.resolution) if tempi else 60.0 / (120.0 * self.resolution)
        for i, (bpm, start) in enumerate(zip(tempi, times)):
            if i > 0:
                tick += (start - last_time) / last_scale
                last_scale = 60.0 / (float(bpm) * self.resolution) if bpm else 60.0 / (120.0 * self.resolution)
                last_time = start
            scale = last_scale
            tick_scales.append((int(round(tick)), scale))
        self._tick_scales = tick_scales
        self._update_tick_to_time(int(round(tick)) + 1)

    def _get_tempo_changes(self):
        times = []
        tempi = []
        last_tick = 0
        for tick, scale in self._tick_scales:
            if tick == 0:
                times.append(0.0)
            else:
                times.append(self.tick_to_time(tick))
            tempi.append(60.0 / (scale * self.resolution))
            last_tick = tick
        return tempi, times

    pretty_midi.PrettyMIDI.set_tempo_changes = _set_tempo_changes  # type: ignore[attr-defined]
    pretty_midi.PrettyMIDI.get_tempo_changes = _get_tempo_changes  # type: ignore[attr-defined]

try:
    import utilities  # noqa: F401
except Exception:
    pass
