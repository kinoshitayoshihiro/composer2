from __future__ import annotations

import pretty_midi


class ModularComposer:
    """Simple composer stub for regression tests."""

    def __init__(self, *, global_settings: dict | None = None) -> None:
        self.settings = global_settings or {}
        self.tempo_bpm = self.settings.get("tempo_bpm", 120)

    def render(self, *, style: str, output_path: str) -> None:
        pattern = {
            "rock_drive_loop": [36, 38, 42, 38],
            "brush_light_loop": [38, 42, 40, 42],
            "funk_groove": [36, 38, 44, 38],
            "ballad_backbeat": [36, 42, 38, 42],
        }.get(style, [60])

        pm = pretty_midi.PrettyMIDI(initial_tempo=self.tempo_bpm)
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        for i, pitch in enumerate(pattern):
            start = i * 0.5
            end = start + 0.5
            inst.notes.append(
                pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
            )
        pm.instruments.append(inst)
        pm.write(str(output_path))
