import json
import tempfile
from pathlib import Path

from music21 import note, stream

from generator.drum_generator import (
    DrumGenerator,
    RESOLUTION,
    HAT_SUPPRESSION_THRESHOLD,
)


class SimpleDrumGenerator(DrumGenerator):
    """Subclass with simplified rendering to expose heatmap suppression."""

    def _render_part(self, section_data, next_section_data=None):
        part = stream.Part(id=self.part_name)
        start = section_data.get("absolute_offset", 0.0)
        measures = section_data.get("length_in_measures", 1)
        end = start + measures * 4.0
        t = start
        while t < end:
            grid_idx = int((t * RESOLUTION) % RESOLUTION)
            weight = self.heatmap.get(grid_idx, 0)
            rel = weight / max(self.heatmap.values()) if self.heatmap else 0
            if rel < HAT_SUPPRESSION_THRESHOLD:
                n = note.Note()
                n.pitch.midi = 42
                n.duration.quarterLength = 0.25
                part.insert(t, n)
            t += 0.25
        return part


def test_hat_suppressed_by_heatmap(tmp_path: Path, rhythm_library):
    heatmap = [{"grid_index": i, "count": (10 if i == 4 else 1)} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "rng_seed": 0,
        "paths": {"drum_pattern_files": ["data/drum_patterns.yml"]},
    }
    drum = SimpleDrumGenerator(
        main_cfg=cfg,
        part_name="drums",
        part_parameters=rhythm_library.drum_patterns or {},
    )

    section = {"absolute_offset": 0.0, "length_in_measures": 1}
    part = drum.compose(section_data=section)

    hats = [n for n in part.flatten().notes if n.pitch.midi == 42]
    suppressed = [n for n in hats if int((n.offset * RESOLUTION) % RESOLUTION) == 4]
    assert len(suppressed) == 0
    assert len(hats) == 12
