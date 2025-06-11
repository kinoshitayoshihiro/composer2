import os
import sys
import json
import tempfile
from pathlib import Path
from music21 import note, stream

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("generator")
pkg.__path__ = [str(ROOT / "generator")]
sys.modules.setdefault("generator", pkg)

spec = importlib.util.spec_from_file_location(
    "generator.drum_generator",
    ROOT / "generator" / "drum_generator.py",
    submodule_search_locations=[str(ROOT / "generator")],
)
_mod = importlib.util.module_from_spec(spec)
sys.modules["generator.drum_generator"] = _mod
spec.loader.exec_module(_mod)
DrumGenerator = _mod.DrumGenerator
RESOLUTION = _mod.RESOLUTION
HAT_SUPPRESSION_THRESHOLD = _mod.HAT_SUPPRESSION_THRESHOLD


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


def test_hat_suppressed_by_heatmap(tmp_path: Path):
    heatmap = [{"grid_index": i, "count": (10 if i == 4 else 1)} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    cfg = {"vocal_midi_path_for_drums": "", "heatmap_json_path_for_drums": str(heatmap_path), "rng_seed": 0}
    drum = SimpleDrumGenerator(main_cfg=cfg, part_name="drums")

    section = {"absolute_offset": 0.0, "length_in_measures": 1}
    part = drum.compose(section_data=section)

    hats = [n for n in part.flatten().notes if n.pitch.midi == 42]
    suppressed = [n for n in hats if int((n.offset * RESOLUTION) % RESOLUTION) == 4]
    assert len(suppressed) == 0
    assert len(hats) == 12
