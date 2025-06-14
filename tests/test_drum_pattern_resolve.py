from pathlib import Path

from generator.drum_generator import DrumGenerator

class DummyDrum(DrumGenerator):
    def _render_part(self, section_data, next_section_data=None):
        pass


def test_relative_path_resolved(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(Path("data/heatmap.json").resolve()),
        "paths": {
            "rhythm_library_path": str(
                Path(__file__).resolve().parents[1] / "data" / "rhythm_library.yml"
            )
        },
    }
    drum = DummyDrum(
        main_cfg=cfg,
        part_name="drums",
        default_instrument=None,
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    lib = drum._load_pattern_lib(["data/drum_patterns.yml"])
    assert lib

