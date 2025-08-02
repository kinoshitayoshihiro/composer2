from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("arranger", REPO_ROOT / "generator" / "arranger.py")
arranger_module = importlib.util.module_from_spec(spec)
sys.modules["arranger"] = arranger_module
spec.loader.exec_module(arranger_module)
Arranger = arranger_module.Arranger


def test_arranger_writes_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "chordmap.yaml").write_text(
        """
sections:
  Verse:
    emotion: calm
    progression: [C, F]
  Chorus:
    emotion: happy
    progression: [G, C]
""",
        encoding="utf-8",
    )

    (data_dir / "emotion_profile.yaml").write_text(
        """
calm:
  rhythm_key: piano_fallback_block
happy:
  rhythm_key: piano_fallback_block
""",
        encoding="utf-8",
    )

    (data_dir / "rhythm_library.yaml").write_text(
        """
piano_patterns:
  piano_fallback_block:
    length_beats: 4
    pattern:
      - beat: 0
        duration: 4
""",
        encoding="utf-8",
    )

    arranger = Arranger(data_dir)
    out_dir = tmp_path / "out"
    written = arranger.arrange(out_dir)

    assert len(written) == 2
    names = sorted(p.name for p in written)
    assert names[0].startswith("01_")
    assert names[1].startswith("02_")
