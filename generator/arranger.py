"""Simple arranger that builds piano parts per section based on YAML configs.

This module reads ``chordmap.yaml``, ``rhythm_library.yaml`` and
``emotion_profile.yaml`` from the :mod:`data` directory.  For each musical
section defined in the chord map it chooses a chord progression and rhythm
pattern according to the emotion associated with that section.  The resulting
music21 stream is exported as a MIDI file whose name follows a VOCALOID /
SynthV friendly convention such as ``01_Verse_piano.mid``.

The implementation is intentionally lightweight: it aims to provide a clear
example of how higher level song structure could be realised from declarative
YAML data.  It is not a full arranger but rather a starting point that
illustrates how chord and rhythm libraries might be combined.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import logging
import re

import yaml
from music21 import chord as m21chord
from music21 import harmony, meter, stream, tempo
from utilities.rhythm_library_loader import load_rhythm_library

logger = logging.getLogger(__name__)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Arranger:
    """Load YAML configuration and render simple MIDI accompaniments."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self.chordmap = self._load_yaml(self.data_dir / "chordmap.yaml")
        # rhythm library may use either .yaml or .yml extension
        rhythm_path = self.data_dir / "rhythm_library.yaml"
        if not rhythm_path.exists():
            rhythm_path = self.data_dir / "rhythm_library.yml"
        self.rhythm_library = load_rhythm_library(str(rhythm_path)).model_dump()
        self.emotion_profile = self._load_yaml(self.data_dir / "emotion_profile.yaml")
        self._chord_cache: dict[str, list] = {}

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def arrange(self, output_dir: Path | None = None) -> list[Path]:
        """Render one MIDI file per section.

        Parameters
        ----------
        output_dir:
            Directory where MIDI files will be written.  Defaults to
            ``outputs/arrangements`` under the project root.

        Returns
        -------
        list[Path]
            List of written MIDI file paths.
        """

        out_dir = output_dir or (Path(__file__).resolve().parent.parent / "outputs" / "arrangements")
        out_dir.mkdir(parents=True, exist_ok=True)

        global_settings = self.chordmap.get("global_settings", {})
        tempo_val = global_settings.get("tempo", 120)
        time_sig = global_settings.get("time_signature", "4/4")
        sec_map = self.chordmap.get("sections", self.chordmap.get("global_settings", {}).get("sections", {}))

        # determine ordering
        used_orders = {int(v["order"]) for v in sec_map.values() if "order" in v}
        next_order = 1

        def _next_order() -> int:
            nonlocal next_order
            while next_order in used_orders:
                next_order += 1
            used_orders.add(next_order)
            return next_order

        sections: list[tuple[int, str, dict[str, Any]]] = []
        for name, meta in sec_map.items():
            order = int(meta["order"]) if "order" in meta else _next_order()
            sections.append((order, name, meta))

        sections.sort(key=lambda x: x[0])

        written: list[Path] = []
        for _, section_name, section_data in sections:
            emotion = self._section_emotion(section_name, section_data)
            progression = self._select_progression(section_name, emotion, section_data)
            rhythm = self._select_rhythm(emotion)
            score = self._build_stream(progression, rhythm, tempo_val, time_sig)
            idx = len(written) + 1
            safe_section_name = re.sub(r"\W+", "_", section_name).strip("_")
            fname = f"{idx:02d}_{safe_section_name}_piano.mid"
            path = out_dir / fname
            score.write("midi", fp=str(path))
            written.append(path)
        return written

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _section_emotion(self, name: str, meta: dict[str, Any]) -> str:
        """Return the emotion associated with *name*.

        The function first looks for ``emotion`` inside the section metadata.
        Failing that, a ``sections`` mapping inside ``emotion_profile`` is
        consulted.  If still not found the first emotion defined in the
        profile is used as a fallback.
        """

        if "emotion" in meta:
            return str(meta["emotion"])
        section_map = self.emotion_profile.get("sections", {})
        if name in section_map:
            return str(section_map[name])
        fallback = next(iter(self.emotion_profile.keys()))
        logger.warning("Emotion for section '%s' not found; using '%s'", name, fallback)
        return str(fallback)

    def _select_progression(
        self, section_name: str, emotion: str, meta: dict[str, Any]
    ) -> list[str]:
        """Determine chord progression for a section.

        Progressions can be defined explicitly in ``chordmap.yaml`` under the
        section itself or globally under ``chord_progressions`` keyed by
        emotion.  When nothing is defined a common ``I–V–vi–IV`` progression in
        C major is returned as a sensible default.
        """

        if "progression" in meta:
            return list(meta["progression"])
        global_map = self.chordmap.get("chord_progressions", {})
        if emotion in global_map:
            return list(global_map[emotion])
        logger.warning(
            "Progression for section '%s' emotion '%s' missing; using default I–V–vi–IV",
            section_name,
            emotion,
        )
        return ["C", "G", "Am", "F"]

    def _select_rhythm(self, emotion: str) -> dict[str, Any]:
        """Select rhythm pattern description for an emotion.

        ``emotion_profile.yaml`` may specify ``rhythm_key`` for each emotion.
        The returned value is the corresponding entry from
        ``rhythm_library.yaml``; if no key is provided the
        ``piano_fallback_block`` pattern is used.
        """

        profile = self.emotion_profile.get(emotion)
        if not profile:
            logger.warning("Emotion '%s' not defined in emotion_profile; using defaults", emotion)
            rhythm_key = "piano_fallback_block"
        else:
            rhythm_key = profile.get("rhythm_key", "piano_fallback_block")

        patterns = self.rhythm_library.get("piano_patterns", {})
        if rhythm_key not in patterns:
            logger.warning("Rhythm key '%s' missing; using fallback", rhythm_key)
            rhythm_key = "piano_fallback_block"
        return patterns.get(rhythm_key, {})

    def _build_stream(
        self, chords: list[str], rhythm: dict[str, Any], tempo_val: int, time_sig: str
    ) -> stream.Stream:
        """Combine chords with a rhythm pattern into a :class:`music21.stream.Stream`."""

        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=float(tempo_val)))
        s.append(meter.TimeSignature(time_sig))

        pattern = rhythm.get("pattern") or []
        length_beats = rhythm.get("length_beats", 4)
        for chord_symbol in chords:
            pitches = self._chord_pitches(chord_symbol)
            m = stream.Measure()
            if not pattern:
                ch = m21chord.Chord(pitches, quarterLength=length_beats)
                m.insert(0, ch)
            else:
                for event in pattern:
                    offset = float(event.get("offset", 0))
                    dur = float(event.get("duration", length_beats))
                    ch = m21chord.Chord(pitches, quarterLength=dur)
                    m.insert(offset, ch)
            s.append(m)
        return s

    def _chord_pitches(self, symbol: str):
        if symbol not in self._chord_cache:
            cs = harmony.ChordSymbol(symbol)
            self._chord_cache[symbol] = list(cs.pitches)
        return self._chord_cache[symbol]


if __name__ == "__main__":  # pragma: no cover - manual invocation
    arranger = Arranger()
    arranger.arrange()
