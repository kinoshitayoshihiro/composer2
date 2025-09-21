# ──────────────────────────────────────────────────────────────────────────────
# File: generator/obligato_generator.py
# Desc: Short decorative lines (obbligato) layered over vocals/lead
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import math

try:
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None  # type: ignore

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    from music21 import key as m21key, scale as m21scale, pitch as m21pitch
except Exception:  # pragma: no cover
    m21key = None  # type: ignore
    m21scale = None  # type: ignore
    m21pitch = None  # type: ignore

try:
    from .base import BasePartGenerator
except Exception:

    class BasePartGenerator:  # type: ignore
        pass


@dataclass
class OblPattern:
    name: str
    description: str
    rhythm: List[float]
    contour: List[str]  # e.g., ["up_minor3", "down_second", "gliss_up5"]
    register: str  # low/mid/high
    density: str


class ObligatoGenerator(BasePartGenerator):
    """
    Generate short decorative obligato lines to enrich the texture.

    Parameters
    ----------
    instrument : str
        One of {"synth", "guitar", "woodwind", "strings"}.
    program : int
        GM program for PrettyMIDI.
    patterns_yaml : Path to obligato_library.yaml

    Inputs to `generate()`
    ----------------------
    key : str           e.g., "A minor" (important for scale)
    tempo : float       BPM
    emotion : str       e.g., "sad", "warm", "reflective"
    section : str       e.g., "Verse", "Chorus", "Bridge"
    chord_seq : list[(bar_start_beat, chord_symbol)]
    bars : int

    Returns
    -------
    pretty_midi.PrettyMIDI
    """

    DEFAULT_PROGRAMS = {
        "synth": 81,  # Lead 1 (square) as a bright obligato default
        "guitar": 27,  # Electric Guitar (jazz)
        "woodwind": 73,  # Flute
        "strings": 49,  # String Ensemble 1
    }

    REG_RANGES = {
        "low": (48, 60),
        "mid": (60, 72),
        "high": (72, 88),
    }

    CONTOUR_STEPS = {
        # semitone steps
        "up_second": +2,
        "down_second": -2,
        "up_minor3": +3,
        "down_minor3": -3,
        "up_major3": +4,
        "down_major3": -4,
        "up_fourth": +5,
        "down_fourth": -5,
        "up_fifth": +7,
        "down_fifth": -7,
        # Glissandos are treated as target offsets (we render a slide-like fast run)
        "gliss_up5": +7,
        "gliss_down5": -7,
    }

    def __init__(
        self,
        instrument: str = "synth",
        program: Optional[int] = None,
        patterns_yaml: str | Path = "data/obligato_library.yaml",
    ) -> None:
        if pretty_midi is None:
            raise ImportError("pretty_midi is required for ObligatoGenerator")
        self.instrument = instrument if instrument in self.DEFAULT_PROGRAMS else "synth"
        self.program = int(
            program if program is not None else self.DEFAULT_PROGRAMS[self.instrument]
        )
        self.patterns_yaml = Path(patterns_yaml)
        self._patterns = self._load_patterns(self.patterns_yaml)

    def generate(
        self,
        *,
        key: str,
        tempo: float,
        emotion: str,
        section: str,
        chord_seq: List[Tuple[float, str]],
        bars: int = 8,
        seed: Optional[int] = None,
    ) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo or 120.0))
        inst = pretty_midi.Instrument(program=self.program, name=f"Obl:{self.instrument}")
        sec_per_beat = 60.0 / float(tempo or 120.0)

        # Scale context from key (fallback to natural minor/major)
        scale_pcs = self._scale_from_key(key)

        pattern = self._pick_pattern(section, emotion)
        lo, hi = self.REG_RANGES.get(pattern.register, (60, 72))

        for bar_idx in range(bars):
            bar_start_beat = chord_seq[bar_idx % len(chord_seq)][0]
            # pick a starting degree around the upper register center
            current = self._nearest_scale_note((lo + hi) // 2, scale_pcs)
            vel = self._velocity(pattern.density, bar_idx, bars, section)

            for tok in pattern.contour:
                when = pattern.rhythm[min(len(pattern.rhythm) - 1, pattern.contour.index(tok))]
                onset_beat = bar_start_beat + float(when)
                start = onset_beat * sec_per_beat
                dur = 0.5 * sec_per_beat  # short decorative hit by default

                if tok.startswith("gliss_"):
                    # render two fast notes to emulate gliss target
                    target = self._clamp(current + self.CONTOUR_STEPS[tok], lo, hi)
                    mid = current + math.copysign(2, target - current)
                    for p in [int(current), int(self._clamp(mid, lo, hi)), int(target)]:
                        inst.notes.append(
                            pretty_midi.Note(
                                velocity=max(50, vel - 8),
                                pitch=p,
                                start=start,
                                end=start + dur * 0.4,
                            )
                        )
                        start += dur * 0.35
                    current = target
                else:
                    step = self.CONTOUR_STEPS.get(tok, 0)
                    target = self._nearest_scale_note(current + step, scale_pcs)
                    target = self._clamp(target, lo, hi)
                    inst.notes.append(
                        pretty_midi.Note(
                            velocity=vel, pitch=int(target), start=start, end=start + dur
                        )
                    )
                    current = target

        pm.instruments.append(inst)
        return pm

    # ------------------------- Internals -------------------------
    def _load_patterns(self, path: Path) -> Dict[str, Dict[str, List[OblPattern]]]:
        if yaml is None:
            raise ImportError("PyYAML is required to load obligato patterns")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        data: Dict[str, Dict[str, List[OblPattern]]] = {}
        for inst, sec_map in (raw.get("obligato_patterns") or {}).items():
            for sec, emo_map in sec_map.items():
                key = f"{sec}".lower()
                data.setdefault(key, {})
                for emo, plist in emo_map.items():
                    pats: List[OblPattern] = []
                    for p in plist or []:
                        pats.append(
                            OblPattern(
                                name=p.get("name", f"{inst}_{sec}_{emo}"),
                                description=p.get("description", ""),
                                rhythm=[float(x) for x in (p.get("rhythm") or [2.0, 3.5])],
                                contour=[str(x) for x in (p.get("contour") or ["up_second"])],
                                register=str(p.get("register", "mid")),
                                density=str(p.get("density", "low")),
                            )
                        )
                    data[key].setdefault(emo, []).extend(pats)
        return data

    def _pick_pattern(self, section: str, emotion: str) -> OblPattern:
        sec = section.lower()
        emo = emotion.lower()
        pats = self._patterns.get(sec, {}).get(emo) or self._patterns.get(sec, {}).get("neutral")
        if not pats:
            return OblPattern(
                name="synth_bell_fill",
                description="default",
                rhythm=[1.5, 3.0],
                contour=["up_minor3", "down_second"],
                register="high",
                density="low",
            )
        return pats[0]

    def _scale_from_key(self, key_name: str) -> List[int]:
        # Returns allowed pitch classes for simple diatonic filtering
        if m21key is not None and m21scale is not None:
            try:
                k = m21key.Key(key_name)
                scl = (
                    m21scale.MajorScale(k.tonic)
                    if k.mode == "major"
                    else m21scale.MinorScale(k.tonic)
                )
                return [p.pitchClass for p in scl.getPitches(k.tonic, k.tonic.transpose("P8"))]
            except Exception:
                pass
        # Fallback: C major/A minor shape
        return [0, 2, 4, 5, 7, 9, 11]

    def _nearest_scale_note(self, midi: int, pcs: List[int]) -> int:
        if not pcs:
            return midi
        best = midi
        best_d = 128
        for off in range(-6, 7):
            cand = midi + off
            if (cand % 12) in pcs:
                d = abs(off)
                if d < best_d:
                    best, best_d = cand, d
        return best

    def _clamp(self, v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def _velocity(self, density: str, bar_idx: int, total_bars: int, section: str) -> int:
        base = {"low": 68, "mid": 82, "high": 96}.get(density, 76)
        t = (bar_idx + 1) / max(1, total_bars)
        ramp = 0.95 + 0.25 * t  # gentle lift
        if section.lower() == "chorus":
            ramp += 0.05
        return int(max(30, min(127, round(base * ramp))))
