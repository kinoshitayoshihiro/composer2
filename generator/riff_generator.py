# ──────────────────────────────────────────────────────────────────────────────
# File: generator/riff_generator.py
# Desc: Main guitar/bass/keys riff generator using riff_library.yaml patterns
# Deps: pretty_midi, PyYAML, music21 (optional for chord parsing)
# Note: Designed to plug into your BasePartGenerator interface.
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
    from music21 import chord as m21chord, pitch as m21pitch
except Exception:  # pragma: no cover
    m21chord = None  # type: ignore
    m21pitch = None  # type: ignore

# If your project exposes BasePartGenerator, import it. Fallback shim for linting.
try:
    from .base import BasePartGenerator
except Exception:

    class BasePartGenerator:  # type: ignore
        pass


@dataclass
class RiffPattern:
    name: str
    description: str
    rhythm: List[float]  # beat positions within a bar (0..4 in 4/4)
    harmony: List[str]  # e.g., ["power5", "octave"]
    density: str  # low/mid/high


class RiffGenerator(BasePartGenerator):
    """
    Generate a backbone riff track as a dedicated MIDI layer.

    Parameters
    ----------
    instrument : str
        One of {"guitar", "bass", "keys"}. Used for register/program defaults.
    program : int
        General MIDI program number for PrettyMIDI Instrument.
    patterns_yaml : str | Path
        Path to riff_library.yaml. See provided template structure.

    Inputs to `generate()`
    ----------------------
    key : str               e.g., "A minor" (optional but improves chord parsing)
    tempo : float           BPM
    emotion : str           e.g., "sad", "warm", "intense", "neutral"
    section : str           e.g., "Verse", "Chorus", "Bridge"
    chord_seq : list[tuple[float, str]]
        Sequence of (bar_start_beat, chord_symbol like "Am"/"C/G"). Beats are cumulative.
    bars : int
        Number of bars to render (uses chord_seq cycling if needed).

    Returns
    -------
    pretty_midi.PrettyMIDI
        PrettyMIDI object containing a single Instrument with the riff.

    Notes
    -----
    - This class produces deterministic, pattern-based riffs first. You can swap the
      picker with a learned model later while keeping the render pipeline.
    - Velocity scaling is tied to density and section to support "後半ほど激しく" 設計。
    """

    DEFAULT_PROGRAMS = {
        "guitar": 29,  # Overdriven Guitar
        "bass": 34,  # Electric Bass (finger)
        "keys": 5,  # Electric Piano 1
    }

    REGISTERS = {
        "guitar": (40, 76),  # E2..E5-ish
        "bass": (28, 55),  # E1..G3-ish
        "keys": (48, 84),  # C3..C6-ish
    }

    def __init__(
        self,
        instrument: str = "guitar",
        program: Optional[int] = None,
        patterns_yaml: str | Path = "data/riff_library.yaml",
    ) -> None:
        if pretty_midi is None:
            raise ImportError("pretty_midi is required for RiffGenerator")
        self.instrument = instrument if instrument in self.DEFAULT_PROGRAMS else "guitar"
        self.program = int(
            program if program is not None else self.DEFAULT_PROGRAMS[self.instrument]
        )
        self.register = self.REGISTERS[self.instrument]
        self.patterns_yaml = Path(patterns_yaml)
        self._patterns = self._load_patterns(self.patterns_yaml)

    # ------------------------- Public API -------------------------
    def generate(
        self,
        *,
        key: str | None,
        tempo: float,
        emotion: str,
        section: str,
        chord_seq: List[Tuple[float, str]],
        bars: int = 8,
        seed: Optional[int] = None,
        style: Optional[str] = None,
    ) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo or 120.0))
        inst = pretty_midi.Instrument(program=self.program, name=f"Riff:{self.instrument}")

        sec_per_beat = 60.0 / float(tempo or 120.0)
        beats_per_bar = 4.0

        # Pick a pattern according to style/section/emotion
        pattern = self._pick_pattern(style or self._guess_style(section, emotion), section, emotion)

        # Render per bar
        for bar_idx in range(bars):
            bar_start_beat = chord_seq[bar_idx % len(chord_seq)][0]
            chord_sym = chord_seq[bar_idx % len(chord_seq)][1]
            # For dynamics: ramp up toward the end
            vel_scale = self._velocity_scale(section, bar_idx, bars, pattern.density)

            for hit in pattern.rhythm:
                onset_beat = bar_start_beat + float(hit)
                start = onset_beat * sec_per_beat
                dur_beats = self._default_duration(pattern)
                end = (onset_beat + dur_beats) * sec_per_beat
                # Convert harmony token list -> one or multiple notes
                for pitch in self._harmony_to_pitches(chord_sym, pattern.harmony):
                    midi = self._clamp_register(pitch)
                    vel = self._velocity_from_density(pattern.density, vel_scale)
                    inst.notes.append(
                        pretty_midi.Note(velocity=vel, pitch=midi, start=start, end=end)
                    )

        pm.instruments.append(inst)
        return pm

    # ------------------------- Internals -------------------------
    def _load_patterns(self, path: Path) -> Dict[str, Dict[str, Dict[str, List[RiffPattern]]]]:
        if yaml is None:
            raise ImportError("PyYAML is required to load riff patterns")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        data = {}
        for style, sec_map in (raw.get("riff_patterns") or {}).items():
            data[style] = {}
            for sec, emo_map in sec_map.items():
                data[style][sec.lower()] = {}
                for emo, plist in emo_map.items():
                    pats: List[RiffPattern] = []
                    for p in plist or []:
                        pats.append(
                            RiffPattern(
                                name=p.get("name", f"{style}_{sec}_{emo}"),
                                description=p.get("description", ""),
                                rhythm=[float(x) for x in (p.get("rhythm") or [0.0, 2.0])],
                                harmony=[str(x) for x in (p.get("harmony") or ["power5"])],
                                density=str(p.get("density", "mid")),
                            )
                        )
                    data[style][sec.lower()][emo] = pats
        return data

    def _pick_pattern(self, style: str, section: str, emotion: str) -> RiffPattern:
        style = style.lower()
        section = section.lower()
        emotion = emotion.lower()
        # style/section/emotion cascade fallback
        pats = (
            self._patterns.get(style, {}).get(section, {}).get(emotion)
            or self._patterns.get(style, {}).get(section, {}).get("neutral")
            or self._first_any(self._patterns.get(style, {}).get(section, {}))
            or self._first_section_any(style)
        )
        if not pats:
            # extremely defensive default
            return RiffPattern(
                name="default_power8",
                description="",
                rhythm=[0.0, 1.0, 2.0, 3.0],
                harmony=["power5"],
                density="mid",
            )
        # Simple round-robin or random could be added; choose first for determinism
        return pats[0]

    def _first_any(self, emo_map: Dict[str, List[RiffPattern]] | None) -> List[RiffPattern] | None:
        if not emo_map:
            return None
        for _emo, v in emo_map.items():
            if v:
                return v
        return None

    def _first_section_any(self, style: str) -> List[RiffPattern] | None:
        sec_map = self._patterns.get(style) or {}
        for _sec, emo_map in sec_map.items():
            got = self._first_any(emo_map)
            if got:
                return got
        return None

    def _default_duration(self, pattern: RiffPattern) -> float:
        # Basic heuristic: denser patterns get shorter note values
        return 0.5 if pattern.density in ("mid", "mid_high") else 1.0

    def _velocity_from_density(self, density: str, scale: float) -> int:
        base = {"low": 64, "mid": 84, "mid_high": 96, "high": 104}.get(density, 84)
        return int(max(30, min(127, round(base * scale))))

    def _velocity_scale(self, section: str, bar_idx: int, total_bars: int, density: str) -> float:
        # "後半に行くほど激しく"の素直な実装。Bridge/Chorusは少し強め。
        bias = 1.0
        if section.lower() in ("chorus", "bridge"):
            bias = 1.05
        t = (bar_idx + 1) / max(1, total_bars)
        ramp = 0.9 + 0.3 * t  # 0.9 → 1.2
        # Low density shouldn't jump too loud
        if density == "low":
            ramp = 0.85 + 0.2 * t
        return bias * ramp

    # --- Harmony mapping ---
    def _harmony_to_pitches(self, chord_symbol: str, tags: List[str]) -> List[int]:
        root_midi = self._chord_root_midi(chord_symbol)
        pitches: List[int] = []
        for tag in tags:
            if tag == "power5":
                pitches.extend([root_midi, root_midi + 7])
            elif tag == "octave":
                pitches.extend([root_midi, root_midi + 12])
            elif tag == "triad":
                third = 3 if self._is_minor(chord_symbol) else 4
                pitches.extend([root_midi, root_midi + third, root_midi + 7])
            elif tag == "sus2":
                pitches.extend([root_midi, root_midi + 2, root_midi + 7])
            elif tag == "add9":
                third = 3 if self._is_minor(chord_symbol) else 4
                pitches.extend([root_midi, root_midi + third, root_midi + 7, root_midi + 14])
            elif tag == "root":
                pitches.append(root_midi)
            elif tag == "fifth":
                pitches.append(root_midi + 7)
            else:
                pitches.append(root_midi)
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for p in pitches:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq

    def _chord_root_midi(self, chord_symbol: str) -> int:
        # Prefer music21 if available for robust parsing, else naive map (C=48 base)
        if m21pitch is not None:
            try:
                # chord_symbol like "Am" or "C/G" → take root
                root = m21chord.ChordSymbol(chord_symbol).root()
                return int(root.midi)  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback: parse first letter + accidental
        name = chord_symbol.strip()
        pcs = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        semi = 0
        if name:
            base = pcs.get(name[0].upper(), 0)
            rest = name[1:]
            if rest.startswith("#"):
                semi = 1
            elif rest.startswith("b"):
                semi = -1
            pc = (base + semi) % 12
        else:
            pc = 0
        # Put root around typical instrument low register
        base_oct = 3 if self.instrument != "bass" else 2
        return 12 * base_oct + pc

    def _is_minor(self, chord_symbol: str) -> bool:
        cs = chord_symbol.lower().replace(" ", "")
        return ("m" in cs and not cs.startswith("maj")) or "min" in cs

    def _clamp_register(self, midi: int) -> int:
        lo, hi = self.register
        while midi < lo:
            midi += 12
        while midi > hi:
            midi -= 12
        return midi


# ──────────────────────────────────────────────────────────────────────────────
# File: generator/obligato_generator.py
# Desc: Short decorative lines (obbligato) layered over vocals/lead
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

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


# ──────────────────────────────────────────────────────────────────────────────
# File: utilities/pattern_loader.py
# Desc: Shared helpers (optional) — if you later want centralized caching.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:
    yaml = None  # type: ignore


@lru_cache(maxsize=16)
def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML required to load YAML files")
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ──────────────────────────────────────────────────────────────────────────────
# Usage snippet (put in a notebook or a quick script)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal smoke test rendering 4 bars with dummy chord sequence
    chords = [(0.0, "Am"), (4.0, "G"), (8.0, "F"), (12.0, "E")]  # bar_start, symbol

    rg = RiffGenerator(instrument="guitar", patterns_yaml="data/riff_library.yaml")
    pm_riff = rg.generate(
        key="A minor", tempo=92.0, emotion="warm", section="Verse", chord_seq=chords, bars=8
    )
    pm_riff.write("/tmp/demo_riff.mid")

    og = ObligatoGenerator(instrument="synth", patterns_yaml="data/obligato_library.yaml")
    pm_obl = og.generate(
        key="A minor", tempo=92.0, emotion="warm", section="Verse", chord_seq=chords, bars=8
    )
    pm_obl.write("/tmp/demo_obligato.mid")

    print("Wrote /tmp/demo_riff.mid and /tmp/demo_obligato.mid")
