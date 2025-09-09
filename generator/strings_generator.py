"""Simple block-chord generator for string ensembles."""

from __future__ import annotations

import copy
import enum

try:
    StrEnum = enum.StrEnum  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - fallback for Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        """Fallback definition for :class:`enum.StrEnum` for Python < 3.11."""

        pass


import math
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import music21.articulations as articulations
import music21.expressions as expressions
import numpy as np

if not hasattr(articulations, "Trill"):

    class Trill(articulations.Articulation):
        pass

    articulations.Trill = Trill

if not hasattr(articulations, "Tremolo"):

    class Tremolo(articulations.Articulation):
        def __init__(self, marks: int = 3) -> None:
            super().__init__()
            self.marks = marks

    articulations.Tremolo = Tremolo
from pathlib import Path

import music21.spanner as m21spanner
from music21 import chord, harmony
from music21 import instrument as m21instrument
from music21 import interval, note, pitch, stream, tie, volume

from utilities.cc_map import cc_map, load_cc_map
from utilities.cc_tools import finalize_cc_events, merge_cc_events
from utilities.core_music_utils import (
    get_key_signature_object,
    get_time_signature_object,
)
from utilities.effect_preset_loader import EffectPresetLoader
from utilities.expression_map import load_expression_map, resolve_expression
from utilities.harmonic_utils import apply_harmonic_notation, apply_harmonic_to_pitch
from utilities.midi_utils import safe_end_time
from utilities.velocity_curve import interpolate_7pt, resolve_velocity_curve
from utilities.velocity_utils import scale_velocity  # noqa: E402

from .base_part_generator import BasePartGenerator


@dataclass(frozen=True)
class _SectionInfo:
    name: str
    instrument: m21instrument.Instrument
    range_low: str
    range_high: str
    velocity_pos: float
    open_string_midi: list[int] | None = None


class BowPosition(StrEnum):
    """Bow position enumeration."""

    TASTO = "tasto"
    NORMALE = "normale"
    PONTICELLO = "ponticello"


EXEC_STYLE_TRILL = "trill"
EXEC_STYLE_TREMOLO = "tremolo"


def parse_articulation_field(field: Any) -> list[str]:
    """Parse an articulation specification string into a list of names."""
    if field is None:
        return []
    if isinstance(field, (list | tuple | set)):
        result: list[str] = []
        for item in field:
            result.extend(parse_articulation_field(item))
        return result
    text = str(field)
    tokens = [t for t in re.split(r"[+\s]+", text) if t]
    return tokens


def parse_bow_position(value: Any) -> BowPosition | None:
    """Convert *value* into a :class:`BowPosition` or ``None``."""
    if value is None:
        return None
    text = str(value).lower().strip()
    aliases = {
        "sul pont.": BowPosition.PONTICELLO,
        "sul pont": BowPosition.PONTICELLO,
        "pont.": BowPosition.PONTICELLO,
        "sul tasto": BowPosition.TASTO,
    }
    if text in aliases:
        return aliases[text]
    try:
        return BowPosition(text)
    except Exception:
        return None


def _clamp_velocity(value: float) -> int:
    """Clamp *value* into MIDI velocity range."""
    return max(1, min(127, int(round(value))))


class StringsGenerator(BasePartGenerator):
    """Generate very simple block-chord lines for a standard string section."""

    _SECTIONS = [
        _SectionInfo(
            "contrabass",
            m21instrument.Contrabass(),
            "C1",
            "C3",
            0.4,
            [43, 38, 33, 28],
        ),
        _SectionInfo(
            "violoncello",
            m21instrument.Violoncello(),
            "C2",
            "E4",
            0.4,
            [36, 43, 50, 57],
        ),
        _SectionInfo(
            "viola",
            m21instrument.Viola(),
            "C3",
            "A5",
            0.6,
            [48, 55, 62, 69],
        ),
        _SectionInfo(
            "violin_ii",
            m21instrument.Violin(),
            "G3",
            "D7",
            0.6,
            [55, 62, 69, 76],
        ),
        _SectionInfo(
            "violin_i",
            m21instrument.Violin(),
            "G3",
            "D7",
            0.8,
            [55, 62, 69, 76],
        ),
    ]

    def __init__(
        self,
        *args,
        global_settings: dict | None = None,
        default_instrument: m21instrument.Instrument | None = None,
        global_tempo: int | None = None,
        global_time_signature: str | None = None,
        global_key_signature_tonic: str | None = None,
        global_key_signature_mode: str | None = None,
        key: str | tuple[str, str] | None = None,
        tempo: float | None = None,
        emotion: str | None = None,
        voice_allocation: dict[str, int] | None = None,
        default_velocity_curve: list[int] | list[float] | str | None = None,
        expression_maps_path: str | None = None,
        voicing_mode: str = "close",
        divisi: bool | dict[str, str] | None = None,
        avoid_low_open_strings: bool = False,
        timing_jitter_ms: float = 0.0,
        timing_jitter_mode: str = "uniform",
        timing_jitter_scale_mode: str = "absolute",
        balance_scale: float = 1.0,
        enable_harmonics: bool = False,
        prob_harmonic: float = 0.15,
        harmonic_types: list[str] | None = None,
        harmonic_volume_factor: float = 0.85,
        max_harmonic_fret: int = 19,
        rng_seed: int | None = None,
        rng=None,
        ml_velocity_model_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            warnings.warn(
                "Positional arguments are deprecated; use keyword arguments",
                DeprecationWarning,
                stacklevel=2,
            )
            arg_names = [
                "global_settings",
                "default_instrument",
                "global_tempo",
                "global_time_signature",
                "global_key_signature_tonic",
                "global_key_signature_mode",
            ]
            for name, val in zip(arg_names, args):
                if name == "global_settings":
                    global_settings = val
                elif name == "default_instrument":
                    default_instrument = val
                elif name == "global_tempo":
                    global_tempo = val
                elif name == "global_time_signature":
                    global_time_signature = val
                elif name == "global_key_signature_tonic":
                    global_key_signature_tonic = val
                elif name == "global_key_signature_mode":
                    global_key_signature_mode = val
        super().__init__(
            global_settings=global_settings,
            default_instrument=default_instrument or m21instrument.Violin(),
            global_tempo=global_tempo,
            global_time_signature=global_time_signature,
            global_key_signature_tonic=global_key_signature_tonic,
            global_key_signature_mode=global_key_signature_mode,
            key=key,
            tempo=tempo,
            emotion=emotion,
            rng=rng,
            ml_velocity_model_path=ml_velocity_model_path,
            **kwargs,
        )
        self.cfg: dict = kwargs.copy()
        ts_obj = get_time_signature_object(global_time_signature)
        self.measure_duration = (
            ts_obj.barDuration.quarterLength if ts_obj else self.bar_length
        )
        self.bar_length = self.measure_duration
        from collections.abc import Sequence

        self.voice_allocation = voice_allocation or {}
        if isinstance(default_velocity_curve, Sequence) and not isinstance(
            default_velocity_curve, str
        ):
            self.default_velocity_curve = self._prepare_velocity_map(
                default_velocity_curve
            )
        else:
            self.default_velocity_curve = self._select_velocity_curve(
                default_velocity_curve
            )
        self.voicing_mode = str(voicing_mode or "close").lower()
        self.divisi = divisi
        self.avoid_low_open_strings = bool(avoid_low_open_strings)
        self.timing_jitter_ms = float(timing_jitter_ms)
        self.timing_jitter_mode = str(timing_jitter_mode or "uniform").lower()
        self.timing_jitter_scale_mode = str(
            timing_jitter_scale_mode or "absolute"
        ).lower()
        self.balance_scale = float(balance_scale)
        if rng_seed is not None:
            try:
                self.rng.seed(int(rng_seed))
            except Exception:
                pass
        self.enable_harmonics = bool(enable_harmonics)
        self.prob_harmonic = float(prob_harmonic)
        self.harmonic_types = harmonic_types or ["natural", "artificial"]
        self.harmonic_volume_factor = float(harmonic_volume_factor)
        self.max_harmonic_fret = int(max_harmonic_fret)
        self._last_parts: dict[str, stream.Part] | None = None
        self._articulation_map = {
            "sustain": None,
            "staccato": articulations.Staccato(),
            "accent": articulations.Accent(),
            "tenuto": articulations.Tenuto(),
            "legato": m21spanner.Slur(),
            "tremolo": expressions.Tremolo(),
            "pizz": expressions.TextExpression("pizz."),
            "arco": expressions.TextExpression("arco"),
            "detaché": articulations.DetachedLegato(),
            "detache": articulations.DetachedLegato(),
            "spiccato": articulations.Spiccato(),
        }
        self._legato_active: dict[str, list[note.NotRest]] = {}
        self._legato_groups: dict[str, list[tuple[note.NotRest, note.NotRest]]] = {}
        self.expression_maps = self._load_expression_maps(expression_maps_path)
        self.emotion_map = self._load_emo_map(expression_maps_path)
        self.expression_map = load_expression_map(expression_maps_path)

    # ------------------------------------------------------------------
    # Effect utilities
    # ------------------------------------------------------------------
    def apply_effect_preset(self, part: stream.Part, name: str) -> None:
        """Apply effect preset ``name`` to *part*."""
        preset = EffectPresetLoader.get(name)
        if not preset:
            return
        meta = getattr(part, "metadata", None)
        if meta is None:
            from music21 import metadata as m21metadata

            part.metadata = m21metadata.Metadata()
            meta = part.metadata
        ir_path = preset.get("ir_file")
        if ir_path:
            meta.ir_file = str(ir_path)
        cc_map_p = preset.get("cc") or {}
        new_events = set()
        for k, v in cc_map_p.items():
            try:
                cc_num = int(k)
                val = int(v)
            except Exception:
                continue
            new_events.add((0.0, cc_num, val))
        base = getattr(part, "extra_cc", [])
        part.extra_cc = merge_cc_events(base, new_events)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compose(
        self,
        *,
        section_data: dict[str, Any],
        vocal_metrics: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, stream.Part]:
        q_len = float(section_data.get("q_length", self.bar_length))
        mapping = self._select_expression_map(section_data)
        prev_curve = self._apply_expression_map_pre(section_data, mapping)
        try:
            result = super().compose(
                section_data=section_data, vocal_metrics=vocal_metrics, **kwargs
            )
        finally:
            if prev_curve is not None:
                self.default_velocity_curve = prev_curve
        if not isinstance(result, dict):
            raise RuntimeError(
                "StringsGenerator expected dict result from _render_part"
            )
        self._apply_expression_map_post(result, mapping, q_len)
        intent = section_data.get("musical_intent", {})
        style = intent.get("style")
        intensity = intent.get("intensity")
        resolved = resolve_expression(
            section_data.get("section_name"), intensity, style, self.expression_map
        )
        for p in result.values():
            self._apply_expression(p, resolved)
        dim_start = section_data.get("dim_start")
        dim_end = section_data.get("dim_end")
        if dim_start is not None and dim_end is not None:
            self.crescendo(
                result, q_len, start_val=int(dim_start), end_val=int(dim_end)
            )
        elif section_data.get("crescendo", q_len >= self.bar_length):
            self.crescendo(result, q_len)

        mute_spec = (
            section_data.get("part_params", {}).get("mute")
            if "mute" in section_data.get("part_params", {})
            else section_data.get("style_params", {}).get("mute")
        )
        if mute_spec is not None:
            val = (
                127
                if str(mute_spec).lower()
                in {"true", "con sord.", "con sord", "con sordino"}
                else 0
            )
            factor = float(mapping.get("mute_velocity_factor", 0.85))
            if val == 127:
                factor *= 1.0 + self.rng.uniform(-0.03, 0.03)
            for p in result.values():
                events = getattr(p, "_extra_cc", set())
                events.add((0.0, cc_map.get("mute_toggle", 20), int(val)))
                p._extra_cc = events
                if val == 127:
                    for n in p.recurse().notes:
                        if n.volume and n.volume.velocity is not None:
                            n.volume.velocity = int(n.volume.velocity * factor)

        bow_flags = section_data.get("part_params", {})
        if bow_flags.get("sul_pont"):
            cc_num = cc_map.get("sul_pont", 64)
            for p in result.values():
                ev = getattr(p, "_extra_cc", set())
                for n in p.recurse().notes:
                    ev.add((float(n.offset), cc_num, 127))
                p._extra_cc = ev
        if bow_flags.get("sul_tasto"):
            cc_num = cc_map.get("sul_tasto", 65)
            for p in result.values():
                ev = getattr(p, "_extra_cc", set())
                for n in p.recurse().notes:
                    ev.add((float(n.offset), cc_num, 127))
                p._extra_cc = ev

        macro = section_data.get("part_params", {}).get("macro_envelope")
        if macro and macro.get("type") in {"cresc", "dim"}:
            beats = float(macro.get("beats", q_len))
            if macro["type"] == "cresc":
                self.crescendo(
                    result,
                    beats,
                    start_val=int(macro.get("start", 20)),
                    end_val=int(macro.get("end", 90)),
                )
            else:
                self.apply_dim(
                    result,
                    beats,
                    start_val=int(macro.get("start", 90)),
                    end_val=int(macro.get("end", 20)),
                )

        for part in result.values():
            finalize_cc_events(part)
        self._last_parts = result

        score = stream.Score()
        for info in self._SECTIONS:
            p = result.get(info.name)
            if p is not None:
                score.insert(0, p)

        try:
            self.last_audio = self.export_audio(score, **self.cfg.get("ir_options", {}))
        except Exception:  # pragma: no cover - best effort
            self.logger.debug("export_audio failed", exc_info=True)
            self.last_audio = None

        return result

    def export_musicxml(self, path: str) -> None:
        if not self._last_parts:
            raise ValueError("No generated parts available for export")
        score = stream.Score()
        for info in self._SECTIONS:
            part = self._last_parts.get(info.name)
            if part is None:
                part = stream.Part(id=info.name)
                part.partName = f"Empty {info.name.replace('_', ' ').title()}"
                part.insert(0, info.instrument)
            score.insert(0, part)
        score.write("musicxml", fp=path)

    def crescendo(
        self,
        parts: dict[str, stream.Part] | stream.Part,
        length_beats: float,
        *,
        start_val: int = 20,
        end_val: int = 80,
    ) -> None:
        """Apply a CC11 ramp from ``start_val`` to ``end_val``."""
        from utilities.cc_tools import add_cc_events

        cc_num = cc_map.get("expression", 11)
        steps = max(2, int(math.ceil(length_beats)))

        def _frac(x: float) -> float:
            return 3 * x * x - 2 * x * x * x

        events = [
            {
                "time": length_beats * (i / steps),
                "cc": cc_num,
                "val": int(round(start_val + (end_val - start_val) * _frac(i / steps))),
            }
            for i in range(steps + 1)
        ]

        target_parts = parts.values() if isinstance(parts, dict) else [parts]
        for p in target_parts:
            add_cc_events(p, events)

    def apply_dim(
        self,
        parts: dict[str, stream.Part] | stream.Part,
        length_beats: float,
        *,
        start_val: int = 90,
        end_val: int = 20,
    ) -> None:
        """Apply a decreasing CC11 ramp."""

        self.crescendo(parts, length_beats, start_val=start_val, end_val=end_val)

    def diminuendo(
        self,
        parts: dict[str, stream.Part] | stream.Part,
        length_beats: float,
        *,
        start_val: int = 90,
        end_val: int = 20,
    ) -> None:
        """Alias for :meth:`apply_dim`."""

        self.apply_dim(parts, length_beats, start_val=start_val, end_val=end_val)

    def _insert_cc(
        self, part: stream.Part, time_ql: float, cc: int, value: int
    ) -> None:
        """Insert or replace a CC event at ``time_ql``."""

        from utilities.cc_tools import merge_cc_events, to_sorted_dicts

        base = getattr(part, "extra_cc", [])
        merged = merge_cc_events(base, [(time_ql, cc, value)])
        part.extra_cc = to_sorted_dicts(merged)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_velocity_map(self, spec: Any) -> list[int]:
        """Return a normalized 128-point velocity curve."""
        curve = resolve_velocity_curve(spec)
        if not curve:
            curve = self._default_log_curve()
        if all(0.0 <= v <= 1.5 for v in curve):
            curve = [int(127 * v) for v in curve]
        else:
            curve = [int(v) for v in curve]
        if len(curve) == 3:
            c0, c1, c2 = curve
            curve = [
                c0 + (c1 - c0) * (i / 3) if i <= 3 else c1 + (c2 - c1) * ((i - 3) / 3)
                for i in range(7)
            ]
        if len(curve) == 7:
            curve = interpolate_7pt(curve)
        if len(curve) != 128:
            result: list[int] = []
            default_curve = curve
            for i in range(128):
                pos = i / 127 * (len(default_curve) - 1)
                idx0 = int(round(pos))
                idx1 = min(len(default_curve) - 1, idx0 + 1)
                frac = pos - idx0
                interpolated = (
                    default_curve[idx0] * (1 - frac) + default_curve[idx1] * frac
                )
                rounded = round(interpolated)
                clipped = min(default_curve[-1] - 1, rounded)
                result.append(int(max(0, min(127, clipped))))
            curve = result
        return [max(0, min(127, int(v))) for v in curve]

    def _select_velocity_curve(
        self, style: Sequence[int | float] | str | None
    ) -> list[int]:
        """Return velocity curve list for *style* or fallback."""
        from collections.abc import Sequence as _Seq

        if style is None:
            base = getattr(self, "default_velocity_curve", None)
            if base is not None:
                return list(base)
            return self._default_log_curve()

        if not isinstance(style, str) and isinstance(style, _Seq):
            try:
                import numpy as np  # type: ignore

                if isinstance(style, np.ndarray):
                    curve = style.tolist()
                else:
                    curve = list(style)
            except Exception:
                curve = list(style)
            return self._prepare_velocity_map(curve)

        curve = resolve_velocity_curve(style)
        if not curve:
            return self._default_log_curve()
        return self._prepare_velocity_map(curve)

    @staticmethod
    def _default_log_curve() -> list[int]:
        min_v, max_v = 32, 112
        result = []
        for i in range(128):
            frac = math.log1p(i) / math.log1p(127)
            result.append(int(round(min_v + (max_v - min_v) * frac)))
        return result

    def _load_expression_maps(self, path: str | None) -> dict[str, dict]:
        """Load expression map definitions merging defaults and overrides."""

        import json

        import yaml

        result: dict[str, dict] = {}
        default_path = (
            Path(__file__).resolve().parents[1] / "data" / "expression_maps.yml"
        )
        for p in [default_path, Path(path)] if path else [default_path]:
            try:
                with open(p, encoding="utf-8") as fh:
                    if str(p).endswith(".json"):
                        data = json.load(fh)
                    else:
                        data = yaml.safe_load(fh)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.error("Failed to load expression maps: %s", exc)
                continue
            if isinstance(data, dict):
                for k, v in data.items():
                    if k == "emotion_map" or not isinstance(v, dict):
                        continue
                    result[str(k)] = dict(v)
        return result

    def _load_emo_map(self, path: str | None) -> dict[tuple[str, str], str]:
        """Load emotion→map lookup merging defaults and overrides."""

        import json

        import yaml

        result: dict[tuple[str, str], str] = {}
        default_path = (
            Path(__file__).resolve().parents[1] / "data" / "expression_maps.yml"
        )

        def _parse(data: dict) -> None:
            emap = data.get("emotion_map")
            if isinstance(emap, dict):
                for emo, sub in emap.items():
                    if isinstance(sub, dict):
                        for inten, name in sub.items():
                            result[(str(emo).lower(), str(inten).lower())] = str(name)

        for p in [default_path, Path(path)] if path else [default_path]:
            try:
                with open(p, encoding="utf-8") as fh:
                    if str(p).endswith(".json"):
                        data = json.load(fh)
                    else:
                        data = yaml.safe_load(fh)
            except Exception as exc:  # pragma: no cover - optional
                self.logger.error("Failed to load emotion map: %s", exc)
                continue
            if isinstance(data, dict):
                _parse(data)
        return result

    @staticmethod
    def _fit_pitch(
        p: pitch.Pitch, low: int, high: int, above: int | None
    ) -> pitch.Pitch:
        base = p.midi
        n_min = math.ceil((low - base) / 12)
        n_max = math.floor((high - base) / 12)
        candidates = [base + 12 * n for n in range(n_min, n_max + 1)]
        if not candidates:
            val = max(min(base, high), low)
            return pitch.Pitch(midi=int(val))
        if above is not None:
            near = [c for c in candidates if abs(c - above) <= 4]
            if near:
                val = min(near, key=lambda x: abs(x - above))
            else:
                val = min(candidates, key=lambda x: abs(x - above))
        else:
            val = min(candidates, key=lambda x: abs(x - base))
        return pitch.Pitch(midi=int(val))

    def _estimate_velocity(self, section: _SectionInfo) -> tuple[int, float] | None:
        """Return (base velocity, factor) for *section*."""
        if not self.default_velocity_curve:
            return None
        idx = min(127, int(round(127 * section.velocity_pos)))
        base = self.default_velocity_curve[idx]
        factor = section.velocity_pos * self.balance_scale
        return base, factor

    def _velocity_for(self, section: _SectionInfo) -> int | None:
        result = self._estimate_velocity(section)
        if result is None:
            return None
        base, factor = result
        val = base * factor
        val = max(20, min(127, int(round(val))))
        return val

    def _pitch_to_midi(self, obj: Any) -> int | None:
        """Return MIDI number from ``obj`` if possible."""
        if obj is None:
            return None
        if hasattr(obj, "pitch") and obj.pitch:
            return obj.pitch.midi
        if hasattr(obj, "root"):
            root = obj.root()
            if root is not None and hasattr(root, "midi"):
                return root.midi
        return None

    # ------------------------------------------------------------------
    # Expression map helpers
    # ------------------------------------------------------------------
    def _select_expression_map(self, section: dict[str, Any]) -> dict[str, Any]:
        pmap = section.get("part_params", {}).get("strings", {})
        name = pmap.get("expression_map")
        if name and name in self.expression_maps:
            return self.expression_maps[name]

        emotion = str(
            section.get("musical_intent", {}).get("emotion", "default")
        ).lower()
        intensity = str(
            section.get("musical_intent", {}).get("intensity", "default")
        ).lower()
        key = (emotion, intensity)
        name = (
            self.emotion_map.get(key)
            or self.emotion_map.get((emotion, "default"))
            or self.emotion_map.get(("default", intensity))
            or "gentle_legato"
        )
        return self.expression_maps.get(name, {})

    def _apply_expression_map_pre(
        self, section: dict[str, Any], mapping: dict[str, Any]
    ) -> list[int] | None:
        arts = mapping.get("articulations") or []
        if arts:
            pmap = section.setdefault("part_params", {}).setdefault("strings", {})
            defaults = list(pmap.get("default_articulations", []))
            for a in arts:
                if a not in defaults:
                    defaults.append(a)
            pmap["default_articulations"] = defaults
        curve = mapping.get("velocity_curve_name")
        if curve:
            prev = self.default_velocity_curve
            self.default_velocity_curve = self._select_velocity_curve(curve)
            return prev
        return None

    def _apply_expression_map_post(
        self, parts: dict[str, stream.Part], mapping: dict[str, Any], length: float
    ) -> None:
        cc_map = mapping.get("cc") or {}
        for k, val in cc_map.items():
            try:
                cc_num = int(k)
            except Exception:
                continue
            if isinstance(val, (list, tuple)) and len(val) == 2:
                self._apply_expression_cc(
                    parts,
                    length=length,
                    start_val=int(val[0]),
                    end_val=int(val[1]),
                    cc_num=cc_num,
                )
            else:
                for p in parts.values():
                    events = getattr(p, "_extra_cc", set())
                    events.add((0.0, cc_num, int(val)))
                    p._extra_cc = events

    def _humanize_timing(
        self,
        el: note.NotRest,
        jitter_ms: float,
        *,
        scale_mode: str = "absolute",
    ) -> None:
        if not jitter_ms:
            return
        jitter_val = jitter_ms
        reference_bpm = 120.0
        if scale_mode == "bpm_relative":
            current = float(self.global_tempo or reference_bpm)
            jitter_val *= reference_bpm / current
        if self.timing_jitter_mode == "gauss":
            jitter = self.rng.gauss(0.0, jitter_val / 2.0)
        else:
            jitter = self.rng.uniform(-jitter_val / 2.0, jitter_val / 2.0)
        ql_shift = jitter * reference_bpm / 60000.0
        new_offset = float(el.offset) + ql_shift
        el.offset = max(0.0, new_offset)

    def _apply_vibrato(
        self,
        el: note.NotRest | chord.Chord,
        depth: float,
        rate_hz: float,
    ) -> None:
        """Approximate vibrato by storing a microtone curve."""
        bpm = float(self.global_tempo or 120.0)
        dur_sec = float(el.quarterLength) * 60.0 / bpm
        step = 0.1
        curve: list[tuple[float, float]] = []
        t = 0.0
        while t <= dur_sec + 1e-6:
            cents = depth * 100.0 * math.sin(2 * math.pi * rate_hz * t)
            curve.append((t, cents))
            t += step
        targets = el.notes if hasattr(el, "notes") else [el]
        for n in targets:
            n.pitch.microtone = pitch.Microtone(curve[0][1])
            n.editorial.vibrato_curve = curve

    def _apply_expression(self, part: stream.Part, mapping: dict[str, Any]) -> None:
        """Apply expression map to *part*."""

        dyn_map = {"p": 40, "mp": 55, "mf": 70, "f": 90, "ff": 110}
        base_dyn = mapping.get("base_dynamic")
        if base_dyn:
            val = dyn_map.get(str(base_dyn).lower(), 64)
            self._insert_cc(part, 0.0, cc_map.get("expression", 11), val)
            for n in part.recurse().notes:
                if n.volume is None:
                    n.volume = volume.Volume(velocity=val)
                else:
                    n.volume.velocity = val

        artic = mapping.get("default_artic")
        if artic:
            art_obj = self._articulation_map.get(str(artic))
            if art_obj:
                for n in part.recurse().notes:
                    n.articulations = [copy.deepcopy(art_obj)]

        if mapping.get("crescendo"):
            length = float(part.duration.quarterLength)
            self.crescendo(part, length)

        vib = mapping.get("vibrato") or {}
        depth = vib.get("depth")
        rate = vib.get("rate") or vib.get("rate_hz")
        delay = float(vib.get("delay", 0.0))
        if depth and rate:
            depth_val = float(depth) / 100.0
            rate_val = float(rate)
            for n in part.recurse().notes:
                if n.offset >= delay:
                    self._apply_vibrato(n, depth_val, rate_val)

        if mapping.get("mute"):
            self._insert_cc(part, 0.0, cc_map.get("mute_toggle", 20), 64)
            for n in part.recurse().notes:
                n.expressions.append(expressions.TextExpression("con sord."))

        bow = mapping.get("bow_position")
        if bow:
            val = {"pont": 90, "tasto": 40, "ordinario": 64}.get(str(bow), 64)
            self._insert_cc(part, 0.0, cc_map.get("bow_position", 71), val)

    # ------------------------------------------------------------------
    # Articulation helpers
    # ------------------------------------------------------------------
    def _handle_legato(
        self, part_name: str, note_obj: note.NotRest, apply: bool
    ) -> None:
        if apply:
            buf = self._legato_active.get(part_name)
            if buf:
                buf[1] = note_obj
            else:
                self._legato_active[part_name] = [note_obj, note_obj]
        else:
            buf = self._legato_active.pop(part_name, None)
            if buf and buf[0] != buf[1]:
                self._legato_groups.setdefault(part_name, []).append((buf[0], buf[1]))

    def _apply_articulations(
        self,
        elem: note.NotRest,
        art_names: Any,
        part_name: str,
    ) -> bool:
        names = parse_articulation_field(art_names)
        legato = False
        for art_name in names:
            art_obj = self._articulation_map.get(art_name)
            if art_obj is None:
                if art_name in self._articulation_map:
                    continue
                self.logger.warning("Unknown articulation '%s'", art_name)
                continue
            if art_name == "legato":
                legato = True
            elif art_name in {"pizz", "arco"}:
                elem.expressions.append(copy.deepcopy(art_obj))
            elif art_name == "tremolo" and isinstance(elem, chord.Chord):
                trem = copy.deepcopy(art_obj)
                if hasattr(trem, "rapid"):
                    trem.rapid = True
                elem.expressions.append(trem)
            else:
                elem.articulations.append(copy.deepcopy(art_obj))
        return legato  # legato グループ操作は呼び出し側で

    def _create_notes_from_event(
        self,
        base_pitch: pitch.Pitch | chord.Chord,
        duration_ql: float,
        part_name: str,
        event_articulations: list[str] | None,
        velocity: int | None,
        velocity_factor: float = 1.0,
        bow_position: BowPosition | None = None,
        event_opts: dict | None = None,
    ) -> list[note.NotRest]:
        pattern = (event_opts or {}).get("pattern_type") if event_opts else None
        pattern = str(pattern).lower() if pattern else ""
        result: list[note.NotRest] = []
        if pattern in {EXEC_STYLE_TRILL, EXEC_STYLE_TREMOLO}:
            rate_hz = float((event_opts or {}).get("rate_hz", 6))
            spacing = 60.0 / (float(self.global_tempo or 120.0) * rate_hz)
            interval_val = int((event_opts or {}).get("interval", 1))
            if isinstance(base_pitch, chord.Chord):
                if base_pitch.root():
                    p_base = base_pitch.root().pitch
                else:
                    p_base = base_pitch.pitches[0]
            else:
                p_base = base_pitch
            p_alt = (
                p_base.transpose(interval_val)
                if pattern == EXEC_STYLE_TRILL
                else p_base
            )
            t = 0.0
            toggle = False
            while t < duration_ql - 1e-6:
                dur = min(spacing, duration_ql - t)
                p_sel = p_base if pattern == EXEC_STYLE_TREMOLO or toggle else p_alt
                n = note.Note(p_sel, quarterLength=dur)
                n.offset = t
                n.articulations.append(
                    articulations.Trill()
                    if pattern == EXEC_STYLE_TRILL
                    else articulations.Tremolo(3)
                )
                result.append(n)
                t += spacing
                toggle = not toggle
        else:
            if isinstance(base_pitch, chord.Chord):
                n = chord.Chord(base_pitch.pitches, quarterLength=duration_ql)
            else:
                n = note.Note(base_pitch, quarterLength=duration_ql)
            result.append(n)
        if velocity is not None:
            final_vel = max(1, min(127, int(round(velocity * velocity_factor))))
            vol = volume.Volume(velocity=final_vel)
            try:
                vol.velocityScalar = final_vel / 127.0
            except Exception:
                pass
            if hasattr(vol, "expressiveDynamic"):
                try:
                    vol.expressiveDynamic = final_vel / 127.0
                except Exception:
                    pass
            for n_el in result:
                n_el.volume = vol
        if bow_position:
            value = bow_position.value
            for n_el in result:
                if hasattr(n_el.style, "bowPosition"):
                    setattr(n_el.style, "bowPosition", value)
                else:
                    setattr(n_el.style, "other", value)

        if self.enable_harmonics:
            for elem in result:
                if isinstance(elem, chord.Chord):
                    info = next(
                        (s for s in self._SECTIONS if s.name == part_name), None
                    )
                    base_midis = info.open_string_midi if info else None
                    for idx, n_el in enumerate(elem.notes):
                        new_p, arts, factor, meta = apply_harmonic_to_pitch(
                            n_el.pitch,
                            chord_pitches=elem.pitches,
                            tuning_offsets=[0, 0, 0, 0],
                            base_midis=base_midis,
                            max_fret=self.max_harmonic_fret,
                            allowed_types=self.harmonic_types,
                            rng=self.rng,
                            prob=self.prob_harmonic,
                            volume_factor=self.harmonic_volume_factor,
                        )
                        if arts:
                            for art in arts:
                                n_el.articulations.append(art)
                            if n_el.volume and n_el.volume.velocity is not None:
                                n_el.volume.velocity = scale_velocity(
                                    n_el.volume.velocity,
                                    factor,
                                )
                            if meta:
                                apply_harmonic_notation(n_el, meta)
                        n_el.pitch = new_p
                else:
                    info = next(
                        (s for s in self._SECTIONS if s.name == part_name), None
                    )
                    base_midis = info.open_string_midi if info else None
                    new_p, arts, factor, meta = apply_harmonic_to_pitch(
                        elem.pitch,
                        chord_pitches=[elem.pitch],
                        tuning_offsets=[0, 0, 0, 0],
                        base_midis=base_midis,
                        max_fret=self.max_harmonic_fret,
                        allowed_types=self.harmonic_types,
                        rng=self.rng,
                        prob=self.prob_harmonic,
                        volume_factor=self.harmonic_volume_factor,
                    )
                    if arts:
                        for art in arts:
                            elem.articulations.append(art)
                        if elem.volume and elem.volume.velocity is not None:
                            elem.volume.velocity = scale_velocity(
                                elem.volume.velocity,
                                factor,
                            )
                        if meta:
                            apply_harmonic_notation(elem, meta)
                    elem.pitch = new_p

        return result

    def _finalize_part(self, part: stream.Part, part_name: str) -> stream.Part:
        buf = self._legato_active.pop(part_name, None)
        if buf and buf[0] != buf[1]:
            self._legato_groups.setdefault(part_name, []).append((buf[0], buf[1]))
        for start, end in self._legato_groups.get(part_name, []):
            try:
                part.insert(0, m21spanner.Slur(start, end))
            except Exception:
                pass
        self._legato_groups[part_name] = []
        return part

    def _split_durations(self, q_len: float) -> list[float]:
        remaining = q_len
        segments: list[float] = []
        while remaining > 0:
            if remaining > self.bar_length:
                seg = self.bar_length * 0.95
            else:
                seg = remaining
            segments.append(seg)
            remaining -= seg
        if not segments:
            segments.append(q_len)
        total = sum(segments)
        if abs(total - q_len) > 1e-6:
            segments[-1] += q_len - total
        return segments

    def _voiced_pitches(self, cs: harmony.ChordSymbol) -> list[pitch.Pitch]:
        pitches_sorted = sorted(
            {p.pitchClass: p for p in cs.pitches}.values(), key=lambda p: p.midi
        )
        if pitches_sorted and pitches_sorted[0].octave <= 3:
            pitches_sorted = [p.transpose(12) for p in pitches_sorted]
        if self.voicing_mode == "open":
            voiced = [p.transpose(12 * (i // 2)) for i, p in enumerate(pitches_sorted)]
        elif self.voicing_mode == "spread":
            voiced = [p.transpose(12 * i) for i, p in enumerate(pitches_sorted)]
        else:
            voiced = pitches_sorted
        return voiced

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------
    def _render_part(
        self,
        section_data: dict[str, Any],
        next_section_data: dict[str, Any] | None = None,
        vocal_metrics: dict | None = None,
    ) -> dict[str, stream.Part]:
        chord_label = (
            section_data.get("chord_symbol_for_voicing")
            or section_data.get("original_chord_label")
            or "C"
        )
        q_length = float(section_data.get("q_length", self.bar_length))
        events = section_data.get("events")
        if events:
            durations = [float(e.get("duration", 1.0)) for e in events]
            event_articulations = [e.get("articulations") for e in events]
        else:
            durations = self._split_durations(q_length)
            event_articulations = [None] * len(durations)
        try:
            cs = harmony.ChordSymbol(chord_label)
            base_pitches = self._voiced_pitches(cs)
        except Exception as exc:  # pragma: no cover - parsing errors are logged
            self.logger.error("Invalid chord '%s': %s", chord_label, exc)
            base_pitches = []

        parts: dict[str, stream.Part] = {}
        if not base_pitches:
            for info in self._SECTIONS:
                part = stream.Part(id=info.name)
                part.insert(0, info.instrument)
                for dur in durations:
                    part.append(note.Rest(quarterLength=dur))
                parts[info.name] = part
            return parts

        # ── after: unified implementation ──
        # 元の音高列を保持しておく
        if not base_pitches:
            raise ValueError("base_pitches must contain at least one pitch")

        original_len = len(base_pitches)
        base_orig = list(base_pitches)  # 参照用コピー

        # _SECTIONS の数に合わせて必要分を追加
        while len(base_pitches) < len(self._SECTIONS):
            if self.voicing_mode == "close":
                # クローズ Voicing：常に最低音を複製して密集配置
                idx = 0
            else:
                # オープン／その他：元配列を循環参照
                idx = len(base_pitches) % original_len
            base_pitches.append(base_orig[idx])

        extras_map: dict[str, list[pitch.Pitch]] = {s.name: [] for s in self._SECTIONS}
        if not self.divisi and len(base_pitches) > len(self._SECTIONS):
            extras = base_pitches[len(self._SECTIONS) :]
            target_sections = [
                s.name
                for s in self._SECTIONS
                if s.name in {"violin_i", "violin_ii", "viola"}
            ]
            t_idx = 0
            for p_extra in extras:
                for _ in range(len(target_sections)):
                    sec_name = target_sections[t_idx % len(target_sections)]
                    sec_info = next(s for s in self._SECTIONS if s.name == sec_name)
                    low = pitch.Pitch(sec_info.range_low).midi
                    high = pitch.Pitch(sec_info.range_high).midi
                    adj_extra = self._fit_pitch(p_extra, low, high, None)
                    if low <= adj_extra.midi <= high:
                        extras_map[sec_name].append(adj_extra)
                        t_idx += 1
                        break
                    t_idx += 1

        prev_midi: int | None = None
        divisi_map: dict[str, str] = {}
        if isinstance(self.divisi, bool) and self.divisi:
            divisi_map = {"violin_i": "octave", "violin_ii": "octave"}
        elif isinstance(self.divisi, dict):
            divisi_map = {k: str(v) for k, v in self.divisi.items()}

        default_arts = (
            section_data.get("part_params", {})
            .get("strings", {})
            .get("default_articulations")
        )

        for idx, info in enumerate(self._SECTIONS):
            pitch_idx = self.voice_allocation.get(info.name, idx)
            part = stream.Part(id=info.name)
            part.partName = info.name.replace("_", " ").title()
            part.insert(0, info.instrument)

            if pitch_idx is None or pitch_idx < 0:
                for dur in durations:
                    part.append(note.Rest(quarterLength=dur))
                parts[info.name] = part
                continue

            src = base_pitches[pitch_idx % len(base_pitches)]
            low = pitch.Pitch(info.range_low).midi
            high = pitch.Pitch(info.range_high).midi
            ref_pitch = prev_midi if self.voicing_mode == "close" else None
            adj = self._fit_pitch(src, low, high, ref_pitch)
            if self.avoid_low_open_strings and info.name in {
                "violin_i",
                "violin_ii",
                "viola",
            }:
                name_oct = adj.nameWithOctave
                if info.name == "viola" and name_oct in {"C3", "G3"}:
                    if adj.midi + 12 <= high:
                        adj = adj.transpose(12)
                elif info.name != "viola" and name_oct == "G3":
                    if adj.midi + 12 <= high:
                        adj = adj.transpose(12)
            vel_info = self._estimate_velocity(info)
            if vel_info is None:
                vel_base, vel_factor = None, 1.0
            else:
                vel_base, vel_factor = vel_info
            offset = 0.0
            prev_note: note.NotRest | None = None
            last_bow = None
            for i, dur in enumerate(durations):
                arts = event_articulations[i] if i < len(event_articulations) else None
                if not arts:
                    arts = default_arts
                bow_pos = None
                vib_spec = None
                if events and i < len(events):
                    bow_pos = parse_bow_position(events[i].get("bow_position"))
                    vib_spec = events[i].get("vibrato")
                if bow_pos is None:
                    bow_pos = parse_bow_position(section_data.get("bow_position"))
                if vib_spec is None:
                    vib_spec = (
                        section_data.get("part_params", {})
                        .get("strings", {})
                        .get("vibrato")
                    )
                base_obj: pitch.Pitch | chord.Chord
                pitch_list = [adj]
                if extras_map.get(info.name):
                    pitch_list.extend(extras_map[info.name])
                base_obj = chord.Chord(pitch_list) if len(pitch_list) > 1 else adj
                val_map = {
                    BowPosition.TASTO: 20,
                    BowPosition.PONTICELLO: 100,
                    BowPosition.NORMALE: 64,
                    None: 64,
                }
                cc_val = val_map.get(bow_pos, 64)
                if cc_val != last_bow:
                    ev = getattr(part, "_extra_cc", set())
                    ev.add((offset, cc_map.get("bow_position", 71), cc_val))
                    part._extra_cc = ev
                    last_bow = cc_val
                notes_gen = self._create_notes_from_event(
                    base_obj,
                    dur,
                    info.name,
                    arts,
                    vel_base,
                    vel_factor,
                    bow_pos,
                    events[i] if events and i < len(events) else None,
                )
                for j, n in enumerate(notes_gen):
                    is_legato = self._apply_articulations(n, arts, info.name)
                    self._handle_legato(info.name, n, is_legato)
                    if not is_legato:
                        self._handle_legato(info.name, n, False)
                    self._humanize_timing(
                        n,
                        self.timing_jitter_ms,
                        scale_mode=self.timing_jitter_scale_mode,
                    )
                    if vib_spec:
                        depth = float(vib_spec.get("depth", 0.0))
                        rate = float(vib_spec.get("rate_hz", 5.5))
                        if depth and rate:
                            self._apply_vibrato(n, depth, rate)
                    if len(durations) > 1 and len(notes_gen) == 1 and j == 0:
                        if i == 0:
                            n.tie = tie.Tie("start")
                        elif i == len(durations) - 1:
                            n.tie = tie.Tie("stop")
                        else:
                            n.tie = tie.Tie("continue")
                    elem: note.NotRest = n
                if info.name in divisi_map:
                    mode = divisi_map[info.name]
                    if mode == "octave":
                        extra_pitch = n.pitch.transpose(12)
                    elif mode == "third":
                        key_obj = get_key_signature_object(
                            self.global_key_signature_tonic,
                            self.global_key_signature_mode,
                        )
                        if key_obj:
                            deg = key_obj.getScaleDegreeFromPitch(n.pitch)
                            if deg is not None:
                                extra_pitch = key_obj.pitchFromDegree(deg + 2)
                                extra_pitch.octave = n.pitch.octave
                                if extra_pitch.midi <= n.pitch.midi:
                                    extra_pitch.octave += 1
                            else:
                                qual = "M3" if key_obj.mode == "major" else "m3"
                                extra_pitch = interval.Interval(qual).transposePitch(
                                    n.pitch
                                )
                        else:
                            qual = (
                                "M3"
                                if (self.global_key_signature_mode or "major")
                                == "major"
                                else "m3"
                            )
                            extra_pitch = interval.Interval(qual).transposePitch(
                                n.pitch
                            )
                    else:
                        self.logger.warning(
                            "Unknown divisi '%s' \u2013 defaulting to +4 semitones",
                            mode,
                        )
                        extra_pitch = n.pitch.transpose(4)
                    if extra_pitch:
                        if extra_pitch.midi > high:
                            if extra_pitch.midi - 12 >= low:
                                extra_pitch = extra_pitch.transpose(-12)
                            else:
                                extra_pitch = None
                    if extra_pitch:
                        chd = chord.Chord([n.pitch, extra_pitch])
                        chd.quarterLength = n.quarterLength
                        chd.offset = n.offset
                        if n.tie:
                            chd.tie = n.tie
                        if n.volume and n.volume.velocity is not None:
                            chd.volume = volume.Volume(velocity=int(n.volume.velocity))
                        elem = chd
                if not is_legato:
                    m_cur = self._pitch_to_midi(n)
                    m_prev = self._pitch_to_midi(prev_note)
                    cond_int = (
                        m_cur is not None
                        and m_prev is not None
                        and abs(m_cur - m_prev) <= 2
                    )
                    if (
                        prev_note
                        and not prev_note.isRest
                        and not n.isRest
                        and prev_note.quarterLength >= 0.5
                        and n.quarterLength >= 0.5
                        and cond_int
                    ):
                        self._handle_legato(info.name, prev_note, True)
                        self._handle_legato(info.name, n, True)
                        self._handle_legato(info.name, n, False)
                    else:
                        self._handle_legato(info.name, n, False)
                part.insert(offset + float(n.offset), elem)
                prev_note = n if not n.isRest else prev_note
                offset += dur
            parts[info.name] = self._finalize_part(part, info.name)
            if self.voicing_mode == "close":
                prev_midi = adj.midi
        return parts

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _merge_identical_bars(self, part: stream.Part) -> stream.Part:
        """Merge consecutive bars with identical content."""
        meas = part.makeMeasures(inPlace=False)
        measures = list(meas.getElementsByClass(stream.Measure))
        if len(measures) <= 1:
            return part

        def _sig(m: stream.Measure) -> tuple:
            items = []
            for el in m.notesAndRests:
                if isinstance(el, chord.Chord):
                    name = tuple(p.nameWithOctave for p in el.pitches)
                elif isinstance(el, note.Note):
                    name = el.pitch.nameWithOctave
                else:
                    name = "Rest"
                items.append(
                    (
                        float(el.offset),
                        name,
                        float(el.quarterLength),
                        el.tie.type if el.tie else None,
                    )
                )
            return tuple(items)

        new_measures: list[stream.Measure] = []
        prev_sig: tuple | None = None
        for m in measures:
            sig = _sig(m)
            if new_measures and sig == prev_sig:
                last = new_measures[-1]
                for e_prev, e_cur in zip(last.notesAndRests, m.notesAndRests):
                    e_prev.quarterLength += e_cur.quarterLength
                    if e_prev.tie:
                        if e_cur.tie and e_cur.tie.type == "stop":
                            e_prev.tie.type = "stop"
                        else:
                            if e_prev.tie.type == "start":
                                e_prev.tie.type = "start"
                    elif e_cur.tie:
                        e_prev.tie = tie.Tie(e_cur.tie.type)
            else:
                new_measures.append(m)
                prev_sig = sig

        new_part = stream.Part(id=part.id)
        for m in part.recurse().getElementsByClass(m21instrument.Instrument):
            new_part.insert(0, m)
            break
        offset = 0.0
        import copy

        for m in new_measures:
            for el in m.notesAndRests:
                new_el = copy.deepcopy(el)
                new_part.insert(offset + el.offset, new_el)
            offset += m.duration.quarterLength
        return new_part

    def _apply_expression_cc(
        self,
        parts: dict[str, stream.Part] | stream.Part,
        crescendo: bool = True,
        *,
        length: float | None = None,
        start_val: int | None = None,
        end_val: int | None = None,
        cc_num: int = cc_map.get("expression", 11),
        curve_type: str = "ease",
    ) -> None:
        """Add a CC11 envelope across *length* quarter lengths.

        Parameters
        ----------
        parts:
            Mapping of part names to ``music21`` Parts.
        crescendo:
            If ``True`` and no explicit ``start_val``/``end_val`` provided,
            ramp from 64 to 80, else from 80 to 64.
        length:
            Envelope duration in quarter lengths. Defaults to bar length.
        start_val:
            Starting CC11 value. Overrides ``crescendo`` when provided.
        end_val:
            Ending CC11 value. Overrides ``crescendo`` when provided.
        """
        from utilities.cc_tools import merge_cc_events

        if length is None:
            length = self.bar_length
        if start_val is None or end_val is None:
            start_val = 64 if crescendo else 80
            end_val = 80 if crescendo else 64

        if isinstance(parts, stream.Part):
            to_iter = {"part": parts}
        else:
            to_iter = parts
        steps = max(2, int(math.ceil(length)))

        def _frac(x: float) -> float:
            if curve_type == "linear":
                return x
            if curve_type == "log":
                return math.log1p(9 * x) / math.log(10)
            # ease-in-out
            return 3 * x * x - 2 * x * x * x

        events = [
            (
                length * (i / steps),
                cc_num,
                int(round(start_val + (end_val - start_val) * _frac(i / steps))),
            )
            for i in range(steps + 1)
        ]
        for p in to_iter.values():
            base_events = getattr(p, "_extra_cc", set())
            merged = merge_cc_events(base_events, events)
            p._extra_cc = set(merged)

    def export_audio(
        self,
        score: stream.Score,
        *,
        ir_set: str = "room",
        sample_rate: int = 48_000,
        normalize: bool = True,
        outfile: Path | str | None = None,
    ) -> np.ndarray:
        """Render ``score`` to audio applying impulse responses."""

        import soundfile as sf

        from utilities import convolver as conv

        audio_mix: np.ndarray | None = None
        for part in score.parts:
            sec_name = getattr(part, "id", "") or ""
            try:
                ir_path = self._select_ir_path(ir_set, sec_name)
            except Exception as exc:
                self.logger.debug("%s", exc)
                continue

            sec_audio = self._render_section(part, ir_path, sample_rate=sample_rate)

            if audio_mix is None:
                audio_mix = sec_audio
            else:
                max_len = max(len(audio_mix), len(sec_audio))
                if audio_mix.ndim == 1:
                    audio_mix = audio_mix[:, None]
                if sec_audio.ndim == 1:
                    sec_audio = sec_audio[:, None]
                if audio_mix.shape[0] < max_len:
                    audio_mix = np.pad(
                        audio_mix, ((0, max_len - audio_mix.shape[0]), (0, 0))
                    )
                if sec_audio.shape[0] < max_len:
                    sec_audio = np.pad(
                        sec_audio, ((0, max_len - sec_audio.shape[0]), (0, 0))
                    )
                audio_mix += sec_audio

        if audio_mix is None:
            raise ValueError("No parts rendered")

        if normalize:
            peak = float(np.max(np.abs(audio_mix)))
            if peak > 0:
                audio_mix = audio_mix / peak

        if outfile is not None:
            sf.write(str(outfile), audio_mix, sample_rate)

        return audio_mix

    def _select_ir_path(self, ir_set: str, section_name: str) -> Path:
        """Return IR path for given set and section."""

        base = Path(__file__).resolve().parents[1] / "data" / "irs"
        ir_set = str(ir_set).lower()
        name = section_name.lower()
        group_a = {"violin_i", "violin_ii", "viola"}
        group_b = {"violoncello", "contrabass"}
        if ir_set not in {"room", "hall"}:
            raise ValueError(f"Unknown IR set: {ir_set}")
        if name in group_a:
            fname = f"strings_{ir_set}_a.wav"
        elif name in group_b:
            fname = f"strings_{ir_set}_b.wav"
        else:
            raise ValueError("Unsupported string section")
        path = base / fname
        if not path.is_file():
            fallback = (
                Path(__file__).resolve().parents[1] / "irs" / "blackface-clean.wav"
            )
            if fallback.is_file():
                return fallback
            raise FileNotFoundError(str(path))
        return path

    def _render_section(
        self, section_stream: stream.Part, ir_path: Path, *, sample_rate: int
    ) -> np.ndarray:
        """Render one section and apply ``ir_path``."""

        from utilities import convolver as conv
        from utilities.arrangement_builder import score_to_pretty_midi

        pm = score_to_pretty_midi(stream.Score([section_stream]))
        try:
            dry = pm.fluidsynth(fs=sample_rate)
        except Exception:
            dur = int(safe_end_time(pm) * sample_rate)
            t = np.linspace(0, dur / sample_rate, dur, endpoint=False)
            dry = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        ir_data, ir_sr = conv.load_ir(str(ir_path))
        if ir_sr != sample_rate:
            from fractions import Fraction

            from scipy.signal import resample_poly

            frac = Fraction(sample_rate, ir_sr).limit_denominator(1000)
            ir_data = resample_poly(ir_data, frac.numerator, frac.denominator, axis=0)
        return conv.convolve_ir(dry, ir_data)


def generate_cc_automation(
    part: stream.Part,
    length_ql: float,
    curve: str,
    cc: int = 11,
) -> None:
    """Add controller automation curve to ``part``."""

    steps = max(2, int(math.ceil(length_ql)))
    curve = str(curve)
    if curve == "cresc_linear":
        start_val, end_val = 40, 90
        shape = "linear"
    elif curve == "dim_exp":
        start_val, end_val = 90, 40
        shape = "exp"
    elif curve == "vibrato_sine":
        start_val, end_val = 60, 100
        shape = "sine"
    else:
        return

    def _interp(x: float) -> float:
        if shape == "linear":
            return x
        if shape == "exp":
            return x * x
        if shape == "sine":
            return (1 - math.cos(math.pi * x)) / 2
        return x

    events = [
        (
            length_ql * (i / steps),
            cc,
            int(round(start_val + (end_val - start_val) * _interp(i / steps))),
        )
        for i in range(steps + 1)
    ]
    base = getattr(part, "extra_cc", set())
    part.extra_cc = merge_cc_events(base, events)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render strings with IR")
    ap.add_argument("--ir-set", default="room", help="IR set name")
    ap.add_argument("--wav-out", type=Path, default=None, help="Output WAV file")
    ns = ap.parse_args()

    gen = StringsGenerator()
    sec = {
        "section_name": "demo",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
    }
    parts = gen.compose(section_data=sec)
    score = stream.Score()
    for p in parts.values():
        score.insert(0, p)
    gen.export_audio(score, ir_set=ns.ir_set, outfile=ns.wav_out)
