# --- START OF FILE generator/guitar_generator.py (BasePartGenerator継承・改修版) ---
import music21
from typing import List, Dict, Optional, Tuple, Any, Sequence, Union, cast
import copy
from pathlib import Path
import yaml
import json
import os

import music21.stream as stream
import music21.note as note
import music21.harmony as harmony
import music21.pitch as pitch
import music21.meter as meter
import music21.duration as music21_duration
from utilities.velocity_curve import resolve_velocity_curve
import music21.interval as interval
import music21.tempo as tempo
import music21.chord as m21chord
import music21.articulations as articulations
import music21.volume as m21volume
from music21 import instrument as m21instrument

import random
import logging
logger = logging.getLogger(__name__)


def _get_string_indicator_cls():
    """Return music21 articulations class for indicating string, if available."""
    for name in ("StringIndication", "StringIndicator"):
        if hasattr(articulations, name):
            return getattr(articulations, name)
    logger.warning("No StringIndication/Indicator class found in music21")
    return None


def _get_fret_indicator_cls():
    """Return music21 articulations class for indicating fret, if available."""
    for name in ("FretIndication", "FretIndicator"):
        if hasattr(articulations, name):
            return getattr(articulations, name)
    logger.warning("No FretIndication/Indicator class found in music21")
    return None
import math

from .base_part_generator import BasePartGenerator
from utilities import humanizer

# Minimum note duration for generated notes (quarterLength)
MIN_NOTE_DURATION_QL = 0.0125  # minimum quarterLength for strum notes

try:
    from utilities.safe_get import safe_get
except ImportError:

    def safe_get(data, key_path, default=None, cast_to=None, log_name="dummy_safe_get"):
        val = data.get(key_path.split(".")[0])
        if val is None:
            return default
        if cast_to:
            try:
                return cast_to(val)
            except:
                return default
        return val


try:
    from utilities.core_music_utils import (
        get_time_signature_object,
        sanitize_chord_label,
    )
except ImportError:

    def get_time_signature_object(ts_str: Optional[str]) -> meter.TimeSignature:
        if not ts_str:
            ts_str = "4/4"
        try:
            return meter.TimeSignature(ts_str)
        except Exception:
            return meter.TimeSignature("4/4")

    def sanitize_chord_label(label: Optional[str]) -> Optional[str]:
        if not label or label.strip().lower() in [
            "rest",
            "r",
            "n.c.",
            "nc",
            "none",
            "-",
        ]:
            return "Rest"
        return label.strip()



DEFAULT_GUITAR_OCTAVE_RANGE: Tuple[int, int] = (2, 5)
GUITAR_STRUM_DELAY_QL: float = 0.02
STANDARD_TUNING_OFFSETS = [0, 0, 0, 0, 0, 0]

# User-friendly tuning presets (semitone offsets per string)
TUNING_PRESETS: Dict[str, List[int]] = {
    "standard": STANDARD_TUNING_OFFSETS,
    "drop_d": [-2, 0, 0, 0, 0, 0],
    "open_g": [-2, -2, 0, 0, 0, -2],
}

EXEC_STYLE_BLOCK_CHORD = "block_chord"
EXEC_STYLE_STRUM_BASIC = "strum_basic"
EXEC_STYLE_ARPEGGIO_FROM_INDICES = "arpeggio_from_indices"
EXEC_STYLE_ARPEGGIO_PATTERN = "arpeggio_pattern"
EXEC_STYLE_POWER_CHORDS = "power_chords"
EXEC_STYLE_MUTED_RHYTHM = "muted_rhythm"
EXEC_STYLE_HAMMER_ON = "hammer_on"
EXEC_STYLE_PULL_OFF = "pull_off"

EMOTION_INTENSITY_MAP: Dict[Tuple[str, str], str] = {
    ("quiet_pain_and_nascent_strength", "low"): "guitar_ballad_arpeggio",
    ("deep_regret_gratitude_and_realization", "medium_low"): "guitar_ballad_arpeggio",
    (
        "acceptance_of_love_and_pain_hopeful_belief",
        "medium_high",
    ): "guitar_folk_strum_simple",
    ("self_reproach_regret_deep_sadness", "medium_low"): "guitar_ballad_arpeggio",
    ("supported_light_longing_for_rebirth", "medium"): "guitar_folk_strum_simple",
    (
        "reflective_transition_instrumental_passage",
        "medium_low",
    ): "guitar_ballad_arpeggio",
    ("trial_cry_prayer_unbreakable_heart", "medium_high"): "guitar_power_chord_8ths",
    ("memory_unresolved_feelings_silence", "low"): "guitar_ballad_arpeggio",
    ("wavering_heart_gratitude_chosen_strength", "medium"): "guitar_folk_strum_simple",
    (
        "reaffirmed_strength_of_love_positive_determination",
        "high",
    ): "guitar_power_chord_8ths",
    ("hope_dawn_light_gentle_guidance", "medium"): "guitar_folk_strum_simple",
    (
        "nature_memory_floating_sensation_forgiveness",
        "medium_low",
    ): "guitar_ballad_arpeggio",
    (
        "future_cooperation_our_path_final_resolve_and_liberation",
        "high_to_very_high_then_fade",
    ): "guitar_power_chord_8ths",
    ("default", "default"): "guitar_default_quarters",
    ("default", "low"): "guitar_ballad_arpeggio",
    ("default", "medium_low"): "guitar_ballad_arpeggio",
    ("default", "medium"): "guitar_folk_strum_simple",
    ("default", "medium_high"): "guitar_folk_strum_simple",
    ("default", "high"): "guitar_power_chord_8ths",
}
DEFAULT_GUITAR_RHYTHM_KEY = "guitar_default_quarters"


class GuitarStyleSelector:
    def __init__(self, mapping: Dict[Tuple[str, str], str] | None = None):
        self.mapping = mapping if mapping is not None else EMOTION_INTENSITY_MAP

    def select(
        self,
        *,
        emotion: str | None,
        intensity: str | None,
        cli_override: str | None = None,
        part_params_override_rhythm_key: str | None = None,
        rhythm_library_keys: List[str],
    ) -> str:
        if cli_override and cli_override in rhythm_library_keys:
            return cli_override
        if (
            part_params_override_rhythm_key
            and part_params_override_rhythm_key in rhythm_library_keys
        ):
            return part_params_override_rhythm_key
        effective_emotion = (emotion or "default").lower()
        effective_intensity = (intensity or "default").lower()
        key = (effective_emotion, effective_intensity)
        style_from_map = self.mapping.get(key)
        if style_from_map and style_from_map in rhythm_library_keys:
            return style_from_map
        style_emo_default = self.mapping.get((effective_emotion, "default"))
        if style_emo_default and style_emo_default in rhythm_library_keys:
            return style_emo_default
        style_int_default = self.mapping.get(("default", effective_intensity))
        if style_int_default and style_int_default in rhythm_library_keys:
            return style_int_default
        if DEFAULT_GUITAR_RHYTHM_KEY in rhythm_library_keys:
            return DEFAULT_GUITAR_RHYTHM_KEY
        if rhythm_library_keys:
            return rhythm_library_keys[0]
        return ""


class GuitarGenerator(BasePartGenerator):
    def __init__(
        self,
        *args,
        tuning: Optional[Union[str, Sequence[int]]] = None,
        timing_variation: float = 0.0,
        gate_length_variation: float = 0.0,
        external_patterns_path: Optional[str] = None,
        hammer_on_interval: int = 2,
        pull_off_interval: int = 2,
        hammer_on_probability: float = 0.5,
        pull_off_probability: float = 0.5,
        default_stroke_direction: str | None = None,
        default_palm_mute: bool = False,
        default_velocity_curve: str | dict | None = None,
        timing_jitter_ms: float = 0.0,
        timing_jitter_mode: str = "uniform",
        strum_delay_jitter_ms: float = 0.0,
        swing_ratio: float | None = None,
        velocity_preset_path: str | None = None,
        accent_map: Dict[int, int] | None = None,
        rr_channel_cycle: Sequence[int] | None = None,
        swing_subdiv: int | None = None,
        position_lock: bool = False,
        preferred_position: int = 0,
        open_string_bonus: int = -1,
        string_shift_weight: int = 2,
        fret_shift_weight: int = 1,
        strict_string_order: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.external_patterns_path = external_patterns_path
        if isinstance(tuning, str):
            self.tuning = TUNING_PRESETS.get(tuning.lower(), STANDARD_TUNING_OFFSETS)
        elif tuning is None:
            self.tuning = STANDARD_TUNING_OFFSETS
        else:
            if len(tuning) != 6:
                raise ValueError("tuning must be length 6")
            self.tuning = [int(x) for x in tuning]
        self.timing_variation = timing_variation
        self.gate_length_variation = gate_length_variation
        self.hammer_on_interval = hammer_on_interval
        self.pull_off_interval = pull_off_interval
        self.hammer_on_probability = hammer_on_probability
        self.pull_off_probability = pull_off_probability
        self.default_stroke_direction = (
            default_stroke_direction.lower() if isinstance(default_stroke_direction, str) else None
        )
        self.default_palm_mute = bool(default_palm_mute)
        self.velocity_preset_path = velocity_preset_path
        self.velocity_presets: Dict[str, Dict[str, Any]] = {}
        self.tuning_name = "standard"
        if self.tuning == TUNING_PRESETS.get("drop_d"):
            self.tuning_name = "drop_d"
        elif self.tuning == TUNING_PRESETS.get("open_g"):
            self.tuning_name = "open_g"
        self._load_velocity_presets()
        if isinstance(default_velocity_curve, (list, tuple, dict)):
            self.default_velocity_curve = self._prepare_velocity_map(default_velocity_curve)
        else:
            self.default_velocity_curve = self._select_velocity_curve(default_velocity_curve)
        self.timing_jitter_ms = float(timing_jitter_ms)
        self.timing_jitter_mode = str(timing_jitter_mode or "uniform").lower()
        self.strum_delay_jitter_ms = float(strum_delay_jitter_ms)
        self.swing_ratio = float(swing_ratio) if swing_ratio is not None else None
        self.accent_map = {int(k): int(v) for k, v in (accent_map or {}).items()}
        self.rr_channel_cycle = [int(c) for c in rr_channel_cycle] if rr_channel_cycle else []
        self._rr_index = 0
        self._rr_last_pitch: int | None = None
        self._prev_note_pitch: pitch.Pitch | None = None
        self.position_lock = bool(position_lock)
        self.preferred_position = int(preferred_position)
        self.open_string_bonus = int(open_string_bonus)
        self.string_shift_weight = int(string_shift_weight)
        self.fret_shift_weight = int(fret_shift_weight)
        self.strict_string_order = bool(strict_string_order)
        from utilities.core_music_utils import get_time_signature_object

        ts_obj = get_time_signature_object(self.global_time_signature)
        self.measure_duration = (
            ts_obj.barDuration.quarterLength if ts_obj else 4.0
        )
        self.cfg: dict = kwargs.copy()
        self.style_selector = GuitarStyleSelector()
        self.swing_subdiv = int(swing_subdiv) if swing_subdiv else 8
        # ここから self.part_parameters を参照・初期化する
        if not hasattr(self, "part_parameters"):
            self.part_parameters = {}
        # 以降、self.part_parameters を安全に使える

        # 安全なフォールバック
        if "guitar_default_quarters" not in self.part_parameters:
            self.part_parameters["guitar_default_quarters"] = {
                "pattern": [
                    {
                        "offset": 0,
                        "duration": 1,
                        "velocity_factor": 0.8,
                        "type": "block",
                    }
                ],
                "reference_duration_ql": 1.0,
                "description": "Failsafe default quarter note strum",
            }

        self._load_external_strum_patterns()
        self._add_internal_default_patterns()
        self.part_parameters.setdefault("hammer_on_interval", self.hammer_on_interval)
        self.part_parameters.setdefault("pull_off_interval", self.pull_off_interval)
        self.part_parameters.setdefault("hammer_on_probability", self.hammer_on_probability)
        self.part_parameters.setdefault("pull_off_probability", self.pull_off_probability)
        if "stroke_direction" not in self.part_parameters and self.default_stroke_direction is not None:
            self.part_parameters["stroke_direction"] = self.default_stroke_direction
        if "palm_mute" not in self.part_parameters:
            self.part_parameters["palm_mute"] = self.default_palm_mute

        self._articulation_map = {
            "palm_mute": articulations.FretIndication("palm mute"),
            "staccato": articulations.Staccato(),
            "accent": articulations.Accent(),
            "ghost_note": articulations.FretIndication("ghost note"),
            "slide": articulations.IndeterminateSlide(),
            "slide_in": articulations.IndeterminateSlide(),
            "bend": articulations.FretBend(),
            "hammer_on": articulations.HammerOn(),
            "pull_off": articulations.PullOff(),
        }

        if not self.default_velocity_curve or len(self.default_velocity_curve) != 128:
            self.default_velocity_curve = [
                max(0, min(127, int(round(40 + 45 * math.sin((math.pi / 2) * i / 127)))))
                for i in range(128)
            ]


    def compose(self, *args, **kwargs):
        self._prev_note_pitch = None
        section = kwargs.get("section_data", {}) if kwargs else {}
        part_params = section.get("part_params", {})
        orig_subdiv = self.swing_subdiv
        if isinstance(part_params, dict) and "swing_subdiv" in part_params:
            try:
                self.swing_subdiv = int(part_params.get("swing_subdiv"))
            except Exception:
                pass
        ratio_to_apply = None
        if isinstance(part_params, dict):
            ratio_to_apply = part_params.pop("swing_ratio", None)
            section["part_params"] = part_params
            kwargs["section_data"] = section
        if ratio_to_apply is None:
            ratio_to_apply = self.swing_ratio

        result = super().compose(*args, **kwargs)
        if isinstance(result, stream.Part):
            self._last_part = result
            if ratio_to_apply is not None:
                self._apply_swing_internal(self._last_part, float(ratio_to_apply), self.swing_subdiv)
        elif isinstance(result, dict) and result:
            self._last_part = next(iter(result.values()))
            if ratio_to_apply is not None:
                for p in result.values():
                    self._apply_swing_internal(p, float(ratio_to_apply), self.swing_subdiv)
        else:
            self._last_part = None
        self.swing_subdiv = orig_subdiv
        return result

    def _get_guitar_friendly_voicing(
        self,
        cs: harmony.ChordSymbol,
        num_strings: int = 6,
        preferred_octave_bottom: int = 2,
    ) -> List[pitch.Pitch]:
        if not cs or not cs.pitches:
            return []
        original_pitches = list(cs.pitches)
        try:
            temp_chord = cs.closedPosition(
                forceOctave=preferred_octave_bottom, inPlace=False
            )
            candidate_pitches = sorted(
                list(temp_chord.pitches), key=lambda p_sort: p_sort.ps
            )
        except Exception as e_closed_pos:
            logger.warning(
                f"GuitarGen: Error in closedPosition for {cs.figure}: {e_closed_pos}. Using original pitches."
            )
            candidate_pitches = sorted(original_pitches, key=lambda p_sort: p_sort.ps)
        if not candidate_pitches:
            logger.warning(
                f"GuitarGen: No candidate pitches for {cs.figure} after closedPosition. Returning empty."
            )
            return []
        guitar_min_ps = pitch.Pitch(f"E{DEFAULT_GUITAR_OCTAVE_RANGE[0]}").ps
        guitar_max_ps = pitch.Pitch(f"B{DEFAULT_GUITAR_OCTAVE_RANGE[1]}").ps
        if candidate_pitches and candidate_pitches[0].ps < guitar_min_ps:
            oct_shift = math.ceil((guitar_min_ps - candidate_pitches[0].ps) / 12.0)
            candidate_pitches = [
                p_cand.transpose(int(oct_shift * 12)) for p_cand in candidate_pitches
            ]
            candidate_pitches.sort(key=lambda p_sort: p_sort.ps)
        selected_dict: Dict[str, pitch.Pitch] = {}
        for p_cand_select in candidate_pitches:
            if guitar_min_ps <= p_cand_select.ps <= guitar_max_ps:
                if p_cand_select.name not in selected_dict:
                    selected_dict[p_cand_select.name] = p_cand_select
        final_voiced_pitches = sorted(
            list(selected_dict.values()), key=lambda p_sort: p_sort.ps
        )
        return self._apply_tuning(final_voiced_pitches[:num_strings])

    def _apply_tuning(self, pitches: List[pitch.Pitch]) -> List[pitch.Pitch]:
        tuned = []
        for i, p in enumerate(pitches):
            offset = self.tuning[i % len(self.tuning)]
            tuned.append(p.transpose(offset))
        return tuned

    def _resolve_curve(self, spec: Any) -> List[int]:
        curve = resolve_velocity_curve(spec)
        if not curve:
            return []
        if all(0.0 <= v <= 1.5 for v in curve):
            return [int(127 * v) for v in curve]
        return [int(v) for v in curve]

    def _prepare_velocity_map(self, spec: Any) -> list[int] | None:
        curve = self._resolve_curve(spec)
        if not curve:
            return None
        if len(curve) == 128:
            return [max(0, min(127, int(v))) for v in curve]
        result: list[int] = []
        for i in range(128):
            pos = i / 127 * (len(curve) - 1)
            idx0 = int(math.floor(pos))
            idx1 = min(len(curve) - 1, idx0 + 1)
            frac = pos - idx0
            val = curve[idx0] * (1 - frac) + curve[idx1] * frac
            result.append(max(0, min(127, int(round(val)))) )
        return result

    def _load_velocity_presets(self) -> None:
        path = self.velocity_preset_path
        if not path:
            return
        if not os.path.exists(path):
            logger.warning("Velocity preset path '%s' not found", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.lower().endswith(".json"):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
        except Exception as e:
            logger.warning("Failed to load velocity presets from %s: %s", path, e)
            return
        if not isinstance(data, dict):
            logger.warning("Velocity preset file format invalid: %s", path)
            return
        for tun, presets in data.items():
            if not isinstance(presets, dict):
                continue
            m: Dict[str, Any] = {}
            for style, curve in presets.items():
                if isinstance(curve, list) and len(curve) in (7, 128):
                    m[style] = curve
                else:
                    logger.warning("Invalid curve for %s/%s", tun, style)
            if m:
                self.velocity_presets[tun] = m

    def _select_velocity_curve(self, style_name: str | None) -> list[int] | None:
        curve: list[int] | None = None
        presets = self.velocity_presets.get(self.tuning_name) or {}
        if style_name and style_name in presets:
            curve = self._prepare_velocity_map(presets[style_name])
        elif style_name is None and "default" in presets:
            curve = self._prepare_velocity_map(presets["default"])
        elif style_name and "default" in presets:
            curve = self._prepare_velocity_map(presets["default"])
        if curve is None and style_name:
            curve = self._prepare_velocity_map(style_name)
        if curve is None:
            curve = [
                max(0, min(127, int(round(40 + 45 * math.sin((math.pi / 2) * i / 127)))))
                for i in range(128)
            ]
        return curve

    def _apply_round_robin(self, el: Union[note.Note, m21chord.Chord]) -> None:
        if not self.rr_channel_cycle:
            return
        pitch_val: int | None = None
        if isinstance(el, note.Note):
            pitch_val = int(el.pitch.midi)
        elif isinstance(el, m21chord.Chord) and el.pitches:
            pitch_val = int(el.pitches[0].midi)
        if pitch_val is None:
            return
        if pitch_val == self._rr_last_pitch:
            self._rr_index = (self._rr_index + 1) % len(self.rr_channel_cycle)
        else:
            self._rr_index = 0
        self._rr_last_pitch = pitch_val
        ch = self.rr_channel_cycle[self._rr_index]
        setattr(el, "channel", ch)

    def _jitter(self, offset: float) -> float:
        if self.timing_variation:
            offset += self.rng.uniform(-self.timing_variation, self.timing_variation)
            if offset < 0:
                offset = 0.0
        return offset

    def _humanize_timing(self, el: note.NotRest, jitter_ms: float) -> None:
        if not jitter_ms:
            return
        if self.timing_jitter_mode == "gauss":
            jitter = self.rng.gauss(0.0, jitter_ms / 2.0)
        else:
            jitter = self.rng.uniform(-jitter_ms / 2.0, jitter_ms / 2.0)
        ql_shift = jitter * self.global_tempo / 60000.0
        new_offset = float(el.offset) + ql_shift
        el.offset = max(0.0, new_offset)

    def _apply_swing_internal(self, part: stream.Part, ratio: float, subdiv: int) -> None:
        if ratio is None or abs(ratio - 0.5) < 1e-6:
            return
        if not subdiv or subdiv <= 1:
            return
        pair = (4.0 / subdiv) * 2.0
        if pair <= 0:
            return
        step = pair / 2.0
        if step <= 0:
            return
        tol = step * 0.1
        notes = list(part.recurse().notes)
        for i, n in enumerate(notes):
            pos = float(n.offset)
            pair_start = math.floor(pos / pair) * pair
            within = pos - pair_start
            if abs(within - step) < tol:
                target = pair_start + step + (ratio - 0.5) * pair
                if i > 0:
                    prev = notes[i - 1]
                    prev.duration.quarterLength = max(
                        MIN_NOTE_DURATION_QL,
                        step - (ratio - 0.5) * pair,
                    )
                try:
                    n.setOffsetBySite(part, target)
                except Exception:
                    n.offset = target
        part.coreElementsChanged()

    def _estimate_fingering(
        self, pitches: List[pitch.Pitch]
    ) -> List[Tuple[int, int]]:
        """Estimate guitar fingering (string index, fret) for a sequence of pitches."""
        # Standard tuning MIDI numbers
        base_midis = [40, 45, 50, 55, 59, 64]
        tuned = [m + off for m, off in zip(base_midis, self.tuning)]

        candidates: List[List[Tuple[int, int]]] = []
        max_fret = 20
        for p in pitches:
            opts = []
            pm = int(round(p.midi))
            for s, open_m in enumerate(tuned):
                fret = pm - open_m
                if fret < 0 or fret > max_fret:
                    continue
                if self.position_lock and abs(fret - self.preferred_position) > 2:
                    continue
                opts.append((s, fret))
            if not opts:
                opts.append((0, max(0, pm - tuned[0])))
            candidates.append(opts)

        dp: List[Dict[Tuple[int, int], Tuple[float, Tuple[int, int] | None]]] = []
        for i, opts in enumerate(candidates):
            layer: Dict[Tuple[int, int], Tuple[float, Tuple[int, int] | None]] = {}
            if i == 0:
                for o in opts:
                    cost = self.open_string_bonus if o[1] == 0 else 0
                    layer[o] = (cost, None)
            else:
                prev_layer = dp[i - 1]
                for o in opts:
                    best_cost = float("inf")
                    best_prev: Tuple[int, int] | None = None
                    for prev_o, (prev_cost, _) in prev_layer.items():
                        cost = (
                            prev_cost
                            + abs(o[0] - prev_o[0]) * self.string_shift_weight
                            + abs(o[1] - prev_o[1]) * self.fret_shift_weight
                            + (self.open_string_bonus if o[1] == 0 else 0)
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_prev = prev_o
                    layer[o] = (best_cost, best_prev)
            dp.append(layer)

        # backtrack
        if not dp:
            return []
        last_layer = dp[-1]
        current = min(last_layer.items(), key=lambda x: x[1][0])[0]
        result: List[Tuple[int, int]] = [current]
        for i in range(len(dp) - 1, 0, -1):
            prev = dp[i][current][1]
            if prev is None:
                break
            current = prev
            result.append(current)
        result.reverse()

        # validate frets
        for idx, (s_idx, fret) in enumerate(result):
            if fret > max_fret:
                # try open string alternative
                target_midi = int(round(pitches[idx].midi))
                for s, open_m in enumerate(tuned):
                    if target_midi == open_m:
                        result[idx] = (s, 0)
                        break
                else:
                    best: Tuple[int, int] | None = None
                    best_diff = None
                    for s, open_m in enumerate(tuned):
                        fr = target_midi - open_m
                        if 0 <= fr <= max_fret and abs(fr - self.preferred_position) <= 4:
                            diff = abs(fr - self.preferred_position)
                            if best_diff is None or diff < best_diff:
                                best_diff = diff
                                best = (s, fr)
                    if best is not None:
                        result[idx] = best
                    else:
                        logger.warning(
                            "Fingering exceeds max fret: pitch %s -> fret %d", pitches[idx], fret
                        )
                        result[idx] = (s_idx, max_fret)
        return result

    def _attach_fingering(
        self, n: note.Note, string_idx: int, fret: int
    ) -> None:
        """Attach fingering info to a note using Notations if available."""
        setattr(n, "string", string_idx)
        setattr(n, "fret", fret)
        try:
            if not hasattr(n, "notations"):
                n.notations = note.Notations()
            StringCls = _get_string_indicator_cls()
            FretCls = _get_fret_indicator_cls()
            if StringCls:
                try:
                    n.notations.append(StringCls(number=int(string_idx) + 1))
                except Exception:
                    pass
            if FretCls:
                try:
                    n.notations.append(FretCls(number=int(fret)))
                except Exception:
                    pass
        except Exception:
            pass

    def _create_notes_from_event(
        self,
        cs: harmony.ChordSymbol,
        rhythm_pattern_definition: Dict[str, Any],
        guitar_block_params: Dict[str, Any],
        event_duration_ql: float,
        event_final_velocity: int,
        event_offset_ql: float = 0.0,
    ) -> List[Union[note.Note, m21chord.Chord]]:
        notes_for_event: List[Union[note.Note, m21chord.Chord]] = []
        # イベントパラメータとパターン定義の両方からアーティキュレーションを収集
        art_objs: List[articulations.Articulation] = []
        for src in (guitar_block_params, rhythm_pattern_definition):
            art = src.get("articulation") or src.get("event_articulation")
            if isinstance(art, str):
                base = self._articulation_map.get(art)
                if base is not None:
                    art_objs.append(copy.deepcopy(base))
            elif isinstance(art, list):
                for name in art:
                    base = self._articulation_map.get(name)
                    if base is not None:
                        art_objs.append(copy.deepcopy(base))

        slide_in_offset = guitar_block_params.get(
            "slide_in_offset",
            rhythm_pattern_definition.get("slide_in_offset"),
        )
        slide_out_offset = guitar_block_params.get(
            "slide_out_offset",
            rhythm_pattern_definition.get("slide_out_offset"),
        )
        if slide_in_offset is not None or slide_out_offset is not None:
            slide_art = articulations.IndeterminateSlide()
            if slide_in_offset is not None:
                val = float(slide_in_offset)
                if hasattr(slide_art.editorial, "slide_in_offset"):
                    slide_art.editorial.slide_in_offset = val
                else:
                    slide_art.editorial.setdefault("slide_in_offset", val)
            if slide_out_offset is not None:
                val = float(slide_out_offset)
                if hasattr(slide_art.editorial, "slide_out_offset"):
                    slide_art.editorial.slide_out_offset = val
                else:
                    slide_art.editorial.setdefault("slide_out_offset", val)
            art_objs.append(slide_art)

        bend_amount = guitar_block_params.get(
            "bend_amount",
            rhythm_pattern_definition.get("bend_amount"),
        )
        bend_release_offset = guitar_block_params.get(
            "bend_release_offset",
            rhythm_pattern_definition.get("bend_release_offset"),
        )
        if bend_amount is not None or bend_release_offset is not None:
            bend_art = articulations.FretBend()
            if bend_amount is not None:
                val = float(bend_amount)
                if hasattr(bend_art.editorial, "bend_amount"):
                    bend_art.editorial.bend_amount = val
                else:
                    bend_art.editorial.setdefault("bend_amount", val)
            if bend_release_offset is not None:
                val = float(bend_release_offset)
                if hasattr(bend_art.editorial, "bend_release_offset"):
                    bend_art.editorial.bend_release_offset = val
                else:
                    bend_art.editorial.setdefault(
                        "bend_release_offset", val
                    )
            art_objs.append(bend_art)
        execution_style = rhythm_pattern_definition.get(
            "execution_style", EXEC_STYLE_BLOCK_CHORD
        )

        def _attach_artics(elem: Union[note.Note, m21chord.Chord]) -> None:
            if not art_objs:
                return
            if isinstance(elem, m21chord.Chord):
                for n_el in elem.notes:
                    for art in art_objs:
                        n_el.articulations.append(copy.deepcopy(art))
            else:
                for art in art_objs:
                    elem.articulations.append(copy.deepcopy(art))

        num_strings = guitar_block_params.get(
            "guitar_num_strings",
            guitar_block_params.get(
                "num_strings", 6
            ),  # DEFAULT_CONFIGから取得できるように修正
        )
        preferred_octave_bottom = guitar_block_params.get(
            "guitar_target_octave",
            guitar_block_params.get(
                "target_octave", 3
            ),  # DEFAULT_CONFIGから取得できるように修正
        )
        chord_pitches = self._get_guitar_friendly_voicing(
            cs, num_strings, preferred_octave_bottom
        )
        if not chord_pitches:
            return []

        is_palm_muted = guitar_block_params.get("palm_mute", False)
        stroke_dir = guitar_block_params.get("current_event_stroke") or guitar_block_params.get("stroke_direction")
        if isinstance(stroke_dir, str):
            sd = stroke_dir.lower()
            if sd == "down":
                event_final_velocity = min(127, int(event_final_velocity * 1.1))
            elif sd == "up":
                event_final_velocity = max(1, int(event_final_velocity * 0.9))

        if is_palm_muted:
            event_final_velocity = max(1, int(event_final_velocity * 0.85))

        beat_pos = (event_offset_ql % 4.0) / 4.0
        if self.default_velocity_curve:
            base_velocity = int(
                self.default_velocity_curve[int(round(127 * beat_pos))]
            )
        else:
            base_velocity = event_final_velocity
        accent_adj = int(self.accent_map.get(int(math.floor(event_offset_ql)), 0))
        event_final_velocity = max(1, min(127, base_velocity + accent_adj))

        if execution_style == EXEC_STYLE_POWER_CHORDS and cs.root():
            p_root = pitch.Pitch(cs.root().name)
            target_power_chord_octave = DEFAULT_GUITAR_OCTAVE_RANGE[0]
            if p_root.octave < target_power_chord_octave:
                p_root.octave = target_power_chord_octave
            elif p_root.octave > target_power_chord_octave + 1:
                p_root.octave = target_power_chord_octave + 1

            power_chord_pitches = [p_root, p_root.transpose(interval.PerfectFifth())]
            if num_strings > 2:
                root_oct_up = p_root.transpose(interval.PerfectOctave())
                if (
                    root_oct_up.ps
                    <= pitch.Pitch(f"B{DEFAULT_GUITAR_OCTAVE_RANGE[1]}").ps
                ):
                    power_chord_pitches.append(root_oct_up)

            base_dur = event_duration_ql * (
                0.85 if is_palm_muted else 0.95
            )
            base_dur *= 1 + self.rng.uniform(-self.gate_length_variation, self.gate_length_variation)
            ch = m21chord.Chord(
                power_chord_pitches[:num_strings],
                quarterLength=max(MIN_NOTE_DURATION_QL, base_dur),
            )
            for n_in_ch_note in ch.notes:
                n_in_ch_note.volume.velocity = event_final_velocity
                if is_palm_muted:
                    n_in_ch_note.articulations.append(articulations.Staccatissimo())
            ch.offset = self._jitter(0.0)
            self._humanize_timing(ch, guitar_block_params.get("timing_jitter_ms", self.timing_jitter_ms))
            _attach_artics(ch)
            self._apply_round_robin(ch)
            notes_for_event.append(ch)

        elif execution_style in (EXEC_STYLE_BLOCK_CHORD, EXEC_STYLE_HAMMER_ON, EXEC_STYLE_PULL_OFF):
            base_dur = event_duration_ql * (
                0.85 if is_palm_muted else 0.9
            )
            base_dur *= 1 + self.rng.uniform(-self.gate_length_variation, self.gate_length_variation)
            ch = m21chord.Chord(
                chord_pitches,
                quarterLength=max(MIN_NOTE_DURATION_QL, base_dur),
            )
            for n_in_ch_note in ch.notes:
                n_in_ch_note.volume.velocity = event_final_velocity
                if is_palm_muted:
                    n_in_ch_note.articulations.append(articulations.Staccatissimo())
            ch.offset = self._jitter(0.0)
            self._humanize_timing(ch, guitar_block_params.get("timing_jitter_ms", self.timing_jitter_ms))
            _attach_artics(ch)
            self._apply_round_robin(ch)
            notes_for_event.append(ch)

        elif execution_style == EXEC_STYLE_STRUM_BASIC:
            event_stroke_dir = guitar_block_params.get(
                "current_event_stroke",
                guitar_block_params.get(
                    "strum_direction_cycle", "down,down,up,up"
                ).split(",")[
                    0
                ],  # サイクルからも取得
            )
            is_down = event_stroke_dir.lower() == "down"  # 小文字比較
            play_order = list(reversed(chord_pitches)) if is_down else chord_pitches
            base_factor = 1.0
            if is_down:
                base_factor = 1.10
            elif event_stroke_dir.lower() == "up":
                base_factor = 0.90
            strum_delay_ql = rhythm_pattern_definition.get(
                "strum_delay_ql",
                guitar_block_params.get("strum_delay_ql", GUITAR_STRUM_DELAY_QL),
            )
            jitter_ms = guitar_block_params.get("strum_delay_jitter_ms", self.strum_delay_jitter_ms)
            strum_delay_ms = strum_delay_ql * 60000.0 / self.global_tempo

            for i, p_obj_strum in enumerate(play_order):
                n_strum = note.Note(p_obj_strum)
                base_dur = event_duration_ql * (
                    0.85 if is_palm_muted else 0.9
                )
                base_dur *= 1 + self.rng.uniform(-self.gate_length_variation, self.gate_length_variation)
                n_strum.duration = music21_duration.Duration(
                    quarterLength=max(MIN_NOTE_DURATION_QL, base_dur)
                )
                delay_ms = i * strum_delay_ms
                if jitter_ms:
                    delay_ms += self.rng.uniform(-jitter_ms / 2.0, jitter_ms / 2.0)
                delay_ql = delay_ms / (60000.0 / self.global_tempo)
                n_strum.offset = self._jitter(delay_ql)
                self._humanize_timing(n_strum, guitar_block_params.get("timing_jitter_ms", self.timing_jitter_ms))
                vel_adj_range = 10
                vel_adj = 0
                if len(play_order) > 1:
                    if is_down:
                        vel_adj = int(
                            (
                                (len(play_order) - 1 - i)
                                / (len(play_order) - 1)
                                * vel_adj_range
                            )
                            - (vel_adj_range / 2)
                        )
                    else:
                        vel_adj = int(
                            ((i / (len(play_order) - 1)) * vel_adj_range)
                            - (vel_adj_range / 2)
                        )
                final_vel = max(1, min(127, int(event_final_velocity * base_factor) + vel_adj))
                n_strum.volume = m21volume.Volume(velocity=final_vel)
                if is_palm_muted:
                    n_strum.articulations.append(articulations.Staccatissimo())
                _attach_artics(n_strum)
                self._apply_round_robin(n_strum)
                notes_for_event.append(n_strum)

        elif execution_style == EXEC_STYLE_ARPEGGIO_FROM_INDICES:
            arp_pattern_indices = rhythm_pattern_definition.get(
                "arpeggio_indices", guitar_block_params.get("arpeggio_indices")
            )
            arp_note_dur_ql = rhythm_pattern_definition.get(
                "note_duration_ql", guitar_block_params.get("note_duration_ql", 0.5)
            )
            ordered_arp_pitches: List[pitch.Pitch] = []
            if isinstance(arp_pattern_indices, list) and chord_pitches:
                ordered_arp_pitches = [
                    chord_pitches[idx % len(chord_pitches)]
                    for idx in arp_pattern_indices
                ]
            else:
                ordered_arp_pitches = chord_pitches

            current_offset_in_event = 0.0
            arp_idx = 0
            while current_offset_in_event < event_duration_ql and ordered_arp_pitches:
                p_play_arp = ordered_arp_pitches[arp_idx % len(ordered_arp_pitches)]
                actual_arp_dur = min(
                    arp_note_dur_ql, event_duration_ql - current_offset_in_event
                )
                if actual_arp_dur < MIN_NOTE_DURATION_QL / 4.0:
                    break
                base_dur = actual_arp_dur * (
                    0.85 if is_palm_muted else 0.95
                )
                base_dur *= 1 + self.rng.uniform(-self.gate_length_variation, self.gate_length_variation)
                n_arp = note.Note(
                    p_play_arp,
                    quarterLength=max(MIN_NOTE_DURATION_QL, base_dur),
                )
                n_arp.volume = m21volume.Volume(velocity=event_final_velocity)
                n_arp.offset = self._jitter(current_offset_in_event)
                self._humanize_timing(n_arp, guitar_block_params.get("timing_jitter_ms", self.timing_jitter_ms))
                if is_palm_muted:
                    n_arp.articulations.append(articulations.Staccatissimo())
                _attach_artics(n_arp)
                self._apply_round_robin(n_arp)
                notes_for_event.append(n_arp)
                current_offset_in_event += arp_note_dur_ql
                arp_idx += 1
        elif execution_style == EXEC_STYLE_ARPEGGIO_PATTERN:
            string_order = rhythm_pattern_definition.get(
                "string_order",
                guitar_block_params.get("string_order", None),
            )
            if not string_order:
                string_order = list(range(len(chord_pitches)))
            spacing_ms = rhythm_pattern_definition.get(
                "arpeggio_note_spacing_ms",
                guitar_block_params.get("arpeggio_note_spacing_ms"),
            )
            strict_so = bool(
                guitar_block_params.get(
                    "strict_string_order",
                    rhythm_pattern_definition.get(
                        "strict_string_order", self.strict_string_order
                    ),
                )
            )
            if spacing_ms is not None:
                spacing_ql = spacing_ms * self.global_tempo / 60000.0
            else:
                spacing_ql = event_duration_ql / max(1, len(string_order))

            expected_count = max(1, int(round(event_duration_ql / spacing_ql)))
            if len(string_order) != expected_count:
                if strict_so:
                    logger.warning(
                        "string_order length %d does not match expected note count %d; adjusting automatically",
                        len(string_order),
                        expected_count,
                    )
                if len(string_order) < expected_count:
                    mul = math.ceil(expected_count / len(string_order))
                    string_order = (string_order * mul)[:expected_count]
                else:
                    string_order = string_order[:expected_count]

            base_dur = event_duration_ql * (0.85 if is_palm_muted else 0.95)
            base_dur *= 1 + self.rng.uniform(-self.gate_length_variation, self.gate_length_variation)

            for idx, s_idx in enumerate(string_order):
                p_sel = chord_pitches[s_idx % len(chord_pitches)]
                dur = min(base_dur, spacing_ql * 0.95)
                n_ap = note.Note(p_sel, quarterLength=max(MIN_NOTE_DURATION_QL, dur))
                n_ap.volume = m21volume.Volume(velocity=event_final_velocity)
                n_ap.offset = self._jitter(idx * spacing_ql)
                self._humanize_timing(n_ap, guitar_block_params.get("timing_jitter_ms", self.timing_jitter_ms))
                if is_palm_muted:
                    n_ap.articulations.append(articulations.Staccatissimo())
                _attach_artics(n_ap)
                self._apply_round_robin(n_ap)
                notes_for_event.append(n_ap)
        elif execution_style == EXEC_STYLE_MUTED_RHYTHM:
            mute_note_dur = rhythm_pattern_definition.get(
                "mute_note_duration_ql",
                guitar_block_params.get("mute_note_duration_ql", 0.1),
            )
            mute_interval = rhythm_pattern_definition.get(
                "mute_interval_ql", guitar_block_params.get("mute_interval_ql", 0.25)
            )
            t_mute = 0.0
            if not chord_pitches:
                return []
            mute_base_pitch = chord_pitches[0]
            while t_mute < event_duration_ql:
                actual_mute_dur = min(mute_note_dur, event_duration_ql - t_mute)
                if actual_mute_dur < MIN_NOTE_DURATION_QL / 8.0:
                    break
                n_mute = note.Note(mute_base_pitch)
                n_mute.articulations = [articulations.Staccatissimo()]
                base_dur = actual_mute_dur
                base_dur *= 1 + self.rng.uniform(-self.gate_length_variation, self.gate_length_variation)
                n_mute.duration.quarterLength = max(MIN_NOTE_DURATION_QL, base_dur)
                n_mute.volume = m21volume.Volume(
                    velocity=int(event_final_velocity * 0.6) + random.randint(-5, 5)
                )
                n_mute.offset = self._jitter(t_mute)
                self._humanize_timing(n_mute, guitar_block_params.get("timing_jitter_ms", self.timing_jitter_ms))
                _attach_artics(n_mute)
                self._apply_round_robin(n_mute)
                notes_for_event.append(n_mute)
                t_mute += mute_interval
        else:
            logger.warning(
                f"GuitarGen: Unknown or unhandled execution_style '{execution_style}' for chord {cs.figure if cs else 'N/A'}. No notes generated for this event."
            )
        # Fingering estimation
        flat_pitches: List[pitch.Pitch] = []
        for el in notes_for_event:
            if isinstance(el, m21chord.Chord):
                flat_pitches.extend(n.pitch for n in el.notes)
            else:
                flat_pitches.append(el.pitch)

        finger_info = self._estimate_fingering(flat_pitches)
        i = 0
        for el in notes_for_event:
            if isinstance(el, m21chord.Chord):
                for n_in in el.notes:
                    if i < len(finger_info):
                        self._attach_fingering(n_in, finger_info[i][0], finger_info[i][1])
                    i += 1
            else:
                if i < len(finger_info):
                    self._attach_fingering(el, finger_info[i][0], finger_info[i][1])
                i += 1

        return notes_for_event

    def _render_part(
        self,
        section_data: Dict[str, Any],
        next_section_data: Optional[Dict[str, Any]] = None,
    ) -> stream.Part:
        guitar_part = stream.Part(id=self.part_name)
        actual_instrument = copy.deepcopy(
            self.default_instrument
        )  # BasePartGeneratorで設定されたものを使用
        if not actual_instrument.partName:
            actual_instrument.partName = self.part_name.capitalize()
        if not actual_instrument.partAbbreviation:
            actual_instrument.partAbbreviation = self.part_name[:3].capitalize() + "."
        guitar_part.insert(0, actual_instrument)

        log_blk_prefix = f"GuitarGen._render_part (Section: {section_data.get('section_name', 'Unknown')}, Chord: {section_data.get('original_chord_label', 'N/A')})"

        # パラメータのマージ (chordmapのpart_params と arrangement_overrides)
        # self.overrides は BasePartGenerator.compose() で設定される PartOverride オブジェクト
        guitar_params_from_chordmap = section_data.get("part_params", {}).get(
            self.part_name, {}
        )
        final_guitar_params = guitar_params_from_chordmap.copy()
        # options のマージも考慮 (BassGenerator参考)
        final_guitar_params.setdefault("options", {})

        if self.overrides and hasattr(self.overrides, "model_dump"):
            override_dict = self.overrides.model_dump(exclude_unset=True)
            if not isinstance(final_guitar_params.get("options"), dict):
                final_guitar_params["options"] = {}  # 念のため初期化

            chordmap_options = final_guitar_params.get("options", {})
            override_options = override_dict.pop("options", None)  # popで取り出し

            if isinstance(override_options, dict):  # override側にoptionsがあればマージ
                merged_options = chordmap_options.copy()
                merged_options.update(override_options)
                final_guitar_params["options"] = merged_options
            # options 以外のキーで上書き
            final_guitar_params.update(override_dict)
        logger.debug(f"{log_blk_prefix}: FinalParams={final_guitar_params}")

        final_guitar_params.setdefault("stroke_direction", self.default_stroke_direction)
        final_guitar_params.setdefault("palm_mute", self.default_palm_mute)

        # 必要な情報を section_data から取得
        block_duration_ql = safe_get(
            section_data,
            "humanized_duration_beats",
            default=safe_get(
                section_data, "q_length", default=self.measure_duration, cast_to=float
            ),
            cast_to=float,
        )
        if block_duration_ql <= 0:
            logger.warning(
                f"{log_blk_prefix}: Non-positive duration {block_duration_ql}. Using measure_duration {self.measure_duration}ql."
            )
            block_duration_ql = self.measure_duration

        chord_label_str = section_data.get(
            "chord_symbol_for_voicing", section_data.get("original_chord_label", "C")
        )
        if chord_label_str.lower() in ["rest", "r", "n.c.", "nc", "none", "-"]:
            logger.info(
                f"{log_blk_prefix}: Block is a Rest. Skipping guitar part for this block."
            )
            return guitar_part  # 空のパートを返す

        sanitized_label = sanitize_chord_label(chord_label_str)
        cs_object: Optional[harmony.ChordSymbol] = None
        if sanitized_label and sanitized_label.lower() != "rest":
            try:
                cs_object = harmony.ChordSymbol(sanitized_label)
                specified_bass_str = section_data.get("specified_bass_for_voicing")
                if specified_bass_str:
                    final_bass_str = sanitize_chord_label(specified_bass_str)
                    if final_bass_str and final_bass_str.lower() != "rest":
                        cs_object.bass(final_bass_str)
                if not cs_object.pitches:
                    cs_object = None
            except Exception as e_parse_guitar:
                logger.warning(
                    f"{log_blk_prefix}: Error parsing chord '{sanitized_label}': {e_parse_guitar}."
                )
                cs_object = None
        if cs_object is None:
            logger.warning(
                f"{log_blk_prefix}: Could not create ChordSymbol for '{chord_label_str}'. Skipping block."
            )
            return guitar_part

        # リズムキーの選択
        current_musical_intent = section_data.get("musical_intent", {})
        emotion = current_musical_intent.get("emotion")
        intensity = current_musical_intent.get("intensity")
        # final_guitar_params から cli_override に相当するものを取得 (必要なら)
        # ここではひとまず cli_guitar_style_override は None とする (BasePartGenerator.compose から渡されないため)
        cli_guitar_style_override = final_guitar_params.get("cli_guitar_style_override")

        param_rhythm_key = final_guitar_params.get(
            "guitar_rhythm_key", final_guitar_params.get("rhythm_key")
        )
        final_rhythm_key_selected = self.style_selector.select(
            emotion=emotion,
            intensity=intensity,
            cli_override=cli_guitar_style_override,  # modular_composer.py の args.guitar_style を渡せるようにする想定
            part_params_override_rhythm_key=param_rhythm_key,
            rhythm_library_keys=list(
                self.part_parameters.keys()
            ),  # self.rhythm_lib -> self.part_parameters
        )
        logger.info(
            f"{log_blk_prefix}: Selected rhythm_key='{final_rhythm_key_selected}' for guitar."
        )

        rhythm_details = self.part_parameters.get(
            final_rhythm_key_selected
        )  # self.rhythm_lib -> self.part_parameters
        if not rhythm_details:
            logger.warning(
                f"{log_blk_prefix}: Rhythm key '{final_rhythm_key_selected}' not found. Using default."
            )
            rhythm_details = self.part_parameters.get(DEFAULT_GUITAR_RHYTHM_KEY)
            if not rhythm_details:
                logger.error(
                    f"{log_blk_prefix}: CRITICAL - Default guitar rhythm missing. Using minimal block."
                )
            rhythm_details = {
                "execution_style": EXEC_STYLE_BLOCK_CHORD,
                "pattern": [
                    {
                        "offset": 0,
                        "duration": block_duration_ql,
                        "velocity_factor": 0.7,
                    }
                ],
                "reference_duration_ql": block_duration_ql,
            }

        pattern_type_global = rhythm_details.get("pattern_type", "strum")
        pattern_events = rhythm_details.get("pattern", [])
        if pattern_events is None:
            pattern_events = []

        options = rhythm_details.get("options", {})
        velocity_curve_spec = options.get("velocity_curve")
        if velocity_curve_spec is None:
            velocity_curve_spec = rhythm_details.get("velocity_curve_name")

        if velocity_curve_spec is None and self.default_velocity_curve is not None:
            velocity_curve_list = self.default_velocity_curve
        else:
            velocity_curve_spec = velocity_curve_spec or ""
            velocity_curve_list = self._select_velocity_curve(velocity_curve_spec)

        pattern_ref_duration = rhythm_details.get(
            "reference_duration_ql", self.measure_duration
        )
        if pattern_ref_duration <= 0:
            pattern_ref_duration = self.measure_duration

        # Strum cycle の準備 (パッチ参考)
        strum_cycle_str = final_guitar_params.get(
            "strum_direction_cycle",
            rhythm_details.get("strum_direction_cycle", "D,D,U,U"),
        )
        strum_cycle_list = [s.strip().upper() for s in strum_cycle_str.split(",")]
        current_strum_idx = 0

        for event_idx, event_def in enumerate(pattern_events):
            log_event_prefix = f"{log_blk_prefix}.Event{event_idx}"
            event_offset_in_pattern = safe_get(
                event_def,
                "offset",
                default=0.0,
                cast_to=float,
                log_name=f"{log_event_prefix}.Offset",
            )
            event_duration_in_pattern = safe_get(
                event_def,
                "duration",
                default=1.0,
                cast_to=float,
                log_name=f"{log_event_prefix}.Dur",
            )
            if event_duration_in_pattern <= 0:
                logger.warning(
                    f"{log_event_prefix}: Invalid duration {event_duration_in_pattern}. Using 1.0."
                )
                event_duration_in_pattern = 1.0

            if "velocity_factor" in event_def:
                event_velocity_factor = safe_get(
                    event_def,
                    "velocity_factor",
                    default=1.0,
                    cast_to=float,
                    log_name=f"{log_event_prefix}.VelFactor",
                )
            else:
                event_velocity_factor = None

            current_event_guitar_params = (
                final_guitar_params.copy()
            )  # イベント固有のパラメータ用
            # パターンイベントにstrum_directionがあればそれを優先、なければサイクルから
            event_stroke_direction = event_def.get("strum_direction")
            if not event_stroke_direction and strum_cycle_list:
                event_stroke_direction = strum_cycle_list[
                    current_strum_idx % len(strum_cycle_list)
                ]
                current_strum_idx += 1
            if event_stroke_direction:
                current_event_guitar_params["current_event_stroke"] = (
                    event_stroke_direction
                )
            else:
                if final_guitar_params.get("stroke_direction"):
                    current_event_guitar_params["current_event_stroke"] = final_guitar_params.get("stroke_direction")

            scale_factor = (
                block_duration_ql / pattern_ref_duration
                if pattern_ref_duration > 0
                else 1.0
            )
            # このイベントのブロック内での開始オフセット (絶対ではない)
            current_event_start_offset_in_block = event_offset_in_pattern * scale_factor
            # このイベントのスケールされたデュレーション
            actual_event_dur_scaled = event_duration_in_pattern * scale_factor

            # ブロック境界チェック
            if current_event_start_offset_in_block >= block_duration_ql - (
                MIN_NOTE_DURATION_QL / 16.0
            ):
                continue  # イベントがブロックのほぼ最後か外で始まる

            max_possible_event_dur_from_here = (
                block_duration_ql - current_event_start_offset_in_block
            )
            final_actual_event_dur_for_create = min(
                actual_event_dur_scaled, max_possible_event_dur_from_here
            )

            if final_actual_event_dur_for_create < MIN_NOTE_DURATION_QL / 2.0:
                logger.debug(
                    f"{log_event_prefix}: Skipping very short event (dur: {final_actual_event_dur_for_create:.3f} ql)"
                )
                continue

            # ベロシティの決定
            block_base_velocity_candidate = current_event_guitar_params.get(
                "velocity"
            )  # マージ済みパラメータから
            if block_base_velocity_candidate is None:
                block_base_velocity_candidate = rhythm_details.get("velocity_base", 70)
            if block_base_velocity_candidate is None:
                block_base_velocity_candidate = section_data.get(
                    "emotion_params", {}
                ).get(
                    "humanized_velocity", 70
                )  # humanizerからの値も考慮
            try:
                block_base_velocity = int(block_base_velocity_candidate)
            except (TypeError, ValueError):
                block_base_velocity = 70

            if event_velocity_factor is None and velocity_curve_list:
                vel_from_curve = velocity_curve_list[event_idx % len(velocity_curve_list)]
                final_event_velocity = int(vel_from_curve)
            else:
                ev_factor = float(event_velocity_factor if event_velocity_factor is not None else 1.0)
                final_event_velocity = int(block_base_velocity * ev_factor)
            layer_idx = event_def.get("velocity_layer")
            if velocity_curve_list and layer_idx is not None:
                try:
                    idx = int(layer_idx)
                    if 0 <= idx < len(velocity_curve_list):
                        final_event_velocity = int(final_event_velocity * velocity_curve_list[idx])
                except (TypeError, ValueError):
                    pass
            beat_idx = int(math.floor(current_event_start_offset_in_block))
            if beat_idx in self.accent_map:
                final_event_velocity += int(self.accent_map[beat_idx])
            final_event_velocity = max(1, min(127, final_event_velocity))

            # Palm Mute 判定 (パッチ参考)
            # final_guitar_params に palm_mute があればそれを使い、なければリズム定義から、それもなければFalse
            current_event_guitar_params["palm_mute"] = final_guitar_params.get(
                "palm_mute", rhythm_details.get("palm_mute", False)
            )
            current_event_guitar_params["palm_mute"] = bool(
                event_def.get(
                    "palm_mute",
                    current_event_guitar_params.get("palm_mute", False),
                )
            )

            event_articulation = event_def.get("articulation")
            if event_articulation is not None:
                current_event_guitar_params["articulation"] = event_articulation
                if event_articulation == "palm_mute":
                    current_event_guitar_params["palm_mute"] = True

            this_ptype = event_def.get("pattern_type", pattern_type_global)
            event_rhythm = rhythm_details.copy()
            event_rhythm.update(event_def)
            if this_ptype == "arpeggio":
                event_rhythm.setdefault("execution_style", EXEC_STYLE_ARPEGGIO_PATTERN)
            generated_elements = self._create_notes_from_event(
                cs_object,
                event_rhythm,
                current_event_guitar_params,
                final_actual_event_dur_for_create,
                final_event_velocity,
                current_event_start_offset_in_block,
            )

            exec_style = event_rhythm.get("execution_style", EXEC_STYLE_BLOCK_CHORD)

            for el in generated_elements:
                # el.offset は _create_notes_from_event 内でイベント開始からの相対オフセットになっている
                # これに、このリズムイベントのブロック内での開始オフセットを加算
                el.offset += current_event_start_offset_in_block

                pitch_for_check: pitch.Pitch | None = None
                if isinstance(el, note.Note):
                    pitch_for_check = el.pitch
                elif isinstance(el, m21chord.Chord) and el.pitches:
                    pitch_for_check = el.pitches[0]

                prev_pitch = self._prev_note_pitch
                if pitch_for_check and prev_pitch:
                    semitone_diff = pitch_for_check.ps - prev_pitch.ps
                    if exec_style == EXEC_STYLE_HAMMER_ON and semitone_diff > 0:
                        if 0 < semitone_diff <= self.part_parameters.get(
                            "hammer_on_interval", self.hammer_on_interval
                        ):
                            if self.rng.random() < self.part_parameters.get(
                                "hammer_on_probability", self.hammer_on_probability
                            ):
                                art = articulations.HammerOn()
                                if isinstance(el, m21chord.Chord):
                                    for n_in_ch in el.notes:
                                        n_in_ch.articulations.append(
                                            copy.deepcopy(art)
                                        )
                                else:
                                    el.articulations.append(copy.deepcopy(art))
                    elif exec_style == EXEC_STYLE_PULL_OFF and semitone_diff < 0:
                        if 0 < abs(semitone_diff) <= self.part_parameters.get(
                            "pull_off_interval", self.pull_off_interval
                        ):
                            if self.rng.random() < self.part_parameters.get(
                                "pull_off_probability", self.pull_off_probability
                            ):
                                art = articulations.PullOff()
                                if isinstance(el, m21chord.Chord):
                                    for n_in_ch in el.notes:
                                        n_in_ch.articulations.append(
                                            copy.deepcopy(art)
                                        )
                                else:
                                    el.articulations.append(copy.deepcopy(art))
                if pitch_for_check:
                    self._prev_note_pitch = pitch_for_check

                guitar_part.insert(el.offset, el)  # パート内でのオフセットで挿入

        logger.info(
            f"{log_blk_prefix}: Finished processing. Part has {len(list(guitar_part.flatten().notesAndRests))} elements before groove/humanize."
        )

        profile_name = (
            self.cfg.get("humanize_profile")
            or section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        if profile_name:
            humanizer.apply(guitar_part, profile_name)

        return guitar_part

    def export_musicxml(self, path: str) -> None:
        if not hasattr(self, "_last_part") or self._last_part is None:
            raise ValueError("No generated part available for export")
        score = stream.Score()
        score.insert(0, self._last_part)
        score.write("musicxml", fp=path)

    def export_tab(self, path: str, format: str = "xml") -> None:
        """Export the last generated guitar part as tablature.

        Parameters
        ----------
        path:
            Destination file path.
        format:
            Either ``"xml"`` for MusicXML output or ``"ascii"`` for a text
            representation. Defaults to ``"xml"``.
        """

        if not hasattr(self, "_last_part") or self._last_part is None:
            raise RuntimeError("No part generated yet")

        if format == "xml":
            try:
                from music21 import tab  # type: ignore
                TabContainer = getattr(tab, "TabStaff", None) or getattr(tab, "TabStream", None)
            except Exception:
                TabContainer = None

            try:
                if TabContainer is not None:
                    tab_stream = TabContainer()
                    tab_stream.append(self._last_part.flat)
                    score = stream.Score()
                    score.insert(0, tab_stream)
                else:
                    score = stream.Score()
                    score.insert(0, self._last_part)
                score.write("musicxml", fp=path)
                return
            except Exception:
                # Fall back to ASCII if XML export fails
                format = "ascii"

        if format == "ascii":
            with open(path, "w", encoding="utf-8") as f:
                for el in self._last_part.flatten().notes:
                    if hasattr(el, "pitch"):
                        name = el.pitch.nameWithOctave
                    else:
                        name = "+".join(p.nameWithOctave for p in el.pitches)
                    f.write(f"{name}\t{el.duration.quarterLength}\n")
            return

        if format not in {"xml", "ascii"}:
            raise ValueError(f"Unsupported format: {format}")

    def export_tab_enhanced(self, path: str) -> None:
        """Export simplified tablature with string and fret information."""
        if not hasattr(self, "_last_part") or self._last_part is None:
            raise RuntimeError("No part generated yet")

        with open(path, "w", encoding="utf-8") as f:
            for el in self._last_part.flatten().notes:
                if isinstance(el, m21chord.Chord):
                    tokens = []
                    for n in el.notes:
                        s = getattr(n, "string", None)
                        fr = getattr(n, "fret", None)
                        tokens.append(f"{s}|{fr}" if s is not None and fr is not None else "x|-")
                    if not tokens:
                        continue
                    line = " ".join(tokens)
                else:
                    s = getattr(el, "string", None)
                    fr = getattr(el, "fret", None)
                    line = f"{s}|{fr}" if s is not None and fr is not None else "x|-"
                if line.strip():
                    f.write(line + "\n")

    def export_musicxml_tab(self, path: str) -> None:
        """Export the last generated part with string/fret info as MusicXML."""
        if not hasattr(self, "_last_part") or self._last_part is None:
            raise RuntimeError("No part generated yet")

        flat_part = stream.Part()
        StringCls = _get_string_indicator_cls()
        FretCls = _get_fret_indicator_cls()
        for el in self._last_part.recurse():
            if isinstance(el, m21chord.Chord):
                for n_in in el.notes:
                    n_new = note.Note(n_in.pitch, quarterLength=el.quarterLength)
                    n_new.offset = el.offset
                    n_new.volume = copy.deepcopy(el.volume)
                    s = getattr(n_in, "string", None)
                    f = getattr(n_in, "fret", None)
                    if s is not None and f is not None:
                        setattr(n_new, "string", s)
                        setattr(n_new, "fret", f)
                        if StringCls and FretCls:
                            try:
                                if not hasattr(n_new, 'notations'):
                                    n_new.notations = note.Notations()
                                n_new.notations.append(StringCls(number=int(s) + 1))
                                n_new.notations.append(FretCls(number=int(f)))
                            except Exception:
                                pass
                    flat_part.insert(n_new.offset, n_new)
            elif isinstance(el, note.Note):
                n_new = copy.deepcopy(el)
                s = getattr(n_new, "string", None)
                f = getattr(n_new, "fret", None)
                if s is not None and f is not None:
                    if StringCls and FretCls:
                        try:
                            if not hasattr(n_new, 'notations'):
                                n_new.notations = note.Notations()
                            n_new.notations.append(StringCls(number=int(s) + 1))
                            n_new.notations.append(FretCls(number=int(f)))
                        except Exception:
                            pass
                
                flat_part.insert(n_new.offset, n_new)
        score = stream.Score()
        score.insert(0, flat_part)
        score.write("musicxml", fp=path)
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(path)
            root = tree.getroot()
            notes_xml = root.findall('.//{*}note')
            notes_m21 = list(flat_part.recurse().notes)
            for xn, mn in zip(notes_xml, notes_m21):
                s = getattr(mn, 'string', None)
                f = getattr(mn, 'fret', None)
                if s is None or f is None:
                    continue
                not_elem = xn.find('./{*}notations')
                if not_elem is None:
                    not_elem = ET.SubElement(xn, 'notations')
                tech = ET.SubElement(not_elem, 'technical')
                ET.SubElement(tech, 'string').text = str(int(s) + 1)
                ET.SubElement(tech, 'fret').text = str(int(f))
            tree.write(path, encoding='utf-8')
        except Exception as e:
            logger.warning(f"manual xml tablature failed: {e}")

    def _load_external_strum_patterns(self) -> None:
        """Load additional strum patterns from an external YAML or JSON file."""
        if not self.external_patterns_path:
            return
        path = Path(self.external_patterns_path)
        if not path.exists():
            return
        try:
            text = path.read_text(encoding="utf-8")
            data: dict | None = None
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(text)
            elif path.suffix.lower() == ".json":
                data = json.loads(text)
            else:
                try:
                    data = yaml.safe_load(text)
                except Exception:
                    data = json.loads(text)
            if isinstance(data, dict):
                self.part_parameters.update(data)
        except Exception as e:
            logger.warning(f"Failed to load external strum patterns: {e}")

    def _add_internal_default_patterns(self):
        """Add basic fallback strum patterns if they are missing."""
        quarter_pattern = []
        for i in range(4):
            evt = {
                "offset": float(i),
                "duration": 1.0,
                "velocity_factor": 0.8,
                "type": "block",
            }
            if i in {1, 3}:
                evt["articulation"] = "palm_mute"
            quarter_pattern.append(evt)

        syncopation_pattern = [
            {
                "offset": 0.0,
                "duration": 1.0,
                "velocity_factor": 0.9,
                "type": "block",
                "articulation": "accent",
            },
            {
                "offset": 1.5,
                "duration": 0.5,
                "velocity_factor": 0.9,
                "type": "block",
                "articulation": "staccato",
            },
            {
                "offset": 3.0,
                "duration": 1.0,
                "velocity_factor": 0.9,
                "type": "block",
            },
        ]

        shuffle_pattern: List[Dict[str, Union[float, str]]] = []
        first_len = 2.0 / 3.0
        second_len = 1.0 / 3.0
        current = 0.0
        for _ in range(4):
            shuffle_pattern.append(
                {
                    "offset": round(current, 6),
                    "duration": first_len,
                    "velocity_factor": 0.8,
                    "type": "block",
                }
            )
            current += first_len
            shuffle_pattern.append(
                {
                    "offset": round(current, 6),
                    "duration": second_len,
                    "velocity_factor": 0.8,
                    "type": "block",
                    "articulation": "staccato",
                }
            )
            current += second_len

        defaults = {
            "guitar_rhythm_quarter": {
                "pattern": quarter_pattern,
                "reference_duration_ql": 1.0,
                "description": "Simple quarter-note strum",
            },
            "guitar_arpeggio_basic": {
                "pattern_type": "arpeggio",
                "string_order": [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
                "reference_duration_ql": 1.0,
                "description": "Ascending then descending arpeggio",
            },
            "guitar_rhythm_syncopation": {
                "pattern": syncopation_pattern,
                "reference_duration_ql": 1.0,
                "description": "Syncopated strum pattern",
            },
            "guitar_rhythm_shuffle": {
                "pattern": shuffle_pattern,
                "reference_duration_ql": 1.0,
                "description": "Shuffle feel eighth-note strum",
            },
        }

        for key, val in defaults.items():
            if key not in self.part_parameters:
                self.part_parameters[key] = val



# --- END OF FILE generator/guitar_generator.py ---
