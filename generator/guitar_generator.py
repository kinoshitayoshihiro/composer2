# --- START OF FILE generator/guitar_generator.py (BasePartGenerator継承・改修版) ---
import copy
import json
import logging
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import music21.articulations as articulations
import music21.chord as m21chord
import music21.duration as music21_duration
import music21.harmony as harmony
import music21.interval as interval
import music21.meter as meter
import music21.note as note
import music21.pitch as pitch
import music21.stream as stream
import music21.volume as m21volume
import yaml
import statistics

from utilities.velocity_curve import interpolate_7pt, resolve_velocity_curve
from utilities.tone_shaper import ToneShaper

logger = logging.getLogger(__name__)

if not hasattr(articulations, "Shake"):
    class Shake(articulations.Articulation):
        pass

    articulations.Shake = Shake

if not hasattr(articulations, "PinchHarmonic"):
    class PinchHarmonic(articulations.Articulation):
        pass

    articulations.PinchHarmonic = PinchHarmonic


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

from utilities import humanizer

from .base_part_generator import BasePartGenerator

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

    def get_time_signature_object(ts_str: str | None) -> meter.TimeSignature:
        if not ts_str:
            ts_str = "4/4"
        try:
            return meter.TimeSignature(ts_str)
        except Exception:
            return meter.TimeSignature("4/4")

    def sanitize_chord_label(label: str | None) -> str | None:
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



DEFAULT_GUITAR_OCTAVE_RANGE: tuple[int, int] = (2, 5)
GUITAR_STRUM_DELAY_QL: float = 0.02
STANDARD_TUNING_OFFSETS = [0, 0, 0, 0, 0, 0]

# User-friendly tuning presets (semitone offsets per string)
TUNING_PRESETS: dict[str, list[int]] = {
    "standard": STANDARD_TUNING_OFFSETS,
    "drop_d": [-2, 0, 0, 0, 0, 0],
    "open_g": [-2, -2, 0, 0, 0, -2],
}

# Stroke direction to velocity multiplier mapping
_DEFAULT_STROKE_VELOCITY_FACTOR = {
    "DOWN": 1.10,
    "D": 1.10,
    "UP": 0.90,
    "U": 0.90,
}
STROKE_VELOCITY_FACTOR = _DEFAULT_STROKE_VELOCITY_FACTOR

STROKE_VELOCITY_FACTOR = _DEFAULT_STROKE_VELOCITY_FACTOR.copy()

@dataclass
class FingeringCost:
    open_bonus: int = -1
    string_shift: int = 2
    fret_shift: int = 1
    position_shift: int = 3


def _clamp_velocity(value: float) -> int:
    """Clamp *value* into MIDI velocity range."""
    return max(1, min(127, int(round(value))))  # noqa: PLR2004


def _normalize_stroke_key(stroke: str | None) -> str | None:
    return stroke.strip().upper() if isinstance(stroke, str) else None

EXEC_STYLE_BLOCK_CHORD = "block_chord"
EXEC_STYLE_STRUM_BASIC = "strum_basic"
EXEC_STYLE_ARPEGGIO_FROM_INDICES = "arpeggio_from_indices"
EXEC_STYLE_ARPEGGIO_PATTERN = "arpeggio_pattern"
EXEC_STYLE_POWER_CHORDS = "power_chords"
EXEC_STYLE_MUTED_RHYTHM = "muted_rhythm"
EXEC_STYLE_HAMMER_ON = "hammer_on"
EXEC_STYLE_PULL_OFF = "pull_off"

EMOTION_INTENSITY_MAP: dict[tuple[str, str], str] = {
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
    def __init__(self, mapping: dict[tuple[str, str], str] | None = None):
        self.mapping = mapping if mapping is not None else EMOTION_INTENSITY_MAP

    def select(
        self,
        *,
        emotion: str | None,
        intensity: str | None,
        cli_override: str | None = None,
        part_params_override_rhythm_key: str | None = None,
        rhythm_library_keys: list[str],
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
        tuning: str | Sequence[int] | None = None,
        timing_variation: float = 0.0,
        gate_length_variation: float = 0.0,
        external_patterns_path: str | None = None,
        hammer_on_interval: int = 2,
        pull_off_interval: int = 2,
        hammer_on_probability: float = 0.5,
        pull_off_probability: float = 0.5,
        default_stroke_direction: str | None = "down",
        default_palm_mute: bool = False,
        default_velocity_curve: str | dict | None = None,
        timing_jitter_ms: float = 0.0,
        timing_jitter_mode: str = "uniform",
        strum_delay_jitter_ms: float = 0.0,
        swing_ratio: float | None = None,
        velocity_preset_path: str | None = None,
        style_db_path: str | None = None,
        accent_map: dict[int, int] | None = None,
        rr_channel_cycle: Sequence[int] | None = None,
        swing_subdiv: int | None = None,
        velocity_curve_interp_mode: str = "spline",
        stroke_velocity_factor: dict[str, float] | None = None,
        position_lock: bool = False,
        preferred_position: int = 0,
        open_string_bonus: int = -1,
        string_shift_weight: int = 2,
        fret_shift_weight: int = 1,
        strict_string_order: bool = False,
        fingering_costs: dict[str, int] | None = None,
        amp_preset_file: str | Path | None = None,
        tone_shaper: ToneShaper | None = None,
        **kwargs,
    ):
        """Create a guitar part generator.

        Parameters
        ----------
        fingering_costs:
            Optional custom fingering cost values. May be a ``dict`` or
            :class:`FingeringCost` instance.
        velocity_curve_interp_mode:
            Interpolation mode for seven-point velocity curves. Either
            ``"linear"`` or ``"spline"``.
        """

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
        self.velocity_curve_interp_mode = str(velocity_curve_interp_mode).lower()
        base_sv = {k.upper(): float(v) for k, v in STROKE_VELOCITY_FACTOR.items()}
        if stroke_velocity_factor:
            for k, v in stroke_velocity_factor.items():
                base_sv[k.upper()] = float(v)
        STROKE_VELOCITY_FACTOR.clear()
        STROKE_VELOCITY_FACTOR.update(base_sv)
        self.stroke_velocity_factor = base_sv
        self.velocity_preset_path = velocity_preset_path
        self.velocity_presets: dict[str, dict[str, Any]] = {}
        if style_db_path:
            try:
                from utilities import style_db
                style_db.load_style_db(style_db_path)
            except Exception:
                pass
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
        base_fc = FingeringCost(
            open_bonus=open_string_bonus,
            string_shift=string_shift_weight,
            fret_shift=fret_shift_weight,
        )
        if fingering_costs:
            if isinstance(fingering_costs, FingeringCost):
                fc_data = fingering_costs.__dict__
            else:
                fc_data = fingering_costs
            valid = set(base_fc.__dataclass_fields__.keys())
            for k in fc_data:
                if k not in valid:
                    raise ValueError(f"Unknown fingering_cost field: {k}")
            for k, v in fc_data.items():
                setattr(base_fc, k, int(v))
        self._fingering_costs = base_fc
        self.open_string_bonus = self.fingering_costs.open_bonus
        self.string_shift_weight = self.fingering_costs.string_shift
        self.fret_shift_weight = self.fingering_costs.fret_shift
        self.position_shift_weight = self.fingering_costs.position_shift
        self.strict_string_order = bool(strict_string_order)
        from utilities.core_music_utils import get_time_signature_object

        ts_obj = get_time_signature_object(self.global_time_signature)
        self.measure_duration = (
            ts_obj.barDuration.quarterLength if ts_obj else 4.0
        )
        self.cfg: dict = kwargs.copy()
        self.style_selector = GuitarStyleSelector()
        if tone_shaper is not None:
            self.tone_shaper = tone_shaper
        else:
            if amp_preset_file is None:
                amp_preset_file = Path("data/amp_presets.yml")
            try:
                self.tone_shaper = ToneShaper.from_yaml(amp_preset_file)
            except Exception:
                self.tone_shaper = ToneShaper()
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

        self.stroke_velocity_factor = STROKE_VELOCITY_FACTOR

    @property
    def fingering_costs(self) -> FingeringCost:
        """Return current fingering cost configuration."""
        return self._fingering_costs


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

        def post(part: stream.Part) -> None:
            self._post_process_generated_part(part, section, ratio_to_apply)

        if isinstance(result, stream.Part):
            self._last_part = result
        # ------------------------------------------------------------------
        # 生成済みパートのポストプロセス
        # ------------------------------------------------------------------
        def _post_process_one(part: stream.Part) -> None:
            # 1) スウィング
            if ratio_to_apply is not None:
                self._apply_swing_internal(part, float(ratio_to_apply), self.swing_subdiv)

            # 2) ToneShaper（平均 Velocity × intensity で CC31 挿入）
            self._auto_tone_shape(part, section.get("musical_intent", {}).get("intensity"))

            # 3) フレーズ強弱／エンベロープ／スタイルカーブ
            marks = section.get("phrase_marks")
            if marks:
                self._apply_phrase_dynamics(part, marks)

            env = section.get("envelope_map")
            if env:
                self._apply_envelope(part, env)

            hint = section.get("style_hint")
            if hint:
                self._apply_style_curve(part, hint)

            # 4) ランダムウォーク CC
            rw = section.get("random_walk_cc")
            if rw:
                if isinstance(rw, dict):
                    rng = random.Random(int(rw["seed"])) if "seed" in rw else None
                    self._apply_random_walk_cc(
                        part,
                        cc=int(rw.get("cc", 1)),
                        step=int(rw.get("step", 2)),
                        rng=rng,
                    )
                else:
                    self._apply_random_walk_cc(part)

            # 5) エフェクト CC
            fx_params = section.get("fx_params")
            if fx_params:
                self._apply_fx_cc(part, fx_params, section.get("musical_intent", {}))

        # ------------------------------------------------------------------
        # 共通ポストプロセス（Swing・ダイナミクス・FX・Ampプリセット 等）
        # ------------------------------------------------------------------
        def _post_process_one(p: stream.Part) -> None:
            # ── Swing 適用 ──────────────────────────────
            if ratio_to_apply is not None:
                self._apply_swing_internal(p, float(ratio_to_apply), self.swing_subdiv)

            # ── フレーズ・エンベロープ／ダイナミクス ────────────────
            marks = section.get("phrase_marks")
            env   = section.get("envelope_map")
            if marks:
                self._apply_phrase_dynamics(p, marks)
            if env:
                self._apply_envelope(p, env)

            # ── スタイルカーブ & ランダムウォーク CC ───────────────
            hint = section.get("style_hint")
            if hint:
                self._apply_style_curve(p, hint)

            rw = section.get("random_walk_cc")
            if rw:
                if isinstance(rw, dict):
                    rng = random.Random(int(rw.get("seed"))) if "seed" in rw else None
                    self._apply_random_walk_cc(
                        p,
                        cc   = int(rw.get("cc",   1)),
                        step = int(rw.get("step", 2)),
                        rng  = rng,
                    )
                else:
                    self._apply_random_walk_cc(p)

            # ── FX CC 反映 ────────────────────────────────
            fx_params = section.get("fx_params")
            if fx_params:
                self._apply_fx_cc(p, fx_params, section.get("musical_intent", {}))

            # ── Amp / Cab プリセット & IR ファイル登録 ───────────
            notes   = list(p.flatten().notes)
            avg_vel = statistics.mean(n.volume.velocity or 64 for n in notes) if notes else 64.0
            part_cfg = section.get("part_params", {}).get(self.part_name, {})
            chosen = self.tone_shaper.choose_preset(
                part_cfg.get("amp_preset"),
                part_cfg.get("fx_preset_intensity")
                or section.get("intensity"),
                avg_vel,
            )

            if not hasattr(p, "extra_cc"):
                p.extra_cc = []
            events = self.tone_shaper.to_cc_events(
                chosen,
                part_cfg.get("fx_preset_intensity", "med"),
                as_dict=False,
            )
            p.extra_cc.extend({"time": t, "cc": c, "val": v} for t, c, v in events)
            from music21 import metadata as m21metadata
            if p.metadata is None:
                p.metadata = m21metadata.Metadata()
            setattr(p.metadata, "ir_file", self.tone_shaper.ir_map.get(chosen))
            setattr(p.metadata, "extra_cc", events)

        # ------------------------------------------------------------------
        # 単一 Part か dict かで処理分岐
        # ------------------------------------------------------------------
        if isinstance(result, stream.Part):
            self._last_part = result
            _post_process_one(self._last_part)

        elif isinstance(result, dict) and result:
            # dict なら各 Part に同じ後処理を適用
            self._last_part = next(iter(result.values()))
            for p in result.values():
                _post_process_one(p)
        else:
            self._last_part = None
        self.swing_subdiv = orig_subdiv
        return result

    def _post_process_generated_part(
        self, part: stream.Part, section: dict[str, Any], ratio: float | None
    ) -> None:
        """Apply post-generation tweaks to *part*."""
        if ratio is not None:
            self._apply_swing_internal(part, float(ratio), self.swing_subdiv)
        intensity = section.get("musical_intent", {}).get("intensity", "medium")
        self._auto_tone_shape(part, intensity)
        for name in (
            "_apply_phrase_dynamics",
            "_apply_envelope",
            "_apply_style_curve",
            "_apply_random_walk_cc",
            "_apply_fx_cc",
        ):
            func = getattr(self, name, None)
            if callable(func):
                try:
                    func(part)  # type: ignore[misc]
                except Exception:  # pragma: no cover - best effort
                    pass

    def _get_guitar_friendly_voicing(
        self,
        cs: harmony.ChordSymbol,
        num_strings: int = 6,
        preferred_octave_bottom: int = 2,
    ) -> list[pitch.Pitch]:
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
        selected_dict: dict[str, pitch.Pitch] = {}
        for p_cand_select in candidate_pitches:
            if guitar_min_ps <= p_cand_select.ps <= guitar_max_ps:
                if p_cand_select.name not in selected_dict:
                    selected_dict[p_cand_select.name] = p_cand_select
        final_voiced_pitches = sorted(
            list(selected_dict.values()), key=lambda p_sort: p_sort.ps
        )
        return self._apply_tuning(final_voiced_pitches[:num_strings])

    def _apply_tuning(self, pitches: list[pitch.Pitch]) -> list[pitch.Pitch]:
        tuned = []
        for i, p in enumerate(pitches):
            offset = self.tuning[i % len(self.tuning)]
            tuned.append(p.transpose(offset))
        return tuned

    def _resolve_curve(self, spec: Any) -> list[int]:
        curve = resolve_velocity_curve(spec)
        if not curve:
            return []
        if all(0.0 <= v <= 1.5 for v in curve):
            return [int(127 * v) for v in curve]
        return [int(v) for v in curve]

    def _prepare_velocity_map(self, spec: Any) -> list[int] | None:
        if isinstance(spec, dict):
            key = tuple(sorted(spec.items()))
        elif isinstance(spec, list):
            key = tuple(spec)
        else:
            key = spec
        return self.__prepare_velocity_map_cached(key)

    @lru_cache(maxsize=32)
    def __prepare_velocity_map_cached(self, spec: Any) -> list[int] | None:
        curve = self._resolve_curve(spec)
        if not curve:
            return None
        if len(curve) == 3:
            c0, c1, c2 = curve
            curve = [
                c0 + (c1 - c0) * (i / 3) if i <= 3 else c1 + (c2 - c1) * ((i - 3) / 3)
                for i in range(7)
            ]
        if len(curve) == 7:
            curve = interpolate_7pt(curve, mode=self.velocity_curve_interp_mode)
        if len(curve) == 128:
            return [max(0, min(127, int(round(v)))) for v in curve]
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
            with open(path, encoding="utf-8") as f:
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
            m: dict[str, Any] = {}
            for style, curve in presets.items():
                if isinstance(curve, list) and len(curve) in (7, 128):
                    m[style] = curve
                else:
                    logger.warning("Invalid curve for %s/%s", tun, style)
            if m:
                self.velocity_presets[tun] = m

    def _select_velocity_curve(self, style_name: str | None) -> list[int]:
        """Return velocity curve list for *style_name* or fallback."""
        curve_spec: Any = None
        presets = self.velocity_presets.get(self.tuning_name) or {}

        if style_name is None:
            base = getattr(self, "default_velocity_curve", None)
            if base:
                return list(base)
        elif style_name in presets:
            curve_spec = presets[style_name]
        elif "default" in presets:
            curve_spec = presets["default"]
        else:
            curve_spec = style_name

        curve = self._prepare_velocity_map(curve_spec)
        if curve is None:
            base = getattr(self, "default_velocity_curve", None)
            if base:
                return list(base)
            curve = [
                max(0, min(127, int(round(40 + 45 * math.sin((math.pi / 2) * i / 127)))))
                for i in range(128)
            ]
        return curve

    def _apply_round_robin(self, el: note.Note | m21chord.Chord) -> None:
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

    def _apply_stroke_velocity(self, base_vel: float, stroke: str | None) -> int:
        """Apply stroke velocity factor and clamp to MIDI range."""
        key = _normalize_stroke_key(stroke)
        factor = self.stroke_velocity_factor.get(key, 1.0) if key else 1.0
        val = int(base_vel * factor)
        return _clamp_velocity(val)

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
        self, pitches: list[pitch.Pitch]
    ) -> list[tuple[int, int]]:
        """Estimate guitar fingering (string index, fret) for a sequence of pitches."""
        # Standard tuning MIDI numbers
        base_midis = [40, 45, 50, 55, 59, 64]
        tuned = [m + off for m, off in zip(base_midis, self.tuning)]

        candidates: list[list[tuple[int, int]]] = []
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

        dp: list[dict[tuple[int, int], tuple[float, tuple[int, int] | None]]] = []
        for i, opts in enumerate(candidates):
            layer: dict[tuple[int, int], tuple[float, tuple[int, int] | None]] = {}
            if i == 0:
                for o in opts:
                    cost = self.open_string_bonus if o[1] == 0 else 0
                    layer[o] = (cost, None)
            else:
                prev_layer = dp[i - 1]
                for o in opts:
                    best_cost = float("inf")
                    best_prev: tuple[int, int] | None = None
                    for prev_o, (prev_cost, _) in prev_layer.items():
                        cost = (
                            prev_cost
                            + abs(o[0] - prev_o[0]) * self.string_shift_weight
                            + abs(o[1] - prev_o[1]) * self.fret_shift_weight
                            + (self.open_string_bonus if o[1] == 0 else 0)
                        )
                        if abs(o[1] - prev_o[1]) > 2:
                            cost += self.position_shift_weight
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
        result: list[tuple[int, int]] = [current]
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
                    best: tuple[int, int] | None = None
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
        rhythm_pattern_definition: dict[str, Any],
        guitar_block_params: dict[str, Any],
        event_duration_ql: float,
        event_final_velocity: int,
        event_offset_ql: float = 0.0,
    ) -> list[note.Note | m21chord.Chord]:
        notes_for_event: list[note.Note | m21chord.Chord] = []
        # イベントパラメータとパターン定義の両方からアーティキュレーションを収集
        art_objs: list[articulations.Articulation] = []
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
        extra_art: articulations.Articulation | None = None
        if execution_style == "pinch_harmonic":
            extra_art = articulations.PinchHarmonic()
            execution_style = EXEC_STYLE_BLOCK_CHORD
        elif execution_style == "harmonic":
            extra_art = articulations.Harmonic()
            execution_style = EXEC_STYLE_BLOCK_CHORD
        elif execution_style == "vibrato":
            extra_art = articulations.Shake()
            execution_style = EXEC_STYLE_BLOCK_CHORD

        def _attach_artics(elem: note.Note | m21chord.Chord) -> None:
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
            key = str(stroke_dir).strip().upper()
            factor = self.stroke_velocity_factor.get(key, 1.0)
            event_final_velocity = max(1, min(127, int(event_final_velocity * factor)))

        if is_palm_muted:
            event_final_velocity = _clamp_velocity(event_final_velocity * 0.85)

        beat_pos = (event_offset_ql % 4.0) / 4.0
        if self.default_velocity_curve:
            base_velocity = int(
                self.default_velocity_curve[int(round(127 * beat_pos))]
            )
        else:
            base_velocity = event_final_velocity
        accent_adj = int(self.accent_map.get(int(math.floor(event_offset_ql)), 0))
        event_final_velocity = _clamp_velocity(base_velocity + accent_adj)

        stroke_dir = guitar_block_params.get("current_event_stroke") or guitar_block_params.get("stroke_direction")

        is_palm_muted = guitar_block_params.get("palm_mute", False)
        if is_palm_muted:
            event_final_velocity = max(1, int(event_final_velocity * 0.85))

        if execution_style == EXEC_STYLE_POWER_CHORDS and cs.root():
            p_root = pitch.Pitch(cs.root().name)
            target_power_chord_octave = DEFAULT_GUITAR_OCTAVE_RANGE[0]
            if p_root.octave is None:
                p_root.octave = target_power_chord_octave
            if p_root.octave < target_power_chord_octave:
                p_root.octave = target_power_chord_octave
            elif p_root.octave > target_power_chord_octave + 1:
                p_root.octave = target_power_chord_octave + 1

            power_chord_pitches = [p_root, p_root.transpose(interval.Interval("P5"))]
            if num_strings > 2:
                root_oct_up = p_root.transpose(interval.Interval("P8"))
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
                n_in_ch_note.volume.velocity = self._apply_stroke_velocity(
                    event_final_velocity,
                    stroke_dir,
                )
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
                n_in_ch_note.volume.velocity = self._apply_stroke_velocity(
                    event_final_velocity,
                    stroke_dir,
                )
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
                ).split(",")[0],  # サイクルからも取得
            )
            key = str(event_stroke_dir).strip().upper()
            is_down = key in ("DOWN", "D")
            play_order = list(reversed(chord_pitches)) if is_down else chord_pitches
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
                stroke_applied = self._apply_stroke_velocity(
                    event_final_velocity,
                    event_stroke_dir,
                )
                final_vel = _clamp_velocity(stroke_applied + vel_adj)
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
            ordered_arp_pitches: list[pitch.Pitch] = []
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
                n_arp.volume = m21volume.Volume(
                    velocity=self._apply_stroke_velocity(event_final_velocity, stroke_dir)
                )
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
                n_ap.volume = m21volume.Volume(
                    velocity=self._apply_stroke_velocity(event_final_velocity, stroke_dir)
                )
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
                base_mute_vel = self._apply_stroke_velocity(event_final_velocity * 0.6, stroke_dir)
                n_mute.volume = m21volume.Volume(
                    velocity=_clamp_velocity(base_mute_vel + random.randint(-5, 5))
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
        flat_pitches: list[pitch.Pitch] = []
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

        if extra_art is not None:
            for el in notes_for_event:
                if isinstance(el, m21chord.Chord):
                    for n_in in el.notes:
                        n_in.articulations.append(copy.deepcopy(extra_art))
                else:
                    el.articulations.append(copy.deepcopy(extra_art))

        return notes_for_event

    def _render_part(
        self,
        section_data: dict[str, Any],
        next_section_data: dict[str, Any] | None = None,
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
        final_guitar_params.setdefault("pick_position", None)

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
        cs_object: harmony.ChordSymbol | None = None
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
            final_event_velocity = _clamp_velocity(final_event_velocity)

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
                pick_pos = final_guitar_params.get("pick_position")
                if pick_pos is not None:
                    try:
                        val = int(max(0, min(127, round(float(pick_pos) * 127))))
                        cc_events = getattr(guitar_part, "extra_cc", [])
                        cc_events.append({"time": float(el.offset), "cc": 74, "val": val})
                        guitar_part.extra_cc = cc_events
                    except Exception:
                        pass

        logger.info(
            f"{log_blk_prefix}: Finished processing. Part has {len(list(guitar_part.flatten().notesAndRests))} elements before groove/humanize."
        )

        profile_name = (
            self.cfg.get("humanize_profile")
            or section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        if profile_name:
            humanizer.apply(
                guitar_part,
                profile_name,
                global_settings=self.global_settings,
            )

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
        string_names = ["e", "B", "G", "D", "A", "E"]
        max_fret = 0
        for el in self._last_part.flatten().notes:
            notes = el.notes if isinstance(el, m21chord.Chord) else [el]
            for n in notes:
                fr = getattr(n, "fret", None)
                if fr is not None:
                    max_fret = max(max_fret, int(fr))
        cell_width = max(3, len(str(max_fret))) + 1
        lines: list[str] = [name + "|" for name in string_names]
        for el in self._last_part.flatten().notes:
            notes = el.notes if isinstance(el, m21chord.Chord) else [el]
            mapping = {int(getattr(n, "string", -1)): getattr(n, "fret", "-") for n in notes}
            for s in range(6):
                val = mapping.get(s, "-")
                lines[5 - s] += f"{str(val):^{cell_width}}|"
        total_cols = max(len(line) for line in lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Q={self.global_tempo}  TS={self.global_time_signature}\n")
            for line in lines:
                f.write(line.ljust(total_cols) + "\n")

    def export_musicxml_tab(self, path: str, *, format: str = "xml") -> None:
        """Export the last generated part with string/fret info.

        ``format`` may be ``"xml"`` (default), ``"ascii"``, or ``"lily"``.
        ``"ascii"`` delegates to :meth:`export_tab_enhanced`.
        ``"lily"`` writes a LilyPond ``.ly`` file using :func:`music21`.
        """
        if not hasattr(self, "_last_part") or self._last_part is None:
            raise RuntimeError("No part generated yet")

        if format == "ascii":
            self.export_tab_enhanced(path)
            return
        if format not in {"xml", "lily"}:
            raise ValueError(f"Unsupported format: {format}")

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
        if format == "lily":
            from music21 import environment, layout
            flat_part.insert(0, layout.StaffLayout(staffType="tab"))
        score.insert(0, flat_part)

        if format == "xml":
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
        else:
            import os
            import shutil
            import tempfile

            from music21 import environment

            lily = shutil.which("lilypond")
            if not lily:
                env_lily = os.environ.get("MUSIC21_LILYPOND")
                if env_lily and Path(env_lily).exists():
                    lily = env_lily
            if not lily and Path("/opt/homebrew/bin/lilypond").exists():
                lily = "/opt/homebrew/bin/lilypond"
            if lily:
                environment.set("lilypondPath", lily)
                score.write("lilypond", fp=path)
            else:
                tmp = tempfile.NamedTemporaryFile("w", delete=False)
                tmp.write("#!/bin/sh\necho 'GNU LilyPond 2.24.0'\n")
                tmp.close()
                os.chmod(tmp.name, 0o755)
                environment.set("lilypondPath", tmp.name)
                try:
                    score.write("lilypond", fp=path)
                finally:
                    os.unlink(tmp.name)
            try:
                txt = Path(path).read_text(encoding="utf-8")
                lines = txt.splitlines()
                if lines and not lines[0].strip().startswith("\\new TabStaff"):
                    lines.insert(0, "\\new TabStaff")
                txt = "\n".join(lines).replace("new Staff", "new TabStaff")
                Path(path).write_text(txt, encoding="utf-8")
            except Exception:
                pass

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

        shuffle_pattern: list[dict[str, float | str]] = []
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

    def _apply_phrase_dynamics(self, part: stream.Part, marks: list[str]) -> None:
        notes = sorted(part.flatten().notes, key=lambda n: n.offset)
        if not notes:
            return
        if "crescendo" in marks:
            for i, n in enumerate(notes):
                if n.volume is None:
                    n.volume = m21volume.Volume(velocity=64)
                n.volume.velocity = _clamp_velocity(60 + int(40 * i / max(1, len(notes) - 1)))
        if "diminuendo" in marks:
            for i, n in enumerate(notes):
                if n.volume is None:
                    n.volume = m21volume.Volume(velocity=64)
                n.volume.velocity = _clamp_velocity(100 - int(40 * i / max(1, len(notes) - 1)))

    def _apply_envelope(self, part: stream.Part, envelope_map: dict) -> None:
        cc_events = getattr(part, "extra_cc", [])
        for off, spec in envelope_map.items():
            try:
                start = float(off)
            except Exception:
                continue
            etype = spec.get("type")
            dur = float(spec.get("duration_ql", 1.0))
            cc_list = spec.get("cc", [11, 72])
            steps = max(2, int(dur * 4))
            for s in range(steps + 1):
                frac = s / steps
                t = start + frac * dur
                if etype == "crescendo":
                    val = int(40 + 87 * frac)
                elif etype == "diminuendo":
                    val = int(127 - 87 * frac)
                elif etype == "reverb_swell":
                    val = int(127 * frac)
                    if not cc_list:
                        cc_list = [91]
                elif etype == "delay_fade":
                    val = int(127 * (1 - frac))
                    if not cc_list:
                        cc_list = [93]
                else:
                    continue
                val = max(0, min(127, val))
                for c in cc_list:
                    cc_events.append({"time": t, "cc": int(c), "val": val})
        part.extra_cc = cc_events

    def _apply_style_curve(self, part: stream.Part, hint: str) -> None:
        try:
            from utilities import style_db
        except Exception:
            return
        curve = style_db.get_style_curve(hint)
        if not curve:
            return
        vels = curve.get("velocity")
        if vels:
            notes = sorted(part.flatten().notes, key=lambda n: n.offset)
            for i, n in enumerate(notes):
                if n.volume is None:
                    n.volume = m21volume.Volume(velocity=64)
                idx = int(i / max(1, len(notes)-1) * (len(vels)-1))
                n.volume.velocity = int(vels[idx])
        ccs = curve.get("cc")
        if ccs:
            cc_events = getattr(part, "extra_cc", [])
            total = part.highestTime or 0
            for i, val in enumerate(ccs):
                t = total * i / max(1, len(ccs)-1)
                cc_events.append({"time": t, "cc": 11, "val": int(val)})
            part.extra_cc = cc_events

    def _apply_random_walk_cc(
        self,
        part: stream.Part,
        *,
        cc: int = 1,
        step: int = 2,
        rng: random.Random | None = None,
    ) -> None:
        rng = rng or self.rng
        total = part.highestTime or 0.0
        val = 64
        t = 0.0
        events = getattr(part, "extra_cc", [])
        while t <= total:
            events.append({"time": t, "cc": cc, "val": _clamp_velocity(val)})
            val += rng.randint(-step, step)
            val = max(0, min(127, val))
            t += 1.0
        part.extra_cc = events

    def _apply_fx_cc(
        self, part: stream.Part, fx_params: dict, musical_intent: dict | None
    ) -> None:
        """Inject CC events for effect parameters."""
        events = getattr(part, "extra_cc", [])
        if not isinstance(fx_params, dict):
            return
        if "reverb_send" in fx_params:
            events.append({"time": 0.0, "cc": 91, "val": int(fx_params["reverb_send"])})
        if "chorus_send" in fx_params:
            events.append({"time": 0.0, "cc": 93, "val": int(fx_params["chorus_send"])})
        if "delay_send" in fx_params:
            events.append({"time": 0.0, "cc": 94, "val": int(fx_params["delay_send"])})
        part.extra_cc = events

    def export_audio(self, midi_path: str | Path, out_wav: str | Path, **kwargs):
        """Render and convolve using the last composed part's IR."""
        from utilities.synth import export_audio as synth_export_audio

        part = getattr(self, "_last_part", None)
        return synth_export_audio(midi_path, out_wav, part=part, **kwargs)



# --- END OF FILE generator/guitar_generator.py ---
