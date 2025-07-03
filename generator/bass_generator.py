# --- START OF FILE generator/bass_generator.py (simplified) ---
import copy
import logging
import math
import random  # 追加
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import music21
import numpy as np
from music21 import (
    duration as m21duration,
)
from music21 import (
    harmony,
    interval,
    key,
    meter,
    note,
    pitch,
    scale,
    stream,
)
from music21 import (
    volume as m21volume,
)

from utilities import MIN_NOTE_DURATION_QL, humanizer
from utilities.tone_shaper import ToneShaper

try:
    from cyext import (
        postprocess_kick_lock as cy_postprocess_kick_lock,
    )
    from cyext import (
        velocity_random_walk as cy_velocity_random_walk,
    )
except Exception:  # pragma: no cover - optional
    cy_postprocess_kick_lock = None
    cy_velocity_random_walk = None
import yaml

from utilities.accent_mapper import AccentMapper
from utilities.emotion_profile_loader import load_emotion_profile
from utilities.velocity_curve import resolve_velocity_curve

from .base_part_generator import BasePartGenerator

try:
    from utilities.safe_get import safe_get
except ImportError:
    logger_fallback_safe_get_bass = logging.getLogger(
        __name__ + ".fallback_safe_get_bass"
    )
    logger_fallback_safe_get_bass.error(
        "BassGen: CRITICAL - Could not import safe_get. Fallback will be basic .get()."
    )

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
    from utilities import bass_utils
    from utilities.bass_utils import get_approach_note
    from utilities.core_music_utils import (
        MIN_NOTE_DURATION_QL,
        get_time_signature_object,
        sanitize_chord_label,
    )
    from utilities.override_loader import (
        PartOverride,
    )  # ★★★ PartOverrideModel -> PartOverride ★★★
    from utilities.scale_registry import ScaleRegistry
except ImportError as e:
    print(f"BassGenerator: Warning - could not import some core utilities: {e}")
    MIN_NOTE_DURATION_QL = 0.125

    def get_time_signature_object(ts_str: str | None) -> meter.TimeSignature:
        return meter.TimeSignature(ts_str or "4/4")

    def sanitize_chord_label(label: str | None) -> str | None:
        return label

    class ScaleRegistry:
        @staticmethod
        def get(
            tonic_str: str | None, mode_str: str | None
        ) -> scale.ConcreteScale:
            return scale.MajorScale(tonic_str or "C")

    def get_approach_note(
        from_p,
        to_p,
        scale_o,
        style="chromatic_or_diatonic",
        max_s=2,
        pref_dir: int | str | None = None,
    ):
        if to_p:
            return to_p
        if from_p:
            return from_p
        return pitch.Pitch("C4")

    class DummyBassUtils:
        @staticmethod
        def mirror_pitches(vocal_notes, tonic_pitch, target_octave=2):
            return [tonic_pitch for _ in vocal_notes]

    bass_utils = DummyBassUtils()

    class PartOverride:
        model_config = {}
        model_fields = {}
        velocity_shift: int | None = None
        velocity_shift_on_kick: int | None = None
        velocity: int | None = None
        options: dict[str, Any] | None = None
        rhythm_key: str | None = None

        def model_dump(self, exclude_unset=True):
            return {}


logger = logging.getLogger("modular_composer.bass_generator")


def _apply_tone(part: stream.Part, intensity: str) -> None:
    """Insert CC31 events based on average velocity."""
    notes = list(part.flat.notes)
    if not notes:
        return
    avg_vel = float(np.mean([n.volume.velocity or 0 for n in notes]))
    shaper = ToneShaper()
    preset = shaper.choose_preset(avg_vel, intensity)
    part.extra_cc = getattr(part, "extra_cc", [])
    part.extra_cc.extend(shaper.to_cc_events(preset, 0.0))

DIRECTION_UP = 1
DIRECTION_DOWN = -1

EMOTION_TO_BUCKET_BASS: dict[str, str] = {
    "quiet_pain_and_nascent_strength": "calm",
    "deep_regret_gratitude_and_realization": "calm",
    "self_reproach_regret_deep_sadness": "calm",
    "memory_unresolved_feelings_silence": "calm",
    "nature_memory_floating_sensation_forgiveness": "calm",
    "supported_light_longing_for_rebirth": "groovy",
    "wavering_heart_gratitude_chosen_strength": "groovy",
    "hope_dawn_light_gentle_guidance": "groovy",
    "acceptance_of_love_and_pain_hopeful_belief": "energetic",
    "trial_cry_prayer_unbreakable_heart": "energetic",
    "reaffirmed_strength_of_love_positive_determination": "energetic",
    "future_cooperation_our_path_final_resolve_and_liberation": "energetic",
    "default": "groovy",
}
BUCKET_TO_PATTERN_BASS: dict[tuple[str, str], str] = {
    ("calm", "low"): "root_only",
    ("calm", "medium_low"): "root_fifth",
    ("calm", "medium"): "bass_half_time_pop",
    ("calm", "medium_high"): "bass_half_time_pop",
    ("calm", "high"): "walking",
    ("groovy", "low"): "bass_syncopated_rnb",
    ("groovy", "medium_low"): "walking",
    ("groovy", "medium"): "walking_8ths",
    ("groovy", "medium_high"): "walking_8ths",
    ("groovy", "high"): "bass_funk_octave",
    ("energetic", "low"): "bass_quarter_notes",
    ("energetic", "medium_low"): "bass_pump_8th_octaves",
    ("energetic", "medium"): "bass_pump_8th_octaves",
    ("energetic", "medium_high"): "bass_funk_octave",
    ("energetic", "high"): "bass_funk_octave",
    ("default", "low"): "root_only",
    ("default", "medium_low"): "bass_quarter_notes",
    ("default", "medium"): "walking",
    ("default", "medium_high"): "walking_8ths",
    ("default", "high"): "bass_pump_8th_octaves",
}


class BassGenerator(BasePartGenerator):
    def __init__(
        self,
        *,
        global_settings=None,
        default_instrument=None,
        global_tempo=None,
        global_time_signature=None,
        global_key_signature_tonic=None,
        global_key_signature_mode=None,
        mirror_melody: bool = False,
        main_cfg=None,
        emotion_profile_path: str | Path | None = None,
        **kwargs,
    ):
        """Create a bass part generator.

        Parameters
        ----------
        mirror_melody : bool, optional
            If ``True``, invert the melody line around the tonic when generating
            bass notes. Default is ``False``.  Set this in ``main_cfg.yml`` under
            ``part_defaults.bass`` to apply globally.
        """
        self.kick_lock_cfg = (global_settings or {}).get("kick_lock", {})
        seed = self.kick_lock_cfg.get("random_seed")
        self._rng = random.Random(0 if seed is None else seed)

        super().__init__(
            global_settings=global_settings,
            default_instrument=default_instrument,
            global_tempo=global_tempo,
            global_time_signature=global_time_signature,
            global_key_signature_tonic=global_key_signature_tonic,
            global_key_signature_mode=global_key_signature_mode,
            rng=self._rng,
            **kwargs,
        )
        self.cfg: dict = kwargs.copy()
        self.logger = logging.getLogger("modular_composer.bass_generator")
        self.part_parameters = kwargs.get("part_parameters", {})
        self.vel_shift_on_kick = self.part_parameters.get("velocity_shift_on_kick", 10)
        self.main_cfg = main_cfg or {"global_settings": {}}
        # Whether to mirror the main melody when generating the bass line
        self.mirror_melody = mirror_melody
        # ここで global_ts_str をセット
        self.global_ts_str = global_time_signature
        self.global_time_signature_obj = get_time_signature_object(self.global_ts_str)
        self.global_key_tonic = self.main_cfg.get("global_settings", {}).get(
            "key_tonic"
        )
        self.global_key_mode = self.main_cfg.get("global_settings", {}).get("key_mode")

        if self.global_time_signature_obj is None:
            self.logger.warning(
                "BassGenerator: global_time_signature_obj が設定されていません。"
            )
        self.measure_duration = (
            self.global_time_signature_obj.barDuration.quarterLength
            if self.global_time_signature_obj
            and hasattr(self.global_time_signature_obj, "barDuration")
            else 4.0
        )
        self._add_internal_default_patterns()

        # range and swing settings
        gs = global_settings or {}
        self.bass_range_lo = int(gs.get("bass_range_lo", 30))
        self.bass_range_hi = int(gs.get("bass_range_hi", 72))
        self.swing_ratio = float(gs.get("swing_ratio", 0.0))
        self._step_range = int(gs.get("random_walk_step", 8))

        self._root_offsets: list[float] = []
        self._vel_walk_state: int = 0

        self.phrase_map: dict[str, dict] = {}
        self.intensity_scale: float = 0.5
        self.custom_phrases: dict[str, Any] = {}
        self.phrase_insertions: dict[str, str] = {}

        self.kick_offsets: list[float] = []
        self.base_velocity = 70

        if emotion_profile_path is None:
            default_prof = Path("data/emotion_profile.yaml")
            if default_prof.exists():
                emotion_profile_path = default_prof
        if emotion_profile_path:
            try:
                self.emotion_profile = load_emotion_profile(emotion_profile_path)
            except Exception as exc:
                self.logger.error(f"Failed to load emotion profile: {exc}")
                self.emotion_profile = {}
        else:
            self.emotion_profile = {}

    def compose(
        self,
        *,
        section_data: dict[str, Any],
        overrides_root: Any | None = None,
        groove_profile_path: str | None = None,
        next_section_data: dict[str, Any] | None = None,
        part_specific_humanize_params: dict[str, Any] | None = None,
        shared_tracks: dict[str, Any] | None = None,
    ) -> stream.Part:
        if shared_tracks and "kick_offsets" in shared_tracks:
            self.kick_offsets = list(shared_tracks["kick_offsets"])
        else:
            self.kick_offsets = []

        self.base_velocity = (
            section_data.get("part_params", {})
            .get("bass", {})
            .get("velocity", 70)
        )

        if self.mirror_melody and section_data.get("vocal_notes"):
            tonic_str = (
                section_data.get("tonic_of_section", self.global_key_tonic) or "C"
            )
            tonic_pitch = pitch.Pitch(tonic_str)
            mirrored = bass_utils.mirror_pitches(
                section_data.get("vocal_notes", []), tonic_pitch
            )
            bass_part = stream.Part(id=self.part_name)
            inst = copy.deepcopy(self.default_instrument)
            bass_part.insert(0, inst)
            for vn, mp in zip(section_data["vocal_notes"], mirrored):
                n = note.Note(mp, quarterLength=vn.quarterLength)
                bass_part.insert(vn.offset, n)
            return bass_part

        section_data.setdefault("part_params", {}).setdefault(
            "swing_ratio", self.swing_ratio
        )

        result = super().compose(
            section_data=section_data,
            overrides_root=overrides_root,
            groove_profile_path=groove_profile_path,
            next_section_data=next_section_data,
            part_specific_humanize_params=part_specific_humanize_params,
            shared_tracks=shared_tracks,
        )

        if self.overrides and self.overrides.velocity_shift_on_kick is not None:
            self.vel_shift_on_kick = self.overrides.velocity_shift_on_kick

        def apply_shift(part: stream.Part):
            self._postprocess_notes_for_kick_lock(part, self.kick_offsets)
            self._apply_velocity_random_walk(part)
            self._clamp_range(part)
            if self.kick_lock_cfg.get("enabled", False) and shared_tracks is not None:
                kicks = shared_tracks.get("kick_offsets_sec", [])
                self._apply_kick_lock(part, kicks)
            self._root_offsets = [float(n.offset) for n in part.flatten().notes]
            return part

        intensity_label = section_data.get("musical_intent", {}).get("intensity", "medium")

        if isinstance(result, dict):
            for p in result.values():
                apply_shift(p)
                _apply_tone(p, str(intensity_label))
                label = section_data.get("section_name")
                if label in self.phrase_insertions:
                    self._insert_custom_phrase(p, self.phrase_insertions[label])
                intensity = self.phrase_map.get(label, {}).get("intensity")
                if intensity is not None:
                    self._apply_phrase_dynamics(p, float(intensity))
            return result
        else:
            part = apply_shift(result)
            _apply_tone(part, str(intensity_label))
            label = section_data.get("section_name")
            if label in self.phrase_insertions:
                self._insert_custom_phrase(part, self.phrase_insertions[label])
            intensity = self.phrase_map.get(label, {}).get("intensity")
            if intensity is not None:
                self._apply_phrase_dynamics(part, float(intensity))
            return part

    def _postprocess_notes_for_kick_lock(
        self, part: stream.Part, kick_offsets: Sequence[float], eps: float = 0.03
    ) -> None:
        if cy_postprocess_kick_lock is not None:
            cy_postprocess_kick_lock(part, kick_offsets, self.vel_shift_on_kick, eps)
            return
        if not kick_offsets:
            return
        for n in part.recurse().notes:
            if any(abs(float(n.offset) - k) < eps for k in kick_offsets):
                if n.volume is None:
                    n.volume = m21volume.Volume()
                n.volume.velocity = min(
                    127, (n.volume.velocity or 64) + self.vel_shift_on_kick
                )

    def _apply_velocity_random_walk(self, part: stream.Part) -> None:
        """Apply small bar-by-bar velocity fluctuation."""
        if not part.flatten().notes:
            return
        if cy_velocity_random_walk is not None:
            cy_velocity_random_walk(
                part,
                self.measure_duration,
                self._step_range,
                self._rng,
            )
            self._vel_walk_state = 0
            return
        bar_len = self.measure_duration
        notes = sorted(part.flatten().notes, key=lambda n: n.offset)
        current_bar_start = 0.0
        self._vel_walk_state = 0
        for n in notes:
            while n.offset >= current_bar_start + bar_len - 1e-6:
                current_bar_start += bar_len
                step = self._rng.randint(-self._step_range, self._step_range)
                self._vel_walk_state = max(
                    -self._step_range,
                    min(self._step_range, self._vel_walk_state + step),
                )
            if n.volume is None:
                n.volume = m21volume.Volume()
            n.volume.velocity = max(
                1,
                min(127, (n.volume.velocity or 64) + self._vel_walk_state),
            )

    def _clamp_range(self, part: stream.Part) -> None:
        """Ensure all notes fall within bass range."""
        for n in part.flatten().notes:
            midi = n.pitch.midi
            while midi > self.bass_range_hi:
                midi -= 12
            while midi < self.bass_range_lo:
                midi += 12
            midi = max(self.bass_range_lo, min(self.bass_range_hi, midi))
            n.pitch.midi = midi

    def _apply_kick_lock(self, part: stream.Part, kick_offsets_sec: list[float]) -> None:
        if not kick_offsets_sec:
            return
        cfg = self.kick_lock_cfg
        window = cfg.get("window_ms", 30) / 1000.0
        vel_boost = int(cfg.get("vel_boost", 12))
        gb = cfg.get("ghost_before_ms", [30, 90])
        if not isinstance(gb, (list, tuple)) or len(gb) < 2:
            gb = [30, 90]
        ghost_vel_ratio = float(cfg.get("ghost_vel_ratio", 0.35))
        notes = sorted(part.flatten().notes, key=lambda n: n.offset)

        base_bpm = self.global_tempo or 120
        if (
            hasattr(self, "tempo_map")
            and self.tempo_map
            and hasattr(self.tempo_map, "get_bpm")
        ):
            base_bpm = self.tempo_map.get_bpm(0)
        seconds_per_beat = 60.0 / base_bpm

        def ql_to_sec(ql: float) -> float:
            bpm = (
                self.tempo_map.get_bpm(ql)
                if hasattr(self, "tempo_map")
                and self.tempo_map
                and hasattr(self.tempo_map, "get_bpm")
                else base_bpm
            )
            return ql * 60.0 / bpm

        # Lock velocity boost
        for n in notes:
            off_sec = ql_to_sec(float(n.offset))
            if any(abs(off_sec - k) <= window for k in kick_offsets_sec):
                if n.volume is None:
                    n.volume = m21volume.Volume()
                n.volume.velocity = min(
                    127, max(1, (n.volume.velocity or self.base_velocity) + vel_boost)
                )

        # Lift ghost notes
        if not notes:
            return
        root_pitch = notes[0].pitch
        for k in kick_offsets_sec:
            delay = self._rng.uniform(gb[0], gb[1]) / 1000.0
            if delay < 0.001:
                continue
            ghost_time = k - delay
            if ghost_time < 0:
                continue
            if (
                hasattr(self, "tempo_map")
                and self.tempo_map
                and hasattr(self.tempo_map, "get_bpm")
            ):
                beat_idx = ghost_time / seconds_per_beat
                cur_bpm = self.tempo_map.get_bpm(beat_idx)
            else:
                cur_bpm = self.global_tempo or 120
            off_ql = ghost_time * cur_bpm / 60.0
            prev: note.Note | None = None
            for n in reversed(notes):
                if n.offset <= off_ql:
                    prev = n
                    break
            pitch_obj = prev.pitch if prev else root_pitch
            base_vel = (
                prev.volume.velocity if prev and prev.volume else self.base_velocity
            )
            dur = min(0.1, (prev.duration.quarterLength / 2 if prev else 0.1))
            g = note.Note(pitch_obj)
            g.duration = m21duration.Duration(dur)
            g.volume = m21volume.Volume(velocity=int(base_vel * ghost_vel_ratio))
            g.offset = off_ql
            part.insert(off_ql, g)
            notes.append(g)

    def _add_internal_default_patterns(self):
        # (変更なし)
        defaults_to_add = {
            "basic_chord_tone_quarters": {
                "description": "Algorithmic quarter notes on chord tones.",
                "pattern_type": "algorithmic_chord_tone_quarters",
                "options": {
                    "velocity_base": 75,
                    "velocity_factor": 1.0,
                    "weak_beat_style": "root",
                    "approach_on_4th_beat": True,
                    "approach_style_on_4th": "diatonic_or_chromatic",
                },
                "reference_duration_ql": self.measure_duration,
            },
            "bass_quarter_notes": {
                "description": "Fixed quarter note roots.",
                "pattern_type": "fixed_pattern",
                "pattern": [
                    {
                        "offset": i * 1.0,
                        "duration": 1.0,
                        "velocity_factor": 0.9,
                        "type": "root",
                    }
                    for i in range(int(self.measure_duration))
                ],
                "reference_duration_ql": self.measure_duration,
            },
            "root_quarters": {
                "description": "Fixed quarter note roots.",
                "pattern_type": "fixed_pattern",
                "pattern": [
                    {
                        "offset": i * 1.0,
                        "duration": 1.0,
                        "velocity_factor": 1.0,
                        "type": "root",
                    }
                    for i in range(int(self.measure_duration))
                ],
                "reference_duration_ql": self.measure_duration,
            },
            "walking_quarters": {
                "description": "Quarter roots with walking approach notes.",
                "pattern_type": "walking_quarters",
                "options": {
                    "velocity_base": 70,
                    "velocity_factor": 1.0,
                    "target_octave": 2,
                },
                "reference_duration_ql": self.measure_duration,
            },
            "root_only": {
                "description": "Algorithmic whole notes on root.",
                "pattern_type": "algorithmic_root_only",
                "options": {
                    "velocity_base": 65,
                    "velocity_factor": 1.0,
                    "note_duration_ql": self.measure_duration,
                },
                "reference_duration_ql": self.measure_duration,
            },
            "walking": {
                "description": "Algorithmic walking bass (quarters).",
                "pattern_type": "algorithmic_walking",
                "options": {
                    "velocity_base": 70,
                    "velocity_factor": 1.0,
                    "step_ql": 1.0,
                    "approach_style": "diatonic_prefer_scale",
                },
                "reference_duration_ql": self.measure_duration,
            },
            "walking_8ths": {
                "description": "Algorithmic walking bass (8ths).",
                "pattern_type": "algorithmic_walking_8ths",
                "options": {
                    "velocity_base": 72,
                    "velocity_factor": 0.9,
                    "step_ql": 0.5,
                    "approach_style": "chromatic_or_diatonic",
                    "approach_note_prob": 0.5,
                },
                "reference_duration_ql": self.measure_duration,
            },
            "algorithmic_pedal": {
                "description": "Algorithmic pedal tone.",
                "pattern_type": "algorithmic_pedal",
                "options": {
                    "velocity_base": 68,
                    "velocity_factor": 1.0,
                    "note_duration_ql": self.measure_duration,
                    "subdivision_ql": self.measure_duration,
                    "pedal_note_type": "root",
                },
                "reference_duration_ql": self.measure_duration,
            },
            "root_fifth": {
                "description": "Algorithmic root and fifth.",
                "pattern_type": "algorithmic_root_fifth",
                "options": {
                    "velocity_base": 70,
                    "beat_duration_ql": 1.0,
                    "arrangement": ["R", "5", "R", "5"],
                },
                "reference_duration_ql": self.measure_duration,
            },
            "bass_funk_octave": {
                "description": "Algorithmic funk octave pops.",
                "pattern_type": "funk_octave_pops",
                "options": {
                    "velocity_base": 80,
                    "base_rhythm_ql": 0.25,
                    "accent_factor": 1.2,
                    "ghost_factor": 0.5,
                    "syncopation_prob": 0.3,
                    "octave_jump_prob": 0.6,
                },
                "reference_duration_ql": self.measure_duration,
            },
        }
        for key_pat, val_pat in defaults_to_add.items():
            if key_pat not in self.part_parameters:
                self.part_parameters[key_pat] = val_pat
                self.logger.info(
                    f"BassGenerator: Added default pattern '{key_pat}' to internal rhythm_lib."
                )

    def _choose_bass_pattern_key(self, section_musical_intent: dict) -> str:
        # (変更なし)
        emotion = section_musical_intent.get("emotion", "default")
        intensity = section_musical_intent.get("intensity", "medium").lower()
        bucket = EMOTION_TO_BUCKET_BASS.get(emotion, "default")
        pattern_key = BUCKET_TO_PATTERN_BASS.get(
            (bucket, intensity), "bass_quarter_notes"
        )
        if pattern_key not in self.part_parameters:  # ←修正
            self.logger.warning(
                f"Chosen pattern key '{pattern_key}' (for emotion '{emotion}', intensity '{intensity}') not in library. Falling back to 'basic_chord_tone_quarters'."
            )
            return "basic_chord_tone_quarters"
        return pattern_key

    def _get_rhythm_pattern_details(self, rhythm_key: str) -> dict[str, Any]:
        # (変更なし)
        if not rhythm_key or rhythm_key not in self.part_parameters:  # ←修正
            self.logger.warning(
                f"BassGenerator: Rhythm key '{rhythm_key}' not found. Using 'basic_chord_tone_quarters'."
            )
            rhythm_key = "basic_chord_tone_quarters"
        details = self.part_parameters.get(rhythm_key)  # ←修正
        if not details:
            self.logger.error(
                "BassGenerator: CRITICAL - Default 'basic_chord_tone_quarters' also not found. Using minimal root_only."
            )
            return {
                "pattern_type": "algorithmic_root_only",
                "pattern": [],
                "options": {"velocity_base": 70, "velocity_factor": 1.0},
                "reference_duration_ql": self.measure_duration,
            }
        if not details.get("pattern_type"):
            if rhythm_key == "root_fifth":
                details["pattern_type"] = "algorithmic_root_fifth"
            elif rhythm_key == "bass_funk_octave":
                details["pattern_type"] = "funk_octave_pops"
            elif rhythm_key == "bass_walking_8ths":
                details["pattern_type"] = "algorithmic_walking_8ths"
            elif rhythm_key == "walking":
                details["pattern_type"] = "algorithmic_walking"
        details.setdefault("options", {}).setdefault("velocity_factor", 1.0)
        details.setdefault("reference_duration_ql", self.measure_duration)
        return details

    def _get_bass_pitch_in_octave(
        self, base_pitch_obj: pitch.Pitch | None, target_octave: int
    ) -> int:
        # (変更なし)
        if not base_pitch_obj:
            return pitch.Pitch(f"C{target_octave}").midi
        p_new = pitch.Pitch(base_pitch_obj.name)
        p_new.octave = target_octave
        min_bass_midi = 28
        max_bass_midi = 60
        current_midi = p_new.midi
        while current_midi < min_bass_midi:
            current_midi += 12
        while current_midi > max_bass_midi:
            if current_midi - 12 >= min_bass_midi:
                current_midi -= 12
            else:
                break
        return max(min_bass_midi, min(current_midi, max_bass_midi))

    def _generate_notes_from_fixed_pattern(
        self,
        pattern_events: list[dict[str, Any]],
        m21_cs: harmony.ChordSymbol,
        block_base_velocity: int,
        target_octave: int,
        block_duration: float,
        pattern_reference_duration_ql: float,
        current_scale: scale.ConcreteScale | None = None,
        velocity_curve: list[float] | None = None,
    ) -> list[tuple[float, note.Note]]:
        # (変更なし)
        notes: list[tuple[float, note.Note]] = []
        if not m21_cs or not m21_cs.pitches:
            self.logger.debug(
                "BassGen _generate_notes_from_fixed_pattern: ChordSymbol is None or has no pitches."
            )
            return notes
        root_pitch_obj = m21_cs.root()
        third_pitch_obj = m21_cs.third
        fifth_pitch_obj = m21_cs.fifth
        chord_tones = [
            p for p in [root_pitch_obj, third_pitch_obj, fifth_pitch_obj] if p
        ]
        time_scale_factor = (
            block_duration / pattern_reference_duration_ql
            if pattern_reference_duration_ql > 0
            else 1.0
        )
        for p_event_idx, p_event in enumerate(pattern_events):
            log_prefix = f"BassGen.FixedPattern.Evt{p_event_idx}"
            offset_in_pattern = safe_get(
                p_event,
                "beat",
                default=safe_get(
                    p_event,
                    "offset",
                    default=0.0,
                    cast_to=float,
                    log_name=f"{log_prefix}.OffsetFallback",
                ),
                cast_to=float,
                log_name=f"{log_prefix}.Beat",
            )
            duration_in_pattern = safe_get(
                p_event,
                "duration",
                default=1.0,
                cast_to=float,
                log_name=f"{log_prefix}.Dur",
            )
            if duration_in_pattern <= 0:
                self.logger.warning(
                    f"{log_prefix}: Invalid or zero duration '{p_event.get('duration')}'. Skipping event."
                )
                continue
            vel_factor = safe_get(
                p_event,
                "velocity_factor",
                default=1.0,
                cast_to=float,
                log_name=f"{log_prefix}.VelFactor",
            )
            actual_offset_in_block = offset_in_pattern * time_scale_factor
            actual_duration_ql = duration_in_pattern * time_scale_factor
            if actual_offset_in_block >= block_duration - (MIN_NOTE_DURATION_QL / 16.0):
                continue
            actual_duration_ql = min(
                actual_duration_ql, block_duration - actual_offset_in_block
            )
            if actual_duration_ql < MIN_NOTE_DURATION_QL / 4.0:
                continue
            note_type = (p_event.get("type") or "root").lower()
            final_velocity = max(1, min(127, int(block_base_velocity * vel_factor)))
            layer_idx = p_event.get("velocity_layer")
            if velocity_curve and layer_idx is not None:
                try:
                    idx = int(layer_idx)
                    if 0 <= idx < len(velocity_curve):
                        final_velocity = int(final_velocity * velocity_curve[idx])
                except (TypeError, ValueError):
                    pass
            chosen_pitch_base: pitch.Pitch | None = None
            if note_type == "root" and root_pitch_obj:
                chosen_pitch_base = root_pitch_obj
            elif note_type == "fifth" and fifth_pitch_obj:
                chosen_pitch_base = fifth_pitch_obj
            elif note_type == "third" and third_pitch_obj:
                chosen_pitch_base = third_pitch_obj
            elif note_type == "octave_root" and root_pitch_obj:
                chosen_pitch_base = root_pitch_obj
            elif note_type == "octave_up_root" and root_pitch_obj:
                chosen_pitch_base = root_pitch_obj.transpose(12)
            elif note_type == "random_chord_tone" and chord_tones:
                chosen_pitch_base = self._rng.choice(chord_tones)
            elif note_type == "scale_tone" and current_scale:
                try:
                    p_s = current_scale.pitchFromDegree(
                        self._rng.choice([1, 2, 3, 4, 5, 6, 7])
                    )
                except Exception:
                    p_s = None
                chosen_pitch_base = p_s if p_s else root_pitch_obj
            elif note_type.startswith("approach") and root_pitch_obj:
                chosen_pitch_base = root_pitch_obj.transpose(-1)
            elif note_type == "chord_tone_any" and chord_tones:
                chosen_pitch_base = self._rng.choice(chord_tones)
            elif note_type == "scale_approach_up" and root_pitch_obj and current_scale:
                chosen_pitch_base = current_scale.nextPitch(
                    root_pitch_obj, direction=DIRECTION_UP
                )
            elif (
                note_type == "scale_approach_down" and root_pitch_obj and current_scale
            ):
                chosen_pitch_base = current_scale.nextPitch(
                    root_pitch_obj, direction=DIRECTION_DOWN
                )
            if not chosen_pitch_base and root_pitch_obj:
                chosen_pitch_base = root_pitch_obj
            if chosen_pitch_base:
                midi_pitch_val = self._get_bass_pitch_in_octave(
                    chosen_pitch_base, target_octave
                )
                n = note.Note()
                n.pitch.midi = midi_pitch_val
                n.duration = m21duration.Duration(actual_duration_ql)
                n.volume = m21volume.Volume(velocity=final_velocity)
                if p_event.get("glide_to_next", False):
                    n.tie = music21.tie.Tie("start")
                notes.append((actual_offset_in_block, n))
        return notes

    def _apply_weak_beat(
        self, notes_in_measure: list[tuple[float, note.Note]], style: str
    ) -> list[tuple[float, note.Note]]:
        # (変更なし)
        if style == "none" or not notes_in_measure:
            return notes_in_measure
        new_notes_tuples: list[tuple[float, note.Note]] = []
        beats_in_measure = (
            self.global_time_signature_obj.beatCount
            if self.global_time_signature_obj
            else 4
        )
        beat_q_len = (
            self.global_time_signature_obj.beatDuration.quarterLength
            if self.global_time_signature_obj
            else 1.0
        )
        for rel_offset, note_obj in notes_in_measure:
            is_weak_beat = False
            beat_number_float = rel_offset / beat_q_len
            is_on_beat = abs(beat_number_float - round(beat_number_float)) < 0.01
            beat_index = int(round(beat_number_float))
            if is_on_beat:
                if beats_in_measure == 4 and (beat_index == 1 or beat_index == 3):
                    is_weak_beat = True
                elif beats_in_measure == 3 and (beat_index == 1 or beat_index == 2):
                    is_weak_beat = True
            if is_weak_beat:
                if style == "rest":
                    self.logger.debug(
                        f"BassGen: Removing note at {rel_offset} for weak_beat_style='rest'."
                    )
                    continue
                elif style == "ghost":
                    base_vel_for_ghost = (
                        note_obj.volume.velocity
                        if note_obj.volume and note_obj.volume.velocity is not None
                        else 64
                    )
                    note_obj.volume.velocity = max(1, int(base_vel_for_ghost * 0.4))
                    self.logger.debug(
                        f"BassGen: Ghosting note at {rel_offset} to vel {note_obj.volume.velocity}."
                    )
            new_notes_tuples.append((rel_offset, note_obj))
        return new_notes_tuples

    def _insert_approach_note_to_measure(
        self,
        notes_in_measure: list[tuple[float, note.Note]],
        current_chord_symbol: harmony.ChordSymbol,
        next_chord_root: pitch.Pitch | None,
        current_scale: scale.ConcreteScale,
        approach_style: str,
        target_octave: int,
        effective_velocity_for_approach: int,
    ) -> list[tuple[float, note.Note]]:
        # (変更なし)
        if not next_chord_root or self.measure_duration < 1.0:
            return notes_in_measure
        approach_note_duration_ql = 0.5
        if "16th" in approach_style:
            approach_note_duration_ql = 0.25
        approach_note_rel_offset = self.measure_duration - approach_note_duration_ql
        if approach_note_rel_offset < 0:
            return notes_in_measure
        can_insert = True
        original_last_note_tuple: tuple[float, note.Note] | None = None
        sorted_notes_in_measure = sorted(notes_in_measure, key=lambda x: x[0])
        for rel_offset, note_obj_iter in reversed(sorted_notes_in_measure):
            if rel_offset >= approach_note_rel_offset:
                can_insert = False
                break
            if rel_offset < approach_note_rel_offset and (
                rel_offset + note_obj_iter.duration.quarterLength
                > approach_note_rel_offset
            ):
                original_last_note_tuple = (rel_offset, note_obj_iter)
                break
        if not can_insert:
            self.logger.debug(
                f"BassGen: Cannot insert approach note, existing note at or after {approach_note_rel_offset:.2f}"
            )
            return notes_in_measure
        from_pitch_for_approach = current_chord_symbol.root()
        if original_last_note_tuple:
            from_pitch_for_approach = original_last_note_tuple[1].pitch
        elif sorted_notes_in_measure:
            from_pitch_for_approach = sorted_notes_in_measure[-1][1].pitch
        approach_pitch_obj = get_approach_note(
            from_pitch_for_approach,
            next_chord_root,
            current_scale,
            approach_style=approach_style,
        )
        if approach_pitch_obj:
            app_note = note.Note()
            app_note.pitch.midi = self._get_bass_pitch_in_octave(
                approach_pitch_obj, target_octave
            )
            app_note.duration.quarterLength = approach_note_duration_ql
            app_note.volume.velocity = min(
                127, int(effective_velocity_for_approach * 0.85)
            )
            if original_last_note_tuple:
                orig_rel_offset, orig_note_obj_ref = original_last_note_tuple
                new_dur = approach_note_rel_offset - orig_rel_offset
                if new_dur >= MIN_NOTE_DURATION_QL / 2:
                    orig_note_obj_ref.duration.quarterLength = new_dur
                else:
                    self.logger.debug(
                        f"BassGen: Preceding note for approach became too short ({new_dur:.2f}ql). Skipping approach."
                    )
                    return notes_in_measure
            notes_in_measure.append((approach_note_rel_offset, app_note))
            notes_in_measure.sort(key=lambda x: x[0])
            self.logger.debug(
                f"BassGen: Inserted approach note {app_note.nameWithOctave} at {approach_note_rel_offset:.2f}"
            )
        return notes_in_measure

    def _generate_algorithmic_pattern(
        self,
        pattern_type: str,
        m21_cs: harmony.ChordSymbol,
        algo_pattern_options: dict[str, Any],
        initial_base_velocity: int,
        target_octave: int,
        block_offset_ignored: float,
        block_duration: float,
        current_scale: scale.ConcreteScale,
        next_chord_root: pitch.Pitch | None = None,
        # section_overrides_for_algo は self.overrides を直接参照するため引数から削除
    ) -> list[tuple[float, note.Note]]:
        notes_tuples: list[tuple[float, note.Note]] = []
        if not m21_cs or not m21_cs.pitches:
            return notes_tuples
        root_note_obj = m21_cs.root()
        if not root_note_obj:
            return notes_tuples

        # ベロシティ決定 (self.overrides を考慮)
        effective_base_velocity_candidate = initial_base_velocity
        # self.overrides (PartOverride) から velocity や velocity_shift を取得
        override_velocity_val = (
            self.overrides.velocity
            if self.overrides and self.overrides.velocity is not None
            else None
        )
        override_velocity_shift_val = (
            self.overrides.velocity_shift
            if self.overrides and self.overrides.velocity_shift is not None
            else None
        )

        if override_velocity_val is not None:
            effective_base_velocity_candidate = override_velocity_val
        elif override_velocity_shift_val is not None:
            base_for_shift = (
                initial_base_velocity
                if initial_base_velocity is not None
                else algo_pattern_options.get("velocity_base", 70)
            )
            effective_base_velocity_candidate = (
                base_for_shift + override_velocity_shift_val
            )
        else:
            effective_base_velocity_candidate = algo_pattern_options.get(
                "velocity_base",
                initial_base_velocity if initial_base_velocity is not None else 70,
            )

        if effective_base_velocity_candidate is None:
            effective_base_velocity_candidate = 70

        try:
            effective_base_velocity = int(effective_base_velocity_candidate)
            effective_base_velocity = max(1, min(127, effective_base_velocity))
        except (TypeError, ValueError) as e:
            self.logger.error(
                f"BassGen _generate_algorithmic_pattern: Error converting effective_base_velocity '{effective_base_velocity_candidate}' to int: {e}. Defaulting to 70."
            )
            effective_base_velocity = 70

        overall_velocity_factor = float(
            algo_pattern_options.get("velocity_factor", 1.0)
        )
        final_base_velocity_for_algo = max(
            1, min(127, int(effective_base_velocity * overall_velocity_factor))
        )

        # (以降のパターン生成ロジックは前回提示のものと同様。必要に応じてオプション活用を強化)
        if pattern_type == "algorithmic_chord_tone_quarters":
            strong_beat_vel_boost = safe_get(
                algo_pattern_options,
                "strong_beat_velocity_boost",
                default=10,
                cast_to=int,
            )
            off_beat_vel_reduction = safe_get(
                algo_pattern_options,
                "off_beat_velocity_reduction",
                default=8,
                cast_to=int,
            )
            weak_beat_style_final = algo_pattern_options.get("weak_beat_style", "root")
            approach_on_4th_final = algo_pattern_options.get(
                "approach_on_4th_beat", True
            )
            approach_style_final = algo_pattern_options.get(
                "approach_style_on_4th", "diatonic_or_chromatic"
            )
            beats_per_measure = (
                self.global_time_signature_obj.beatCount
                if self.global_time_signature_obj
                else 4
            )
            beat_duration_ql = (
                self.global_time_signature_obj.beatDuration.quarterLength
                if self.global_time_signature_obj
                else 1.0
            )
            num_measures_in_block = (
                math.ceil(block_duration / self.measure_duration)
                if self.measure_duration > 0
                else 1
            )
            for measure_idx in range(num_measures_in_block):
                measure_offset = measure_idx * self.measure_duration
                measure_notes_raw: list[tuple[float, note.Note]] = []
                remaining_in_section = block_duration - measure_offset
                approach_on_4th_this = approach_on_4th_final or (
                    remaining_in_section <= self.measure_duration * 2
                )
                root_for_measure = root_note_obj
                target_for_approach = next_chord_root
                if (
                    approach_style_final == "subdom_dom"
                    and remaining_in_section <= self.measure_duration * 2
                ):
                    if remaining_in_section > self.measure_duration:
                        try:
                            root_for_measure = current_scale.pitchFromDegree(2)
                        except Exception:
                            root_for_measure = root_note_obj
                        try:
                            target_for_approach = current_scale.pitchFromDegree(5)
                        except Exception:
                            target_for_approach = next_chord_root
                    else:
                        try:
                            root_for_measure = current_scale.pitchFromDegree(5)
                        except Exception:
                            root_for_measure = root_note_obj
                        target_for_approach = next_chord_root
                for beat_idx in range(beats_per_measure):
                    current_rel_offset_in_measure = beat_idx * beat_duration_ql
                    abs_offset_in_block_current_note = (
                        measure_offset + current_rel_offset_in_measure
                    )
                    if abs_offset_in_block_current_note >= block_duration - (
                        MIN_NOTE_DURATION_QL / 16.0
                    ):
                        break
                    chosen_pitch_base: pitch.Pitch | None = None
                    current_note_velocity = final_base_velocity_for_algo
                    note_duration_ql = beat_duration_ql
                    if beat_idx == 0:
                        chosen_pitch_base = root_for_measure
                        current_note_velocity = min(
                            127, final_base_velocity_for_algo + strong_beat_vel_boost
                        )
                    elif beats_per_measure >= 4 and beat_idx == (
                        beats_per_measure // 2
                    ):
                        chosen_pitch_base = (
                            m21_cs.fifth
                            if m21_cs.fifth
                            else (m21_cs.third if m21_cs.third else root_note_obj)
                        )
                        current_note_velocity = min(
                            127,
                            final_base_velocity_for_algo + (strong_beat_vel_boost // 2),
                        )
                    else:
                        chosen_pitch_base = root_for_measure
                        current_note_velocity = max(
                            1, final_base_velocity_for_algo - off_beat_vel_reduction
                        )
                    if chosen_pitch_base:
                        remaining_block_time_from_note_start = (
                            block_duration - abs_offset_in_block_current_note
                        )
                        actual_note_duration = min(
                            note_duration_ql, remaining_block_time_from_note_start
                        )
                        if actual_note_duration < MIN_NOTE_DURATION_QL:
                            continue
                        midi_val = self._get_bass_pitch_in_octave(
                            chosen_pitch_base, target_octave
                        )
                        n_obj = note.Note(
                            pitch.Pitch(midi=midi_val),
                            quarterLength=actual_note_duration,
                        )
                        n_obj.volume.velocity = current_note_velocity
                        measure_notes_raw.append((current_rel_offset_in_measure, n_obj))
                processed_measure_notes = self._apply_weak_beat(
                    measure_notes_raw, weak_beat_style_final
                )
                is_last_measure_in_block = measure_idx == num_measures_in_block - 1
                if approach_on_4th_this and target_for_approach:
                    processed_measure_notes = self._insert_approach_note_to_measure(
                        processed_measure_notes,
                        m21_cs,
                        target_for_approach,
                        current_scale,
                        approach_style_final,
                        target_octave,
                        final_base_velocity_for_algo,
                    )
                for rel_offset_in_measure, note_obj_final in processed_measure_notes:
                    abs_offset_in_block = measure_offset + rel_offset_in_measure
                    if abs_offset_in_block < block_duration:
                        notes_tuples.append((abs_offset_in_block, note_obj_final))
        elif pattern_type == "algorithmic_root_only":
            note_duration_ql = safe_get(
                algo_pattern_options,
                "note_duration_ql",
                default=block_duration,
                cast_to=float,
            )
            if note_duration_ql <= 0:
                note_duration_ql = block_duration
            num_notes = (
                int(block_duration / note_duration_ql) if note_duration_ql > 0 else 0
            )
            for i in range(num_notes):
                current_rel_offset = i * note_duration_ql
                if current_rel_offset >= block_duration - (MIN_NOTE_DURATION_QL / 16.0):
                    break
                actual_dur = min(note_duration_ql, block_duration - current_rel_offset)
                if actual_dur < MIN_NOTE_DURATION_QL:
                    continue
                midi_val = self._get_bass_pitch_in_octave(root_note_obj, target_octave)
                n_obj = note.Note(pitch.Pitch(midi=midi_val), quarterLength=actual_dur)
                n_obj.volume.velocity = final_base_velocity_for_algo
                notes_tuples.append((current_rel_offset, n_obj))
        elif pattern_type == "algorithmic_root_fifth":
            self.logger.info(
                f"BassGen: Generating algorithmic_root_fifth for {m21_cs.figure} with options {algo_pattern_options}"
            )
            beat_q_len = safe_get(
                algo_pattern_options, "beat_duration_ql", default=1.0, cast_to=float
            )
            arrangement = algo_pattern_options.get("arrangement", ["R", "5", "R", "5"])
            if beat_q_len <= 0:
                beat_q_len = 1.0
            num_steps_in_arrangement = len(arrangement)
            if num_steps_in_arrangement == 0:
                arrangement = ["R"]
                num_steps_in_arrangement = 1
            current_block_pos_ql = 0.0
            arrangement_idx = 0
            while current_block_pos_ql < block_duration - (MIN_NOTE_DURATION_QL / 16.0):
                note_type_char = arrangement[arrangement_idx % num_steps_in_arrangement]
                chosen_pitch_for_step: pitch.Pitch | None = None
                if note_type_char == "R":
                    chosen_pitch_for_step = root_note_obj
                elif note_type_char == "5":
                    chosen_pitch_for_step = m21_cs.fifth
                    if not chosen_pitch_for_step:
                        chosen_pitch_for_step = root_note_obj.transpose(7)
                elif note_type_char == "3":
                    chosen_pitch_for_step = m21_cs.third
                    if not chosen_pitch_for_step:
                        chosen_pitch_for_step = root_note_obj.transpose(
                            4 if m21_cs.quality == "major" else 3
                        )
                else:
                    chosen_pitch_for_step = root_note_obj
                if chosen_pitch_for_step:
                    actual_dur = min(beat_q_len, block_duration - current_block_pos_ql)
                    if actual_dur < MIN_NOTE_DURATION_QL:
                        break
                    midi_val = self._get_bass_pitch_in_octave(
                        chosen_pitch_for_step, target_octave
                    )
                    n_obj = note.Note(
                        pitch.Pitch(midi=midi_val), quarterLength=actual_dur
                    )
                    n_obj.volume.velocity = final_base_velocity_for_algo
                    notes_tuples.append((current_block_pos_ql, n_obj))
                current_block_pos_ql += beat_q_len
                arrangement_idx += 1
                if beat_q_len <= 0:
                    break
            if not notes_tuples and root_note_obj:
                midi_val = self._get_bass_pitch_in_octave(root_note_obj, target_octave)
                n_obj = note.Note(
                    pitch.Pitch(midi=midi_val), quarterLength=block_duration
                )
                n_obj.volume.velocity = final_base_velocity_for_algo
                notes_tuples.append((0.0, n_obj))
        elif pattern_type in [
            "algorithmic_walking",
            "algorithmic_walking_8ths",
            "walking",
            "walking_8ths",
        ]:
            self.logger.info(
                f"BassGen: Generating {pattern_type} for {m21_cs.figure} with options {algo_pattern_options}"
            )
            step_ql = safe_get(
                algo_pattern_options,
                "step_ql",
                default=(
                    0.5
                    if "8ths" in pattern_type or "walking_8ths" in pattern_type
                    else 1.0
                ),
                cast_to=float,
            )
            if step_ql <= 0:
                step_ql = (
                    0.5
                    if "8ths" in pattern_type or "walking_8ths" in pattern_type
                    else 1.0
                )
            approach_style = algo_pattern_options.get(
                "approach_style", "diatonic_or_chromatic"
            )
            approach_prob = safe_get(
                algo_pattern_options, "approach_note_prob", default=0.5, cast_to=float
            )
            num_steps = int(block_duration / step_ql) if step_ql > 0 else 0
            last_pitch_obj = root_note_obj
            for i in range(num_steps):
                current_rel_offset = i * step_ql
                if current_rel_offset >= block_duration - (MIN_NOTE_DURATION_QL / 16.0):
                    break
                actual_dur = min(step_ql, block_duration - current_rel_offset)
                if actual_dur < MIN_NOTE_DURATION_QL:
                    continue
                chosen_pitch_for_step: pitch.Pitch | None = None
                is_last_step_before_next_chord_block = (
                    i + 1
                ) * step_ql >= block_duration - (step_ql / 2.0)
                if (
                    is_last_step_before_next_chord_block
                    and next_chord_root
                    and self._rng.random() < approach_prob
                ):
                    chosen_pitch_for_step = get_approach_note(
                        last_pitch_obj, next_chord_root, current_scale, approach_style
                    )
                    if chosen_pitch_for_step:
                        self.logger.debug(
                            f"  Walk: Approaching next chord with {chosen_pitch_for_step.nameWithOctave}"
                        )
                if not chosen_pitch_for_step:
                    if i == 0:
                        chosen_pitch_for_step = root_note_obj
                    else:
                        direction_choice = self._rng.choice(
                            [
                                DIRECTION_UP,
                                DIRECTION_DOWN,
                                DIRECTION_UP,
                                DIRECTION_DOWN,
                                0,
                            ]
                        )
                        next_candidate: pitch.Pitch | None = None
                        if direction_choice != 0:
                            try:
                                next_candidate = current_scale.nextPitch(
                                    last_pitch_obj, direction=direction_choice
                                )
                            except Exception:
                                next_candidate = None
                        if (
                            next_candidate
                            and abs(next_candidate.ps - last_pitch_obj.ps) <= 7
                        ):
                            chosen_pitch_for_step = next_candidate
                        else:
                            available_tones = [
                                t
                                for t in [m21_cs.root(), m21_cs.third, m21_cs.fifth]
                                if t and t.name != last_pitch_obj.name
                            ]
                            if not available_tones:
                                available_tones = [
                                    t
                                    for t in [m21_cs.root(), m21_cs.third, m21_cs.fifth]
                                    if t
                                ]
                            chosen_pitch_for_step = (
                                self._rng.choice(available_tones)
                                if available_tones
                                else root_note_obj
                            )
                if chosen_pitch_for_step:
                    last_pitch_obj = chosen_pitch_for_step
                    midi_val = self._get_bass_pitch_in_octave(
                        chosen_pitch_for_step, target_octave
                    )
                    n_obj = note.Note(
                        pitch.Pitch(midi=midi_val), quarterLength=actual_dur
                    )
                    n_obj.volume.velocity = final_base_velocity_for_algo
                    notes_tuples.append((current_rel_offset, n_obj))
        elif pattern_type == "walking_quarters":
            self.logger.info(
                f"BassGen: Generating walking_quarters for {m21_cs.figure}"
            )
            beats = int(block_duration)
            if beats <= 0:
                beats = 4
            beat_len = block_duration / beats
            half = beat_len / 2.0
            scale_tones = [p for p in [m21_cs.third, m21_cs.fifth] if p]
            next_root_pitch = next_chord_root if next_chord_root else root_note_obj
            for i in range(beats):
                off = i * beat_len
                root_pitch = m21_cs.root() if m21_cs.root() else root_note_obj
                midi_root = self._get_bass_pitch_in_octave(root_pitch, target_octave)
                n_root = note.Note(pitch.Pitch(midi=midi_root), quarterLength=half)
                n_root.volume.velocity = final_base_velocity_for_algo
                notes_tuples.append((off, n_root))
                target = next_root_pitch if i == beats - 1 else root_pitch
                walk_p = bass_utils.get_walking_note(n_root.pitch, target, scale_tones)
                midi_walk = self._get_bass_pitch_in_octave(walk_p, target_octave)
                n_walk = note.Note(pitch.Pitch(midi=midi_walk), quarterLength=half)
                n_walk.volume.velocity = final_base_velocity_for_algo
                notes_tuples.append((off + half, n_walk))
        elif pattern_type == "algorithmic_pedal":
            self.logger.info(
                f"BassGen: Generating algorithmic_pedal for {m21_cs.figure} with options {algo_pattern_options}"
            )
            note_duration_ql = safe_get(
                algo_pattern_options,
                "note_duration_ql",
                default=block_duration,
                cast_to=float,
            )
            subdivision_ql = safe_get(
                algo_pattern_options,
                "subdivision_ql",
                default=note_duration_ql,
                cast_to=float,
            )
            pedal_note_type = safe_get(
                algo_pattern_options, "pedal_note_type", default="root", cast_to=str
            ).lower()
            if note_duration_ql <= 0:
                note_duration_ql = block_duration
            if subdivision_ql <= 0:
                subdivision_ql = note_duration_ql
            pedal_pitch_obj = root_note_obj
            if pedal_note_type == "fifth" and m21_cs.fifth:
                pedal_pitch_obj = m21_cs.fifth
            elif pedal_note_type == "third" and m21_cs.third:
                pedal_pitch_obj = m21_cs.third
            elif pedal_note_type == "bass" and m21_cs.bass():
                pedal_pitch_obj = m21_cs.bass()
            current_block_pos_ql = 0.0
            while current_block_pos_ql < block_duration - (MIN_NOTE_DURATION_QL / 16.0):
                num_subdivisions_in_this_note = (
                    int(note_duration_ql / subdivision_ql) if subdivision_ql > 0 else 1
                )
                for i_sub in range(num_subdivisions_in_this_note):
                    current_rel_offset = current_block_pos_ql + (i_sub * subdivision_ql)
                    if current_rel_offset >= block_duration - (
                        MIN_NOTE_DURATION_QL / 16.0
                    ):
                        break
                    actual_sub_duration = min(
                        subdivision_ql,
                        block_duration - current_rel_offset,
                        note_duration_ql - (i_sub * subdivision_ql),
                    )
                    if actual_sub_duration < MIN_NOTE_DURATION_QL:
                        continue
                    midi_val = self._get_bass_pitch_in_octave(
                        pedal_pitch_obj, target_octave
                    )
                    n_obj = note.Note(
                        pitch.Pitch(midi=midi_val), quarterLength=actual_sub_duration
                    )
                    n_obj.volume.velocity = final_base_velocity_for_algo
                    notes_tuples.append((current_rel_offset, n_obj))
                current_block_pos_ql += note_duration_ql
                if note_duration_ql <= 0:
                    break
        elif pattern_type == "explicit":
            self.logger.info(
                f"BassGen: Generating explicit pattern for {m21_cs.figure} from options {algo_pattern_options}"
            )
            pattern_list_explicit = algo_pattern_options.get("pattern", [])
            ref_dur_explicit = safe_get(
                algo_pattern_options,
                "reference_duration_ql",
                default=self.measure_duration,
                cast_to=float,
            )
            if not pattern_list_explicit:
                self.logger.warning(
                    f"BassGen: Explicit pattern for {m21_cs.figure} is empty. Falling back to root_only."
                )
                return self._generate_algorithmic_pattern(
                    "algorithmic_root_only",
                    m21_cs,
                    self.rhythm_lib.get("root_only", {}).get("options", {}),
                    initial_base_velocity,
                    target_octave,
                    0,
                    block_duration,
                    current_scale,
                    next_chord_root,
                )
            notes_tuples = self._generate_notes_from_fixed_pattern(
                pattern_list_explicit,
                m21_cs,
                final_base_velocity_for_algo,
                target_octave,
                block_duration,
                ref_dur_explicit,
                current_scale,
                velocity_curve_list,
            )
        elif pattern_type == "funk_octave_pops":
            self.logger.info(
                f"BassGen: Generating funk_octave_pops for {m21_cs.figure} with options {algo_pattern_options}"
            )
            base_rhythm_ql = safe_get(
                algo_pattern_options, "base_rhythm_ql", default=0.25, cast_to=float
            )
            accent_factor = safe_get(
                algo_pattern_options, "accent_factor", default=1.2, cast_to=float
            )
            ghost_factor = safe_get(
                algo_pattern_options, "ghost_factor", default=0.5, cast_to=float
            )
            syncopation_prob = safe_get(
                algo_pattern_options, "syncopation_prob", default=0.3, cast_to=float
            )
            octave_jump_prob = safe_get(
                algo_pattern_options, "octave_jump_prob", default=0.6, cast_to=float
            )
            if base_rhythm_ql <= 0:
                base_rhythm_ql = 0.25
            num_steps = int(block_duration / base_rhythm_ql)
            for i in range(num_steps):
                current_rel_offset = i * base_rhythm_ql
                if current_rel_offset >= block_duration - (MIN_NOTE_DURATION_QL / 16.0):
                    break
                actual_dur = min(base_rhythm_ql, block_duration - current_rel_offset)
                if actual_dur < MIN_NOTE_DURATION_QL:
                    continue
                chosen_pitch_for_step = root_note_obj
                current_velocity = final_base_velocity_for_algo
                if self._rng.random() < octave_jump_prob:
                    chosen_pitch_for_step = root_note_obj.transpose(12)
                beat_pos_in_measure_ql = current_rel_offset % self.measure_duration
                beat_unit_ql = (
                    self.global_time_signature_obj.beatDuration.quarterLength
                    if self.global_time_signature_obj
                    else 1.0
                )
                is_on_beat = abs(beat_pos_in_measure_ql % beat_unit_ql) < (
                    MIN_NOTE_DURATION_QL / 4.0
                )
                is_eighth_offbeat = (
                    abs(beat_pos_in_measure_ql % (beat_unit_ql / 2.0))
                    < (MIN_NOTE_DURATION_QL / 4.0)
                    and not is_on_beat
                )
                if is_on_beat:
                    current_velocity = int(final_base_velocity_for_algo * accent_factor)
                elif is_eighth_offbeat:
                    current_velocity = int(
                        final_base_velocity_for_algo * ghost_factor * 1.1
                    )
                else:
                    current_velocity = int(final_base_velocity_for_algo * ghost_factor)
                final_velocity = max(1, min(127, current_velocity))
                if chosen_pitch_for_step:
                    midi_val = self._get_bass_pitch_in_octave(
                        chosen_pitch_for_step, target_octave
                    )
                    n_obj = note.Note(
                        pitch.Pitch(midi=midi_val), quarterLength=actual_dur
                    )
                    n_obj.volume.velocity = final_velocity
                    notes_tuples.append((current_rel_offset, n_obj))
            if not notes_tuples and root_note_obj:
                midi_val = self._get_bass_pitch_in_octave(root_note_obj, target_octave)
                n_obj = note.Note(
                    pitch.Pitch(midi=midi_val), quarterLength=block_duration
                )
                n_obj.volume.velocity = final_base_velocity_for_algo
                notes_tuples.append((0.0, n_obj))
        elif pattern_type == "half_time_pop":
            self.logger.info(
                f"BassGen: Generating half_time_pop for {m21_cs.figure} with options {algo_pattern_options}"
            )
            ghost = algo_pattern_options.get("ghost_on_beat_2_and_a_half", False)
            ghost_ratio = safe_get(
                algo_pattern_options,
                "ghost_velocity_ratio",
                default=0.5,
                cast_to=float,
            )
            pattern_events = [
                (0.0, 2.0, root_note_obj, 1.0),
                (2.0, 1.5, root_note_obj, 1.0),
            ]
            if ghost:
                pattern_events.append((2.5, 0.5, root_note_obj, ghost_ratio))
            anticipation_pitch = next_chord_root if next_chord_root else root_note_obj
            pattern_events.append((block_duration - 0.5, 0.5, anticipation_pitch, 1.0))
            for rel_offset, dur, pitch_obj, vel_factor in pattern_events:
                if rel_offset >= block_duration - (MIN_NOTE_DURATION_QL / 16.0):
                    continue
                dur = min(dur, block_duration - rel_offset)
                if dur < MIN_NOTE_DURATION_QL:
                    continue
                midi_val = self._get_bass_pitch_in_octave(pitch_obj, target_octave)
                n_obj = note.Note(pitch.Pitch(midi=midi_val), quarterLength=dur)
                n_obj.volume.velocity = max(
                    1, int(final_base_velocity_for_algo * vel_factor)
                )
                notes_tuples.append((rel_offset, n_obj))
        elif pattern_type in [
            "walking_blues",
            "latin_tumbao",
            "syncopated_rnb",
            "scale_walk",
            "octave_jump",
            "descending_fifths",
            "pedal_tone",
        ]:
            self.logger.warning(
                f"BassGenerator: Algorithmic pattern_type '{pattern_type}' is defined in library but not yet implemented. Falling back to 'algorithmic_chord_tone_quarters'."
            )
            default_algo_options = self.part_parameters.get(
                "basic_chord_tone_quarters", {}
            ).get("options", {})
            notes_tuples.extend(
                self._generate_algorithmic_pattern(
                    "algorithmic_chord_tone_quarters",
                    m21_cs,
                    default_algo_options,
                    initial_base_velocity,
                    target_octave,
                    0.0,
                    block_duration,
                    current_scale,
                    next_chord_root,
                )
            )
        else:
            self.logger.warning(
                f"BassGenerator: Unknown algorithmic or unhandled pattern_type '{pattern_type}'. Falling back to 'algorithmic_chord_tone_quarters'."
            )
            default_algo_options = self.part_parameters.get(
                "basic_chord_tone_quarters", {}
            ).get("options", {})
            notes_tuples.extend(
                self._generate_algorithmic_pattern(
                    "algorithmic_chord_tone_quarters",
                    m21_cs,
                    default_algo_options,
                    initial_base_velocity,
                    target_octave,
                    0.0,
                    block_duration,
                    current_scale,
                    next_chord_root,
                )
            )
        return notes_tuples

    def _render_part(
        self,
        section_data: dict[str, Any],
        next_section_data: dict[str, Any] | None = None,
    ) -> stream.Part:
        bass_part = stream.Part(id=self.part_name)
        actual_instrument = copy.deepcopy(self.default_instrument)
        if not actual_instrument.partName:
            actual_instrument.partName = self.part_name.capitalize()
        if not actual_instrument.partAbbreviation:
            actual_instrument.partAbbreviation = self.part_name[:3].capitalize() + "."
        bass_part.insert(0, actual_instrument)

        log_blk_prefix = f"BassGen._render_part (Section: {section_data.get('section_name', 'Unknown')})"

        bass_params_from_chordmap = section_data.get("part_params", {}).get("bass", {})
        final_bass_params = bass_params_from_chordmap.copy()
        final_bass_params.setdefault("options", {})

        if self.overrides and hasattr(self.overrides, "model_dump"):
            override_dict = self.overrides.model_dump(exclude_unset=True)
            if not isinstance(final_bass_params.get("options"), dict):
                final_bass_params["options"] = {}
            chordmap_options = final_bass_params.get("options", {})
            override_options = override_dict.pop("options", None)
            if isinstance(override_options, dict):
                merged_options = chordmap_options.copy()
                merged_options.update(override_options)
                final_bass_params["options"] = merged_options
            final_bass_params.update(override_dict)

        block_musical_intent = section_data.get("musical_intent", {})
        rhythm_key_from_params = final_bass_params.get(
            "rhythm_key", final_bass_params.get("style")
        )
        if not rhythm_key_from_params:
            rhythm_key_from_params = self._choose_bass_pattern_key(block_musical_intent)

        pattern_details = self._get_rhythm_pattern_details(rhythm_key_from_params)
        actual_rhythm_key_used = rhythm_key_from_params
        if rhythm_key_from_params not in self.part_parameters:  # ←修正
            actual_rhythm_key_used = "basic_chord_tone_quarters"
        final_bass_params["rhythm_key"] = actual_rhythm_key_used

        if not pattern_details:
            self.logger.warning(
                f"{log_blk_prefix}: No pattern_details for '{actual_rhythm_key_used}'. Skipping block."
            )
            return bass_part

        velocity_curve_list = resolve_velocity_curve(
            pattern_details.get("options", {}).get("velocity_curve")
        )

        block_q_length = section_data.get("q_length", self.measure_duration)
        if block_q_length <= 0:
            block_q_length = self.measure_duration

        chord_label_str = section_data.get("chord_symbol_for_voicing", "C")
        m21_cs_obj: harmony.ChordSymbol | None = None
        sanitized_label = sanitize_chord_label(chord_label_str)
        final_bass_str_for_set: str | None = None

        if sanitized_label and sanitized_label.lower() != "rest":
            try:
                m21_cs_obj = harmony.ChordSymbol(sanitized_label)
                specified_bass_str = section_data.get("specified_bass_for_voicing")
                if specified_bass_str:
                    final_bass_str_for_set = sanitize_chord_label(specified_bass_str)
                    if (
                        final_bass_str_for_set
                        and final_bass_str_for_set.lower() != "rest"
                    ):
                        m21_cs_obj.bass(final_bass_str_for_set)
            except harmony.ChordException as e_bass:
                self.logger.warning(
                    f"{log_blk_prefix}: Error setting bass '{final_bass_str_for_set}' for chord '{sanitized_label}': {e_bass}."
                )
            except Exception as e_chord_parse:
                self.logger.error(
                    f"{log_blk_prefix}: Error parsing chord '{sanitized_label}': {e_chord_parse}. Skipping."
                )
                return bass_part
        elif sanitized_label and sanitized_label.lower() == "rest":
            self.logger.info(f"{log_blk_prefix}: Block is Rest.")
            return bass_part

        if not m21_cs_obj:
            self.logger.warning(
                f"{log_blk_prefix}: Chord '{chord_label_str}' invalid. Skipping."
            )
            return bass_part

        base_vel = safe_get(
            final_bass_params,
            "velocity",
            default=safe_get(
                pattern_details.get("options", {}),
                "velocity_base",
                default=70,
                cast_to=int,
            ),
        )
        base_vel = max(1, min(127, base_vel))
        target_oct = safe_get(
            final_bass_params,
            "octave",
            default=safe_get(
                pattern_details.get("options", {}),
                "target_octave",
                default=2,
                cast_to=int,
            ),
        )

        section_tonic = section_data.get("tonic_of_section", self.global_key_tonic)
        section_mode = section_data.get("mode", self.global_key_mode)
        current_m21_scale = ScaleRegistry.get(section_tonic, section_mode)
        if not current_m21_scale:
            current_m21_scale = scale.MajorScale(
                self.global_key_tonic if self.global_key_tonic else "C"
            )

        next_chord_root_pitch: pitch.Pitch | None = None
        if next_section_data:
            next_chord_label_str = next_section_data.get(
                "chord_symbol_for_voicing",
                next_section_data.get("original_chord_label"),
            )
            if next_chord_label_str:
                next_sanitized_label = sanitize_chord_label(next_chord_label_str)
                if next_sanitized_label and next_sanitized_label.lower() != "rest":
                    try:
                        next_cs_obj_temp = harmony.ChordSymbol(next_sanitized_label)
                        next_specified_bass = next_section_data.get(
                            "specified_bass_for_voicing"
                        )
                        if next_specified_bass:
                            final_next_bass_str: str | None = None
                            final_next_bass_str = sanitize_chord_label(
                                next_specified_bass
                            )
                            if (
                                final_next_bass_str
                                and final_next_bass_str.lower() != "rest"
                            ):
                                next_cs_obj_temp.bass(final_next_bass_str)
                        if next_cs_obj_temp and next_cs_obj_temp.root():
                            next_chord_root_pitch = next_cs_obj_temp.root()
                    except Exception:
                        pass

        generated_notes_for_block: list[tuple[float, note.Note]] = []
        pattern_type_from_lib = pattern_details.get("pattern_type")
        if not pattern_type_from_lib:
            pattern_type_from_lib = "fixed_pattern"

        merged_algo_options = pattern_details.get("options", {}).copy()
        if isinstance(final_bass_params.get("options"), dict):
            merged_algo_options.update(final_bass_params["options"])
        merged_algo_options["velocity_factor"] = final_bass_params.get(
            "velocity_factor", merged_algo_options.get("velocity_factor", 1.0)
        )

        if "algorithmic_" in pattern_type_from_lib or pattern_type_from_lib in [
            "walking",
            "walking_8ths",
            "explicit",
            "root_fifth",
            "funk_octave_pops",
            "explicit",
            "root_fifth",
            "funk_octave_pops",
            "walking_blues",
            "latin_tumbao",
            "half_time_pop",
            "syncopated_rnb",
            "scale_walk",
            "octave_jump",
            "descending_fifths",
            "pedal_tone",
        ]:
            generated_notes_for_block = self._generate_algorithmic_pattern(
                pattern_type_from_lib,
                m21_cs_obj,
                merged_algo_options,
                base_vel,
                target_oct,
                0.0,
                block_q_length,
                current_m21_scale,
                next_chord_root_pitch,
            )
        elif (
            pattern_type_from_lib == "fixed_pattern"
            and "pattern" in pattern_details
            and isinstance(pattern_details["pattern"], list)
        ):
            ref_dur_fixed = safe_get(
                pattern_details,
                "reference_duration_ql",
                default=self.measure_duration,
                cast_to=float,
            )
            if ref_dur_fixed <= 0:
                ref_dur_fixed = self.measure_duration
            generated_notes_for_block = self._generate_notes_from_fixed_pattern(
                pattern_details["pattern"],
                m21_cs_obj,
                base_vel,
                target_oct,
                block_q_length,
                ref_dur_fixed,
                current_m21_scale,
                velocity_curve_list,
            )
        else:
            self.logger.warning(
                f"{log_blk_prefix}: Pattern '{final_bass_params['rhythm_key']}' type '{pattern_type_from_lib}' not handled or missing 'pattern' list. Using fallback 'basic_chord_tone_quarters'."
            )
            fallback_options = self.part_parameters.get(
                "basic_chord_tone_quarters", {}
            ).get("options", {})
            generated_notes_for_block = self._generate_algorithmic_pattern(
                "algorithmic_chord_tone_quarters",
                m21_cs_obj,
                fallback_options,
                base_vel,
                target_oct,
                0.0,
                block_q_length,
                current_m21_scale,
                next_chord_root_pitch,
            )

        for rel_offset, note_obj_to_add in generated_notes_for_block:
            current_note_abs_offset_in_block = rel_offset
            end_of_note_in_block = (
                current_note_abs_offset_in_block
                + note_obj_to_add.duration.quarterLength
            )
            if end_of_note_in_block > block_q_length + 0.001:
                new_dur_for_note = block_q_length - current_note_abs_offset_in_block
                if new_dur_for_note >= MIN_NOTE_DURATION_QL / 2.0:
                    note_obj_to_add.duration.quarterLength = new_dur_for_note
                else:
                    self.logger.debug(
                        f"{log_blk_prefix}: Note at {current_note_abs_offset_in_block:.2f} for {m21_cs_obj.figure} became too short after clipping to block_q_length. Skipping."
                    )
                    continue
            if note_obj_to_add.duration.quarterLength >= MIN_NOTE_DURATION_QL / 2.0:
                bass_part.insert(current_note_abs_offset_in_block, note_obj_to_add)
            else:
                self.logger.debug(
                    f"{log_blk_prefix}: Final note for {m21_cs_obj.figure} at {current_note_abs_offset_in_block:.2f} too short ({note_obj_to_add.duration.quarterLength:.3f}ql). Skipping."
                )
        profile_name = (
            self.cfg.get("humanize_profile")
            or section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        if profile_name:
            humanizer.apply(
                bass_part,
                profile_name,
                global_settings=self.global_settings,
            )

        return bass_part

    def get_root_offsets(self) -> list[float]:
        """Return offsets of notes generated in the last compose call."""
        return list(self._root_offsets)

    # --------------------------- Phrase Utilities ---------------------------
    def load_phrase_templates(self, path: Path) -> None:
        """Load custom phrase templates from YAML or JSON."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.custom_phrases = data.get("phrases", {})

    def set_phrase_insertions(self, mapping: dict[str, str]) -> None:
        """Map section labels to phrase template names."""
        self.phrase_insertions = dict(mapping)

    def set_phrase_map(self, mapping: dict[str, dict]) -> None:
        self.phrase_map = dict(mapping)

    def _apply_phrase_dynamics(self, part: stream.Part, intensity: float) -> None:
        scale = 1.0 + intensity * self.intensity_scale
        for n in part.flatten().notes:
            if n.volume is None:
                n.volume = m21volume.Volume(velocity=self.base_velocity)
            n.volume.velocity = max(1, min(127, int(n.volume.velocity * scale)))

    def _insert_custom_phrase(self, part: stream.Part, template_name: str) -> None:
        tpl = self.custom_phrases.get(template_name)
        if not tpl:
            return
        pattern = tpl.get("pattern", [])
        velocities = tpl.get("velocities", [])
        for idx, spec in enumerate(pattern):
            try:
                off, midi, dur = spec
            except Exception:
                continue
            vel = int(velocities[idx]) if idx < len(velocities) else self.base_velocity
            n = note.Note()
            n.pitch.midi = int(midi)
            n.duration = m21duration.Duration(float(dur))
            n.volume = m21volume.Volume(velocity=vel)
            part.insert(float(off), n)

    # ------------------------------------------------------------------
    # Kick-Lock → Mirror-Melody simplified rendering
    # ------------------------------------------------------------------
    def render_part(
        self,
        section_data: dict[str, Any],
        *,
        next_section_data: dict[str, Any] | None = None,
    ) -> stream.Part:
        """Render a short bass riff following Kick-Lock then Mirror-Melody."""

        emotion = section_data.get("emotion", "default")
        key_signature = section_data.get("key_signature", "C")
        chord_label = section_data.get("chord", key_signature)
        groove_kicks = list(section_data.get("groove_kicks", []))
        melody = list(section_data.get("melody", []))

        def _clamp_pitch_octaves(p_obj: pitch.Pitch) -> pitch.Pitch:
            while p_obj.midi < self.bass_range_lo:
                p_obj = p_obj.transpose(12)
            while p_obj.midi > self.bass_range_hi:
                p_obj = p_obj.transpose(-12)
            return p_obj

        def _degree_to_pitch(base: pitch.Pitch, token: str | int) -> pitch.Pitch:
            semis = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
            shift = 0
            if isinstance(token, str):
                s = token
                if s.startswith("b"):
                    shift = -1
                    s = s[1:]
                elif s.startswith("#"):
                    shift = 1
                    s = s[1:]
                try:
                    deg = int(s)
                except Exception:
                    deg = 1
            else:
                deg = int(token)
            interval_semi = semis.get(deg, 0) + shift
            p_new = base.transpose(interval_semi)
            return _clamp_pitch_octaves(p_new)

        part = stream.Part(id=self.part_name)
        part.insert(0, copy.deepcopy(self.default_instrument))

        try:
            cs = harmony.ChordSymbol(chord_label)
            root_pitch = cs.root() or pitch.Pitch(key_signature)
        except Exception:
            root_pitch = pitch.Pitch(key_signature)

        root_pitch = _clamp_pitch_octaves(root_pitch)

        tick = 1 / 480
        first_kick = None
        for k in groove_kicks:
            if 0 <= float(k) < 0.5:
                first_kick = float(k)
                break
        first_offset = float(first_kick) if first_kick is not None else 0.0
        first_offset += self._rng.uniform(-tick, tick)

        emotion_data = self.emotion_profile.get(emotion, {})
        pattern_defs = emotion_data.get("bass_patterns", [])
        pat = pattern_defs[0] if pattern_defs else {}
        riff = pat.get("riff", [1, 5, 1, 5])
        velocity_layer = pat.get("velocity", "mid") or "mid"
        swing_val = pat.get("swing", "off")
        swing_flag = bool(swing_val) if isinstance(swing_val, bool) else str(swing_val).lower() == "on"
        base_velocity = AccentMapper.map_layer(velocity_layer, rng=self._rng)

        first_note = note.Note(root_pitch)
        first_note.duration = m21duration.Duration(1.0)
        first_note.volume = m21volume.Volume(velocity=base_velocity)

        notes_data: list[tuple[float, note.Note]] = [(first_offset, first_note)]


        for off, pitch_midi, dur in melody:
            if float(off) < 1.0:
                continue
            grid_off = round(float(off) / 0.5) * 0.5
            if grid_off >= self.measure_duration:
                continue
            try:
                m_pitch = pitch.Pitch()
                m_pitch.midi = int(pitch_midi)
            except Exception:
                continue
            intv = interval.Interval(root_pitch, m_pitch).chromatic.mod12
            mirrored = root_pitch.transpose(-intv)
            if mirrored.pitchClass == root_pitch.pitchClass:
                mirrored = root_pitch
            mirrored = _clamp_pitch_octaves(mirrored)
            bn = note.Note(mirrored)
            bn.duration = m21duration.Duration(float(dur))
            bn.volume = m21volume.Volume(
                velocity=AccentMapper.map_layer("low", rng=self._rng)
            )
            notes_data.append((grid_off, bn))

        offsets = [] if melody else [0.0, 1.0, 2.0, 3.0]
        ql_eighth = 0.5
        swing_amt = (self.swing_ratio - 0.5) * ql_eighth if swing_flag else 0.0
        if not melody:
            for idx, deg in enumerate(riff[1:], start=1):
                off = offsets[idx % len(offsets)]
                p_obj = _degree_to_pitch(root_pitch, deg)
                n_obj = note.Note(p_obj)
                n_obj.duration = m21duration.Duration(1.0)
                n_obj.volume = m21volume.Volume(
                    velocity=AccentMapper.map_layer(velocity_layer, rng=self._rng)
                )
                notes_data.append((off, n_obj))

        # --------------------------------------------------------------
        # ii-V build-up detection and generation (beats >= 3 only)
        # --------------------------------------------------------------
        key_tonic = (
            self.global_key_signature_tonic or self.global_key_tonic or "C"
        )
        key_mode = (
            self.global_key_signature_mode or self.global_key_mode or "major"
        )
        try:
            key_obj = key.Key(key_tonic, key_mode)
        except Exception:
            key_obj = key.Key(key_tonic)

        build_up = False
        next_label = None
        if next_section_data:
            next_label = next_section_data.get("chord") or next_section_data.get(
                "chord_symbol_for_voicing"
            )
        if next_label:
            try:
                next_root = harmony.ChordSymbol(next_label).root()
            except Exception:
                next_root = None
            if next_root and key_obj.getScaleDegreeFromPitch(next_root) == 1:
                build_up = True

        if build_up:
            deg_now = key_obj.getScaleDegreeFromPitch(root_pitch)
            pattern_ints: list[int]
            pattern_root = root_pitch
            if deg_now == 2:
                pattern_ints = [0, 3, 7, 9]
            elif deg_now == 5:
                pattern_ints = [0, 4, 7, 8]
            else:
                pattern_root = key_obj.pitchFromDegree(5)
                pattern_root.octave = root_pitch.octave
                pattern_ints = [0, 2, 4, 5]
            pattern_pitches = [pattern_root.transpose(i) for i in pattern_ints]

            build_offsets = [2.0, 3.0]
            for b_off, p_obj in zip(build_offsets, pattern_pitches[2:]):
                midi_val = p_obj.midi
                while midi_val < self.bass_range_lo:
                    midi_val += 12
                while midi_val > self.bass_range_hi:
                    midi_val -= 12
                p_obj.midi = midi_val
                notes_data = [d for d in notes_data if not (b_off <= d[0] < b_off + 1.0)]
                n_bu = note.Note(p_obj)
                n_bu.duration = m21duration.Duration(1.0)
                n_bu.volume = m21volume.Volume(
                    velocity=AccentMapper.map_layer("mid", rng=self._rng)
                )
                notes_data.append((b_off, n_bu))

        def _clamp_note_durations(
            notes: list[tuple[float, note.Note]],
        ) -> list[tuple[float, note.Note]]:
            notes.sort(key=lambda x: x[0])
            clamped: list[tuple[float, note.Note]] = []
            for off, n in notes:
                if clamped:
                    prev_off, prev_n = clamped[-1]
                    if off < prev_off + prev_n.duration.quarterLength:
                        prev_n.duration.quarterLength = max(
                            MIN_NOTE_DURATION_QL, off - prev_off
                        )
                clamped.append((off, n))
            if clamped:
                last_off, last_n = clamped[-1]
                last_n.duration.quarterLength = max(
                    MIN_NOTE_DURATION_QL,
                    min(last_n.duration.quarterLength, self.measure_duration - last_off),
                )
            return clamped

        merged = _clamp_note_durations(notes_data)

        humanize_opts = {
            opt.strip() for opt in str(section_data.get("humanize", "")).split(",") if opt
        }
        swung: list[tuple[float, note.Note]] = []
        for idx, (off, n) in enumerate(merged):
            insert_off = off
            if swing_amt and idx % 2 == 1:
                insert_off += swing_amt
                prev_off, prev_n = swung[-1]
                prev_n.duration.quarterLength = max(
                    MIN_NOTE_DURATION_QL, insert_off - prev_off
                )
            swung.append((insert_off, n))

        for i, (off, n) in enumerate(swung):
            if i < len(swung) - 1:
                next_off = swung[i + 1][0]
                n.duration.quarterLength = min(n.duration.quarterLength, next_off - off)
            else:
                n.duration.quarterLength = min(
                    n.duration.quarterLength,
                    max(MIN_NOTE_DURATION_QL, self.measure_duration - off),
                )
            part.insert(off, n)
            custom: dict[str, float] = {}
            if "vel" not in humanize_opts:
                custom["velocity_variation"] = 0.0
            if "micro" not in humanize_opts:
                custom["time_variation"] = 0.0
            humanizer.apply_humanization_to_element(n, custom_params=custom)

        part.coreElementsChanged()
        return part


# --- END OF FILE generator/bass_generator.py ---
