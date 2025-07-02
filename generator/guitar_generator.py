# --- START OF FILE generator/guitar_generator.py (BasePartGenerator継承・改修版) ---
import music21
from typing import List, Dict, Optional, Tuple, Any, Sequence, Union, cast
import copy
from pathlib import Path
import yaml
import json

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


logger = logging.getLogger("modular_composer.guitar_generator")

DEFAULT_GUITAR_OCTAVE_RANGE: Tuple[int, int] = (2, 5)
GUITAR_STRUM_DELAY_QL: float = 0.02
STANDARD_TUNING_OFFSETS = [0, 0, 0, 0, 0, 0]

EXEC_STYLE_BLOCK_CHORD = "block_chord"
EXEC_STYLE_STRUM_BASIC = "strum_basic"
EXEC_STYLE_ARPEGGIO_FROM_INDICES = "arpeggio_from_indices"
EXEC_STYLE_POWER_CHORDS = "power_chords"
EXEC_STYLE_MUTED_RHYTHM = "muted_rhythm"

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
        tuning: Optional[List[int]] = None,
        timing_variation: float = 0.0,
        gate_length_variation: float = 0.0,
        external_patterns_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.external_patterns_path = external_patterns_path
        self.tuning = tuning if tuning is not None else STANDARD_TUNING_OFFSETS
        self.timing_variation = timing_variation
        self.gate_length_variation = gate_length_variation
        from utilities.core_music_utils import get_time_signature_object

        ts_obj = get_time_signature_object(self.global_time_signature)
        self.measure_duration = (
            ts_obj.barDuration.quarterLength if ts_obj else 4.0
        )
        self.cfg: dict = kwargs.copy()
        self.style_selector = GuitarStyleSelector()
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

        if self.external_patterns_path:
            self._load_external_strum_patterns()

    def compose(self, *args, **kwargs):
        result = super().compose(*args, **kwargs)
        if isinstance(result, stream.Part):
            self._last_part = result
        elif isinstance(result, dict) and result:
            # store first part if dict
            self._last_part = next(iter(result.values()))
        else:
            self._last_part = None
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

    def _jitter(self, offset: float) -> float:
        if self.timing_variation:
            offset += self.rng.uniform(-self.timing_variation, self.timing_variation)
            if offset < 0:
                offset = 0.0
        return offset

    def _create_notes_from_event(
        self,
        cs: harmony.ChordSymbol,
        rhythm_pattern_definition: Dict[str, Any],
        guitar_block_params: Dict[str, Any],
        event_duration_ql: float,
        event_final_velocity: int,
    ) -> List[Union[note.Note, m21chord.Chord]]:
        notes_for_event: List[Union[note.Note, m21chord.Chord]] = []
        execution_style = rhythm_pattern_definition.get(
            "execution_style", EXEC_STYLE_BLOCK_CHORD
        )

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

            base_dur = event_duration_ql * (0.7 if is_palm_muted else 0.95)
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
            notes_for_event.append(ch)

        elif execution_style == EXEC_STYLE_BLOCK_CHORD:
            base_dur = event_duration_ql * (0.7 if is_palm_muted else 0.9)
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
            strum_delay = rhythm_pattern_definition.get(
                "strum_delay_ql",
                guitar_block_params.get("strum_delay_ql", GUITAR_STRUM_DELAY_QL),
            )

            for i, p_obj_strum in enumerate(play_order):
                n_strum = note.Note(p_obj_strum)
                base_dur = event_duration_ql * (0.6 if is_palm_muted else 0.9)
                base_dur *= 1 + self.rng.uniform(-self.gate_length_variation, self.gate_length_variation)
                n_strum.duration = music21_duration.Duration(
                    quarterLength=max(MIN_NOTE_DURATION_QL, base_dur)
                )
                n_strum.offset = self._jitter(i * strum_delay)
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
                n_strum.volume = m21volume.Volume(
                    velocity=max(1, min(127, event_final_velocity + vel_adj))
                )
                if is_palm_muted:
                    n_strum.articulations.append(articulations.Staccatissimo())
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
                base_dur = actual_arp_dur * (0.8 if is_palm_muted else 0.95)
                base_dur *= 1 + self.rng.uniform(-self.gate_length_variation, self.gate_length_variation)
                n_arp = note.Note(
                    p_play_arp,
                    quarterLength=max(MIN_NOTE_DURATION_QL, base_dur),
                )
                n_arp.volume = m21volume.Volume(velocity=event_final_velocity)
                n_arp.offset = self._jitter(current_offset_in_event)
                if is_palm_muted:
                    n_arp.articulations.append(articulations.Staccatissimo())
                notes_for_event.append(n_arp)
                current_offset_in_event += arp_note_dur_ql
                arp_idx += 1
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
                notes_for_event.append(n_mute)
                t_mute += mute_interval
        else:
            logger.warning(
                f"GuitarGen: Unknown or unhandled execution_style '{execution_style}' for chord {cs.figure if cs else 'N/A'}. No notes generated for this event."
            )
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

        pattern_events = rhythm_details.get("pattern", [])
        if pattern_events is None:
            pattern_events = []

        options = rhythm_details.get("options", {})
        velocity_curve_list = resolve_velocity_curve(options.get("velocity_curve"))

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

            event_velocity_factor = safe_get(
                event_def,
                "velocity_factor",
                default=1.0,
                cast_to=float,
                log_name=f"{log_event_prefix}.VelFactor",
            )

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

            final_event_velocity = int(block_base_velocity * event_velocity_factor)
            layer_idx = event_def.get("velocity_layer")
            if velocity_curve_list and layer_idx is not None:
                try:
                    idx = int(layer_idx)
                    if 0 <= idx < len(velocity_curve_list):
                        final_event_velocity = int(final_event_velocity * velocity_curve_list[idx])
                except (TypeError, ValueError):
                    pass
            final_event_velocity = max(1, min(127, final_event_velocity))

            # Palm Mute 判定 (パッチ参考)
            # final_guitar_params に palm_mute があればそれを使い、なければリズム定義から、それもなければFalse
            current_event_guitar_params["palm_mute"] = final_guitar_params.get(
                "palm_mute", rhythm_details.get("palm_mute", False)
            )

            generated_elements = self._create_notes_from_event(
                cs_object,
                rhythm_details,  # execution_style などを含むリズム定義
                current_event_guitar_params,  # palm_mute, current_event_stroke などを含む
                final_actual_event_dur_for_create,
                final_event_velocity,
            )

            for el in generated_elements:
                # el.offset は _create_notes_from_event 内でイベント開始からの相対オフセットになっている
                # これに、このリズムイベントのブロック内での開始オフセットを加算
                el.offset += current_event_start_offset_in_block
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
        # 旧呼び出しを noop にする互換 stub
        return


# --- END OF FILE generator/guitar_generator.py ---
