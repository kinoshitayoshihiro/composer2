# --- START OF FILE generator/piano_generator.py (safe_get 適用版) ---
import music21
from typing import (
    cast,
    List,
    Dict,
    Optional,
    Tuple,
    Any,
    Union,
    Sequence,
)
import copy

# music21 のサブモジュールを正しい形式でインポート
import music21.stream as stream
import music21.note as note
import music21.harmony as harmony
import music21.pitch as pitch
import music21.meter as meter
import music21.duration as duration
import music21.instrument as m21instrument
import music21.tempo as tempo
import music21.chord as m21chord
import music21.expressions as expressions
import music21.volume as m21volume
from music21 import articulations
from music21 import exceptions21

import random
import logging

# safe_get をインポート
try:
    from utilities.safe_get import safe_get
except ImportError:
    logger_fallback_safe_get = logging.getLogger(__name__ + ".fallback_safe_get_piano")
    logger_fallback_safe_get.error(
        "PianoGen: CRITICAL - Could not import safe_get from utilities. Fallback will be basic .get()."
    )

    # インポート失敗時のダミー safe_get (エラーを防ぐため)
    def safe_get(data, key_path, default=None, cast_to=None, log_name="dummy_safe_get"):
        val = data.get(key_path.split(".")[0])  # 単純な .get() のみ
        if val is None:
            return default
        if cast_to:
            try:
                return cast_to(val)
            except:
                return default
        return val


try:
    from utilities.override_loader import (
        get_part_override,
        Overrides as OverrideModelType,
        PartOverride as PartOverrideModel,  # PartOverrideModelもインポート
    )
    from utilities.core_music_utils import (
        MIN_NOTE_DURATION_QL,
        get_time_signature_object,
        sanitize_chord_label,
    )
    from utilities.humanizer import apply_humanization_to_part, HUMANIZATION_TEMPLATES
except ImportError:
    logging.basicConfig(level=logging.DEBUG)
    logger_fallback_utils_piano = logging.getLogger(__name__ + ".fallback_utils_piano")
    logger_fallback_utils_piano.warning(
        "PianoGen: Could not import from utilities. Using basic fallbacks."
    )
    MIN_NOTE_DURATION_QL = 0.125

    def get_time_signature_object(ts_str: Optional[str]) -> meter.TimeSignature:
        if not ts_str:
            ts_str = "4/4"
        try:
            return meter.TimeSignature(ts_str)
        except:
            return meter.TimeSignature("4/4")  # noqa E722

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

    def apply_humanization_to_part(part, template_name=None, custom_params=None):
        return part

    HUMANIZATION_TEMPLATES = {}

    class OverrideModelType:
        root: Dict = {}  # ダミー

    class PartOverrideModel:
        model_config = {}
        model_fields = {}  # ダミー

    def get_part_override(overrides, section, part) -> PartOverrideModel:
        return PartOverrideModel()  # ダミー


logger = logging.getLogger("modular_composer.piano_generator")

DEFAULT_PIANO_LH_OCTAVE: int = 2
DEFAULT_PIANO_RH_OCTAVE: int = 4
DEFAULT_PIANO_FALLBACK_RHYTHM_KEY = "piano_fallback_block"
DEFAULT_PIANO_RH_RHYTHM_KEY = "piano_rh_block_chords_quarters"
DEFAULT_PIANO_LH_RHYTHM_KEY = "piano_lh_roots_whole"


class PianoGenerator:
    def __init__(
        self,
        rhythm_library: Optional[Dict[str, Any]] = None,
        chord_voicer_instance: Optional[Any] = None,
        default_instrument_rh=m21instrument.Piano(),
        default_instrument_lh=m21instrument.Piano(),
        global_tempo: int = 120,
        global_time_signature: str = "4/4",
    ):
        self.rhythm_library = rhythm_library if rhythm_library is not None else {}
        self.chord_voicer = chord_voicer_instance
        if not self.chord_voicer:
            logger.warning(
                "PianoGen __init__: No ChordVoicer instance provided. Voicing might be basic."
            )

        self.instrument_rh = default_instrument_rh
        self.instrument_lh = default_instrument_lh
        self.global_tempo = global_tempo
        self.global_time_signature_str = global_time_signature
        try:
            self.global_time_signature_obj = get_time_signature_object(
                global_time_signature
            )
            if self.global_time_signature_obj is None:
                logger.warning(
                    f"PianoGen __init__: get_time_signature_object returned None for '{global_time_signature}'. Defaulting to 4/4."
                )
                self.global_time_signature_obj = meter.TimeSignature("4/4")
        except Exception as e_ts:
            logger.error(
                f"PianoGen __init__: Error initializing time signature from '{global_time_signature}': {e_ts}. Defaulting to 4/4.",
                exc_info=True,
            )
            self.global_time_signature_obj = meter.TimeSignature("4/4")

        bar_dur_ql = (
            self.global_time_signature_obj.barDuration.quarterLength
            if self.global_time_signature_obj
            else 4.0
        )

        default_patterns_to_ensure = {
            DEFAULT_PIANO_FALLBACK_RHYTHM_KEY: {
                "pattern": [
                    {
                        "offset": 0.0,
                        "duration": bar_dur_ql,
                        "velocity_factor": 0.7,
                        "type": "chord",
                    }
                ],
                "description": "Fallback block chord for the entire bar duration.",
                "reference_duration_ql": bar_dur_ql,
                "pattern_type": "fixed_pattern",
            },
            DEFAULT_PIANO_RH_RHYTHM_KEY: {
                "pattern": [
                    {
                        "offset": i * 1.0,
                        "duration": 1.0,
                        "velocity_factor": 0.75 - (i % 2 * 0.05),
                        "type": "chord",
                    }
                    for i in range(int(bar_dur_ql))
                ],
                "description": "Default RH quarter notes (chord).",
                "reference_duration_ql": bar_dur_ql,
                "pattern_type": "fixed_pattern",
            },
            DEFAULT_PIANO_LH_RHYTHM_KEY: {
                "pattern": [
                    {
                        "offset": 0.0,
                        "duration": bar_dur_ql,
                        "velocity_factor": 0.7,
                        "type": "root",
                    }
                ],
                "description": "Default LH root as whole note.",
                "reference_duration_ql": bar_dur_ql,
                "pattern_type": "fixed_pattern",
            },
        }
        for key, val_pattern in default_patterns_to_ensure.items():
            if key not in self.rhythm_library:
                self.rhythm_library[key] = val_pattern
                logger.info(
                    f"PianoGen __init__: Added default rhythm '{key}' to piano rhythm library."
                )

    def _get_piano_chord_pitches(
        self,
        cs_obj: Optional[harmony.ChordSymbol],
        num_voices_param: Optional[int],
        target_octave_param: int,
        voicing_style_name_param: str,
        specified_bass_for_voicing: Optional[str] = None,
    ) -> List[pitch.Pitch]:
        # (このメソッドは前回から変更なし)
        if cs_obj is None or not cs_obj.pitches:
            logger.debug(
                f"PianoGen._get_pitches: ChordSymbol is None or has no pitches ('{cs_obj}'). Returning empty list."
            )
            return []
        final_num_voices = (
            num_voices_param
            if num_voices_param is not None and num_voices_param > 0
            else None
        )
        logger.debug(
            f"PianoGen._get_pitches: For '{cs_obj.figure}', num_voices={final_num_voices}, target_oct={target_octave_param}, style='{voicing_style_name_param}', spec_bass='{specified_bass_for_voicing}'"
        )

        if self.chord_voicer and hasattr(self.chord_voicer, "_apply_voicing_style"):
            try:
                cs_for_voicing = copy.deepcopy(cs_obj)
                if specified_bass_for_voicing:
                    clean_bass = sanitize_chord_label(specified_bass_for_voicing)
                    if clean_bass and clean_bass.lower() != "rest":
                        cs_for_voicing.bass(clean_bass)
                voiced_pitches = self.chord_voicer._apply_voicing_style(
                    cs_for_voicing,
                    voicing_style_name_param,
                    target_octave_for_bottom_note=target_octave_param,
                    num_voices_target=final_num_voices,
                )
                logger.debug(
                    f"  ChordVoicer returned: {[p.nameWithOctave for p in voiced_pitches if p]}"
                )
                return voiced_pitches
            except Exception as e_cv:
                logger.warning(
                    f"PianoGen._get_pitches: Error in ChordVoicer for '{cs_obj.figure}' with style '{voicing_style_name_param}': {e_cv}. Falling back.",
                    exc_info=False,
                )
        else:
            logger.debug(
                "PianoGen._get_pitches: ChordVoicer not available or _apply_voicing_style missing. Using simple voicing."
            )

        try:  # Simple voicing fallback
            if not cs_obj.pitches:
                return []
            temp_m21_chord = m21chord.Chord(cs_obj.pitches)
            if specified_bass_for_voicing:
                clean_bass = sanitize_chord_label(specified_bass_for_voicing)
                if clean_bass and clean_bass.lower() != "rest":
                    try:
                        temp_m21_chord.bass(clean_bass)
                    except Exception as e_set_bass_simple:
                        logger.warning(
                            f"  Simple voicing: could not set bass {clean_bass}: {e_set_bass_simple}"
                        )

            closed_pitches_obj = temp_m21_chord.closedPosition(inPlace=False)
            if not closed_pitches_obj.pitches:
                return []

            pitches_to_voice_simple = list(closed_pitches_obj.pitches)
            reference_note_for_octave = (
                temp_m21_chord.bass()
                if temp_m21_chord.bass()
                else temp_m21_chord.root()
            )
            if not reference_note_for_octave and pitches_to_voice_simple:
                reference_note_for_octave = pitches_to_voice_simple[0]
            elif not reference_note_for_octave:
                return []

            current_bottom_pitch_of_voicing = min(
                pitches_to_voice_simple, key=lambda p: p.ps
            )
            target_bottom_ref_pitch = pitch.Pitch(reference_note_for_octave.name)
            target_bottom_ref_pitch.octave = target_octave_param
            semitones_to_shift = (
                round(
                    (target_bottom_ref_pitch.ps - current_bottom_pitch_of_voicing.ps)
                    / 12.0
                )
                * 12
            )

            voiced_pitches = [
                p.transpose(semitones_to_shift) for p in pitches_to_voice_simple
            ]
            voiced_pitches = sorted(voiced_pitches, key=lambda p: p.ps)

            if final_num_voices is not None:
                if len(voiced_pitches) > final_num_voices:
                    voiced_pitches = voiced_pitches[:final_num_voices]
                elif len(voiced_pitches) < final_num_voices and voiced_pitches:
                    can_add = final_num_voices - len(voiced_pitches)
                    for i_add in range(can_add):
                        if not voiced_pitches:
                            break
                        p_to_double = voiced_pitches[(i_add % len(voiced_pitches))]
                        doubled_pitch = p_to_double.transpose(12)
                        if doubled_pitch not in voiced_pitches:
                            voiced_pitches.append(doubled_pitch)
                    voiced_pitches = sorted(voiced_pitches, key=lambda p: p.ps)

            logger.debug(
                f"  Simple voicing returned: {[p.nameWithOctave for p in voiced_pitches if p]}"
            )
            return voiced_pitches
        except Exception as e_simple:
            logger.warning(
                f"PianoGen._get_pitches: Simple voicing for '{cs_obj.figure}' failed: {e_simple}. Raw.",
                exc_info=False,
            )
            raw_pitches = sorted(list(cs_obj.pitches), key=lambda p_sort: p_sort.ps)
            if final_num_voices is not None and raw_pitches:
                return raw_pitches[:final_num_voices]
            return raw_pitches if raw_pitches else []

    def _apply_pedal_to_part(
        self,
        part_to_apply_pedal: stream.Part,
        block_abs_offset: float,
        block_duration_ql: float,
    ):
        # (このメソッドは前回から変更なし)
        if block_duration_ql <= 0.25:
            return
        pedal_on = expressions.TextExpression("Ped.")
        pedal_off = expressions.TextExpression("*")
        insert_offset_on = block_abs_offset + 0.01
        insert_offset_off = block_abs_offset + block_duration_ql - 0.05
        if insert_offset_off > insert_offset_on:
            existing_ped_on = [
                el
                for el in part_to_apply_pedal.getElementsByOffset(
                    insert_offset_on, classList=[expressions.TextExpression]
                )
                if el.content == "Ped."
            ]
            if not existing_ped_on:
                part_to_apply_pedal.insert(insert_offset_on, pedal_on)
            existing_ped_off = [
                el
                for el in part_to_apply_pedal.getElementsByOffset(
                    insert_offset_off, classList=[expressions.TextExpression]
                )
                if el.content == "*"
            ]
            if not existing_ped_off:
                part_to_apply_pedal.insert(insert_offset_off, pedal_off)
            logger.debug(
                f"  Applied pedal at on:{insert_offset_on:.2f}, off:{insert_offset_off:.2f} for block duration {block_duration_ql:.2f}"
            )

    def _generate_piano_hand_part_for_block(
        self,
        hand_LR: str,
        cs_or_rest: Optional[music21.Music21Object],
        block_duration_ql: float,  # このブロックの実際のデュレーション
        block_abs_offset: float,
        block_event_data: Dict[str, Any],
        final_params_for_piano_hand: Dict[str, Any],
        emotion_params_for_block: Dict[str, Any],
        rhythm_patterns_for_piano: Dict[str, Any],
    ) -> stream.Part:
        hand_part_obj = stream.Part(
            id=f"Piano_{hand_LR}_temp_block_{block_abs_offset:.2f}"
        )
        logger.debug(
            f"PianoGen._generate_hand_part: START {hand_LR} for block at {block_abs_offset:.2f} (dur: {block_duration_ql:.2f}), Chord: {cs_or_rest}"
        )

        default_rh_key = DEFAULT_PIANO_RH_RHYTHM_KEY
        default_lh_key = DEFAULT_PIANO_LH_RHYTHM_KEY
        rhythm_key_specific_hand = final_params_for_piano_hand.get(
            f"piano_{hand_LR.lower()}_rhythm_key"
        )
        rhythm_key_generic_piano = final_params_for_piano_hand.get("piano_rhythm_key")
        current_rhythm_key_to_try = rhythm_key_specific_hand or rhythm_key_generic_piano
        if not current_rhythm_key_to_try:
            current_rhythm_key_to_try = (
                default_rh_key if hand_LR == "RH" else default_lh_key
            )
        actual_rhythm_key_to_use = (
            current_rhythm_key_to_try or DEFAULT_PIANO_FALLBACK_RHYTHM_KEY
        )
        logger.debug(
            f"  {hand_LR} rhythm key: Trying '{current_rhythm_key_to_try}', Actual used: '{actual_rhythm_key_to_use}'"
        )

        # ベロシティ解決 (前回と同様のロジック)
        block_vel_override = final_params_for_piano_hand.get("velocity")
        base_vel_for_block: int
        if block_vel_override is not None:
            try:
                base_vel_for_block = int(block_vel_override)
            except (ValueError, TypeError):
                logger.warning(
                    f"  {hand_LR} velocity override '{block_vel_override}' invalid. Using emotion or default."
                )
                base_vel_for_block = int(emotion_params_for_block.get("velocity", 64))
        else:
            base_vel_for_block = int(
                emotion_params_for_block.get(
                    "velocity",
                    final_params_for_piano_hand.get(
                        f"default_velocity_{hand_LR.lower()}", 64
                    ),
                )
            )
        logger.debug(f"  {hand_LR} base_vel_for_block (resolved): {base_vel_for_block}")

        # ボイシングパラメータ (前回と同様)
        voicing_style = final_params_for_piano_hand.get(
            f"piano_{hand_LR.lower()}_voicing_style",
            final_params_for_piano_hand.get(
                (
                    "default_rh_voicing_style"
                    if hand_LR == "RH"
                    else "default_lh_voicing_style"
                ),
                "closed_low" if hand_LR == "LH" else "spread_upper",
            ),
        )
        target_octave = int(
            final_params_for_piano_hand.get(
                f"piano_{hand_LR.lower()}_target_octave",
                final_params_for_piano_hand.get(
                    (
                        "default_rh_target_octave"
                        if hand_LR == "RH"
                        else "default_lh_target_octave"
                    ),
                    (
                        DEFAULT_PIANO_LH_OCTAVE
                        if hand_LR == "LH"
                        else DEFAULT_PIANO_RH_OCTAVE
                    ),
                ),
            )
        )
        num_voices_raw = final_params_for_piano_hand.get(
            f"piano_{hand_LR.lower()}_num_voices",
            final_params_for_piano_hand.get(
                "default_rh_num_voices" if hand_LR == "RH" else "default_lh_num_voices"
            ),
        )
        num_voices: Optional[int] = None
        if num_voices_raw is not None:
            try:
                num_voices = int(num_voices_raw)
            except (ValueError, TypeError):
                logger.warning(
                    f"  Invalid num_voices '{num_voices_raw}' for {hand_LR}, using None."
                )
        logger.debug(
            f"  {hand_LR} voicing: style='{voicing_style}', oct={target_octave}, voices={num_voices}"
        )

        articulation_from_emotion = emotion_params_for_block.get("articulation")
        logger.debug(
            f"  {hand_LR} articulation_from_emotion: {articulation_from_emotion}"
        )

        weak_beat_style = final_params_for_piano_hand.get(
            f"weak_beat_style_{hand_LR.lower()}", "none"
        )
        fill_on_4th_piano = final_params_for_piano_hand.get("fill_on_4th", False)
        fill_length_beats_piano = float(
            final_params_for_piano_hand.get("fill_length_beats", 0.5)
        )

        if isinstance(cs_or_rest, note.Rest) or not cs_or_rest:
            hand_part_obj.insert(0, note.Rest(quarterLength=block_duration_ql))
            logger.debug(
                f"  {hand_LR}: Inserted Rest for duration {block_duration_ql:.2f}"
            )
            return hand_part_obj

        current_cs = cast(harmony.ChordSymbol, cs_or_rest)
        spec_bass_for_voicing = block_event_data.get("specified_bass_for_voicing")
        voiced_pitches_for_hand = self._get_piano_chord_pitches(
            current_cs, num_voices, target_octave, voicing_style, spec_bass_for_voicing
        )
        if not voiced_pitches_for_hand:
            hand_part_obj.insert(0, note.Rest(quarterLength=block_duration_ql))
            logger.debug(
                f"  {hand_LR}: No voiced pitches for '{current_cs.figure}'. Inserted Rest."
            )
            return hand_part_obj
        logger.debug(
            f"  {hand_LR}: Voiced pitches for '{current_cs.figure}' ({voicing_style}): {[p.nameWithOctave for p in voiced_pitches_for_hand]}"
        )

        rhythm_pattern_data = rhythm_patterns_for_piano.get(actual_rhythm_key_to_use)
        if (
            not rhythm_pattern_data
            or "pattern" not in rhythm_pattern_data
            or not rhythm_pattern_data["pattern"]
        ):
            logger.warning(
                f"PianoGen._generate_hand_part: Rhythm key '{actual_rhythm_key_to_use}' for {hand_LR} not found/invalid. Using fallback '{DEFAULT_PIANO_FALLBACK_RHYTHM_KEY}'."
            )
            rhythm_pattern_data = rhythm_patterns_for_piano.get(
                DEFAULT_PIANO_FALLBACK_RHYTHM_KEY
            )
            if (
                not rhythm_pattern_data
                or "pattern" not in rhythm_pattern_data
                or not rhythm_pattern_data["pattern"]
            ):
                logger.error(
                    f"PianoGen._generate_hand_part: CRITICAL! Fallback rhythm '{DEFAULT_PIANO_FALLBACK_RHYTHM_KEY}' not found/invalid. Skipping block for {hand_LR}."
                )
                hand_part_obj.insert(0, note.Rest(quarterLength=block_duration_ql))
                return hand_part_obj

        pattern_event_list = rhythm_pattern_data.get("pattern", [])

        # --- pattern_reference_ql の取得を safe_get に修正 ---
        pattern_reference_ql = safe_get(
            rhythm_pattern_data,  # 対象辞書
            "reference_duration_ql",  # キーパス
            default=block_duration_ql,  # Noneの場合のデフォルト値 (ブロック長)
            cast_to=float,  # float型にキャスト
            log_name=f"PianoGen.{hand_LR}.ref_dur",  # ログ用識別子
        )
        if (
            pattern_reference_ql <= 0
        ):  # キャスト後またはデフォルト値が0以下の場合のフォールバック
            logger.warning(
                f"PianoGen: pattern_reference_ql for key '{actual_rhythm_key_to_use}' was {pattern_reference_ql} (<=0). Defaulting to block_duration_ql: {block_duration_ql}"
            )
            pattern_reference_ql = block_duration_ql
        # --- ここまで pattern_reference_ql の修正 ---

        pattern_type_from_lib = rhythm_pattern_data.get("pattern_type", "fixed_pattern")
        time_sig_obj = self.global_time_signature_obj
        num_beats_in_bar = time_sig_obj.beatCount if time_sig_obj else 4
        beat_ql = time_sig_obj.beatDuration.quarterLength if time_sig_obj else 1.0

        # (以降の _generate_piano_hand_part_for_block のロジックは前回から変更なし)
        for p_event_idx, p_event_def in enumerate(pattern_event_list):
            event_offset_from_pattern_start = float(p_event_def.get("offset", 0.0))
            event_duration_from_pattern = float(p_event_def.get("duration", beat_ql))
            event_velocity_factor = float(p_event_def.get("velocity_factor", 1.0))
            event_note_type_hint = p_event_def.get("type")
            scale_factor_for_timing = block_duration_ql / pattern_reference_ql
            current_event_rel_offset_in_block = (
                event_offset_from_pattern_start * scale_factor_for_timing
            )
            current_event_actual_duration = (
                event_duration_from_pattern * scale_factor_for_timing
            )
            if current_event_rel_offset_in_block >= block_duration_ql - (
                MIN_NOTE_DURATION_QL / 16.0
            ):
                continue
            current_event_actual_duration = min(
                current_event_actual_duration,
                block_duration_ql - current_event_rel_offset_in_block,
            )
            if current_event_actual_duration < MIN_NOTE_DURATION_QL / 4.0:
                continue
            final_event_vel = max(
                1, min(127, int(base_vel_for_block * event_velocity_factor))
            )
            beat_pos_in_bar = (
                current_event_rel_offset_in_block / beat_ql
            ) % num_beats_in_bar
            is_on_weak_beat = False
            if num_beats_in_bar == 4 and (
                1.0 <= beat_pos_in_bar < 2.0 or 3.0 <= beat_pos_in_bar < 4.0
            ):
                is_on_weak_beat = True
            elif num_beats_in_bar == 3 and (1.0 <= beat_pos_in_bar < 3.0):
                is_on_weak_beat = True
            if is_on_weak_beat:
                if weak_beat_style == "rest":
                    logger.debug(
                        f"    {hand_LR} event at {current_event_rel_offset_in_block:.2f} skipped (weak_beat_style='rest')."
                    )
                    continue
                elif weak_beat_style == "ghost":
                    final_event_vel = max(1, int(final_event_vel * 0.5))
                    logger.debug(
                        f"    {hand_LR} event at {current_event_rel_offset_in_block:.2f} ghosted (vel: {final_event_vel})."
                    )
            is_on_4th_beat_trigger = False
            if num_beats_in_bar == 4 and abs(
                current_event_rel_offset_in_block - (beat_ql * (num_beats_in_bar - 1))
            ) < (beat_ql / 4.0):
                is_on_4th_beat_trigger = True
            if (
                hand_LR == "RH"
                and fill_on_4th_piano
                and is_on_4th_beat_trigger
                and fill_length_beats_piano > 0
            ):
                original_event_duration_for_fill = current_event_actual_duration
                current_event_actual_duration = max(
                    MIN_NOTE_DURATION_QL,
                    original_event_duration_for_fill - fill_length_beats_piano,
                )
                fill_start_offset_in_block = (
                    current_event_rel_offset_in_block + current_event_actual_duration
                )
                if (
                    voiced_pitches_for_hand
                    and fill_length_beats_piano > MIN_NOTE_DURATION_QL
                ):
                    fill_notes_to_use = voiced_pitches_for_hand[-2:]
                    num_fill_notes_generated = max(
                        1, int(fill_length_beats_piano / (MIN_NOTE_DURATION_QL * 1.5))
                    )
                    single_fill_note_ql = (
                        fill_length_beats_piano / num_fill_notes_generated
                        if num_fill_notes_generated > 0
                        else MIN_NOTE_DURATION_QL
                    )
                    for i_fill in range(num_fill_notes_generated):
                        if not fill_notes_to_use:
                            break
                        f_pitch = fill_notes_to_use[i_fill % len(fill_notes_to_use)]
                        fill_note_obj = note.Note(
                            f_pitch, quarterLength=single_fill_note_ql * 0.85
                        )
                        fill_note_obj.volume.velocity = min(127, final_event_vel + 10)
                        if articulation_from_emotion == "staccato":
                            fill_note_obj.articulations.append(articulations.Staccato())
                        offset_for_this_fill_note = fill_start_offset_in_block + (
                            i_fill * single_fill_note_ql
                        )
                        if offset_for_this_fill_note < block_duration_ql - (
                            MIN_NOTE_DURATION_QL / 2
                        ):
                            hand_part_obj.insert(
                                offset_for_this_fill_note, fill_note_obj
                            )
                    logger.debug(
                        f"    {hand_LR} added fill at {fill_start_offset_in_block:.2f} for {fill_length_beats_piano:.2f} beats."
                    )
                if current_event_actual_duration < MIN_NOTE_DURATION_QL / 4.0:
                    logger.debug(
                        f"    {hand_LR} original event at {current_event_rel_offset_in_block:.2f} skipped after fill made it too short."
                    )
                    continue
            notes_or_chord_to_insert: Optional[music21.note.GeneralNote] = None
            if pattern_type_from_lib == "arpeggio_indices" and voiced_pitches_for_hand:
                arp_indices_pattern = rhythm_pattern_data.get(
                    "arpeggio_indices",
                    (
                        [0, 1, 2, 1]
                        if len(voiced_pitches_for_hand) >= 3
                        else ([0, 1] if len(voiced_pitches_for_hand) == 2 else [0])
                    ),
                )
                arp_note_ql_from_pattern = float(
                    rhythm_pattern_data.get("note_duration_ql", 0.5)
                )
                arp_note_actual_ql = arp_note_ql_from_pattern * scale_factor_for_timing
                num_arp_notes_possible = (
                    int(current_event_actual_duration / arp_note_actual_ql)
                    if arp_note_actual_ql > 0
                    else 0
                )
                for i_arp in range(num_arp_notes_possible):
                    arp_pitch_index = arp_indices_pattern[
                        i_arp % len(arp_indices_pattern)
                    ]
                    actual_pitch_for_arp = voiced_pitches_for_hand[
                        arp_pitch_index % len(voiced_pitches_for_hand)
                    ]
                    arp_note_offset_in_event = i_arp * arp_note_actual_ql
                    arp_n = note.Note(
                        actual_pitch_for_arp, quarterLength=arp_note_actual_ql * 0.95
                    )
                    arp_n.volume = m21volume.Volume(
                        velocity=final_event_vel + random.randint(-3, 3)
                    )
                    if articulation_from_emotion == "staccato":
                        arp_n.articulations.append(articulations.Staccato())
                    elif articulation_from_emotion == "tenuto":
                        arp_n.articulations.append(articulations.Tenuto())
                    hand_part_obj.insert(
                        current_event_rel_offset_in_block + arp_note_offset_in_event,
                        arp_n,
                    )
                logger.debug(
                    f"    {hand_LR} added arpeggio at {current_event_rel_offset_in_block:.2f} (pattern: {pattern_type_from_lib})"
                )
                continue
            elif pattern_type_from_lib == "fixed_pattern":
                pitches_to_play_fixed: List[pitch.Pitch] = []
                hint = event_note_type_hint or ("chord" if hand_LR == "RH" else "root")
                if hint == "root" and voiced_pitches_for_hand:
                    pitches_to_play_fixed.append(
                        min(voiced_pitches_for_hand, key=lambda p: p.ps)
                    )
                elif hint == "octave_root" and voiced_pitches_for_hand:
                    root_p = min(voiced_pitches_for_hand, key=lambda p: p.ps)
                    pitches_to_play_fixed.append(root_p)
                    pitches_to_play_fixed.append(root_p.transpose(12))
                elif hint == "fifth_or_root" and voiced_pitches_for_hand:
                    fifth_interval = music21.interval.Interval("P5")
                    root_of_cs = current_cs.root()
                    if root_of_cs:
                        fifth_pitch = fifth_interval.transposePitch(root_of_cs)
                        pitches_to_play_fixed.append(
                            fifth_pitch
                            if fifth_pitch
                            else min(voiced_pitches_for_hand, key=lambda p: p.ps)
                        )
                    else:
                        pitches_to_play_fixed.append(
                            min(voiced_pitches_for_hand, key=lambda p: p.ps)
                        )
                elif hint == "chord" and voiced_pitches_for_hand:
                    pitches_to_play_fixed.extend(voiced_pitches_for_hand)
                elif hint == "chord_mid_voices" and voiced_pitches_for_hand:
                    if len(voiced_pitches_for_hand) >= 3:
                        pitches_to_play_fixed.extend(
                            voiced_pitches_for_hand[1:-1]
                            or [
                                voiced_pitches_for_hand[
                                    len(voiced_pitches_for_hand) // 2
                                ]
                            ]
                        )
                    else:
                        pitches_to_play_fixed.extend(voiced_pitches_for_hand)
                elif hint == "chord_high_voices" and voiced_pitches_for_hand:
                    if len(voiced_pitches_for_hand) >= 2:
                        pitches_to_play_fixed.extend(
                            voiced_pitches_for_hand[
                                -(len(voiced_pitches_for_hand) // 2) :
                            ]
                        )
                    else:
                        pitches_to_play_fixed.extend(voiced_pitches_for_hand)
                elif voiced_pitches_for_hand:
                    pitches_to_play_fixed.extend(
                        voiced_pitches_for_hand
                        if hand_LR == "RH"
                        else [min(voiced_pitches_for_hand, key=lambda p: p.ps)]
                    )
                if pitches_to_play_fixed:
                    if len(pitches_to_play_fixed) > 1:
                        notes_or_chord_to_insert = m21chord.Chord(
                            pitches_to_play_fixed,
                            quarterLength=current_event_actual_duration,
                        )
                    elif pitches_to_play_fixed:
                        notes_or_chord_to_insert = note.Note(
                            pitches_to_play_fixed[0],
                            quarterLength=current_event_actual_duration,
                        )
            else:
                logger.warning(
                    f"PianoGen: Unsupported pattern_type '{pattern_type_from_lib}' for rhythm key '{actual_rhythm_key_to_use}'. Skipping this event in pattern."
                )
                continue
            if notes_or_chord_to_insert:
                if isinstance(notes_or_chord_to_insert, m21chord.Chord):
                    for n_in_chord_item in notes_or_chord_to_insert.notes:
                        n_in_chord_item.volume = m21volume.Volume(
                            velocity=final_event_vel
                        )
                elif isinstance(notes_or_chord_to_insert, note.Note):
                    notes_or_chord_to_insert.volume = m21volume.Volume(
                        velocity=final_event_vel
                    )
                if articulation_from_emotion:
                    if notes_or_chord_to_insert.articulations is None:
                        notes_or_chord_to_insert.articulations = []
                    if articulation_from_emotion.lower() == "staccato":
                        notes_or_chord_to_insert.articulations.append(
                            articulations.Staccato()
                        )
                    elif articulation_from_emotion.lower() == "tenuto":
                        notes_or_chord_to_insert.articulations.append(
                            articulations.Tenuto()
                        )
                    elif articulation_from_emotion.lower() == "accent":
                        notes_or_chord_to_insert.articulations.append(
                            articulations.Accent()
                        )
                hand_part_obj.insert(
                    current_event_rel_offset_in_block, notes_or_chord_to_insert
                )
                common_name_attr = getattr(
                    notes_or_chord_to_insert,
                    "pitchedCommonName",
                    getattr(notes_or_chord_to_insert, "name", "Rest"),
                )
                logger.debug(
                    f"    {hand_LR} added at relative {current_event_rel_offset_in_block:.2f}: {common_name_attr}, Vel: {final_event_vel}, Art: {articulation_from_emotion}, Dur: {notes_or_chord_to_insert.duration.quarterLength:.2f}"
                )
        logger.debug(
            f"PianoGen._generate_hand_part: END {hand_LR} for block at {block_abs_offset:.2f}. Generated {len(list(hand_part_obj.flatten().notesAndRests))} elements."
        )
        return hand_part_obj

    def compose(
        self,
        processed_chord_stream: List[Dict],
        overrides: Optional[OverrideModelType] = None,  # 型ヒントを修正
    ) -> stream.Score:
        piano_score = stream.Score(id="PianoScore_OtoKotoba")

        rh_instrument = copy.deepcopy(self.instrument_rh)
        rh_instrument.partName = "Piano RH"
        rh_instrument.partAbbreviation = "RH"
        piano_rh_part = stream.Part(id="PianoRH_OtoKotoba")
        piano_rh_part.insert(0, rh_instrument)
        lh_instrument = copy.deepcopy(self.instrument_lh)
        lh_instrument.partName = "Piano LH"
        lh_instrument.partAbbreviation = "LH"
        piano_lh_part = stream.Part(id="PianoLH_OtoKotoba")
        piano_lh_part.insert(0, lh_instrument)

        piano_score.insert(0, tempo.MetronomeMark(number=self.global_tempo))
        current_ts_obj = (
            self.global_time_signature_obj
            if self.global_time_signature_obj
            else meter.TimeSignature("4/4")
        )
        piano_score.insert(0, copy.deepcopy(current_ts_obj))
        piano_rh_part.insert(0, copy.deepcopy(current_ts_obj))
        piano_lh_part.insert(0, copy.deepcopy(current_ts_obj))

        if not processed_chord_stream:
            logger.warning(
                "PianoGen.compose: processed_chord_stream is empty. Returning score with only global settings."
            )
            piano_score.append(piano_rh_part)
            piano_score.append(piano_lh_part)
            return piano_score
        logger.info(
            f"PianoGen.compose: Beginning piano part generation for {len(processed_chord_stream)} processed chord events."
        )

        for blk_idx, block_data_original in enumerate(processed_chord_stream):
            blk_data = copy.deepcopy(block_data_original)

            # --- オフセットとデュレーションの取得 (safe_get を使用) ---
            abs_block_offset = safe_get(
                blk_data,
                "absolute_offset",
                default=safe_get(
                    blk_data,
                    "offset",
                    default=0.0,
                    cast_to=float,
                    log_name=f"PianoCompose.Blk{blk_idx}.OffsetFallback",
                ),
                cast_to=float,
                log_name=f"PianoCompose.Blk{blk_idx}.AbsOffset",
            )
            block_ql = safe_get(
                blk_data,
                "humanized_duration_beats",
                default=safe_get(
                    blk_data,
                    "q_length",
                    default=4.0,
                    cast_to=float,
                    log_name=f"PianoCompose.Blk{blk_idx}.QLFallback",
                ),
                cast_to=float,
                log_name=f"PianoCompose.Blk{blk_idx}.HumDur",
            )
            if block_ql <= 0:  # デュレーションが0以下ならフォールバック
                default_duration_fallback_pc = (
                    self.global_time_signature_obj.barDuration.quarterLength
                    if self.global_time_signature_obj
                    else 4.0
                )
                logger.warning(
                    f"PianoGen.compose: Block #{blk_idx} has non-positive duration {block_ql}. Using {default_duration_fallback_pc}ql."
                )
                block_ql = default_duration_fallback_pc
            # --- ここまでオフセットとデュレーションの取得修正 ---

            original_label = blk_data.get("original_chord_label", "N.C.")
            chord_for_voicing_label = blk_data.get(
                "chord_symbol_for_voicing", original_label
            )
            current_section_name = blk_data.get(
                "section_name", f"UnnamedSection_{blk_idx}"
            )

            part_specific_overrides_model: Optional[PartOverrideModel] = None
            if overrides:
                part_specific_overrides_model = get_part_override(
                    overrides, current_section_name, "piano"
                )

            piano_params_from_chordmap = blk_data.get("part_params", {}).get(
                "piano", {}
            )
            final_piano_params = piano_params_from_chordmap.copy()
            if part_specific_overrides_model and hasattr(
                part_specific_overrides_model, "model_dump"
            ):  # ダミーでないことを確認
                override_dict = part_specific_overrides_model.model_dump(
                    exclude_unset=True
                )
                chordmap_options = final_piano_params.get("options", {})
                override_options = override_dict.pop("options", None)
                if isinstance(chordmap_options, dict) and isinstance(
                    override_options, dict
                ):
                    merged_options = chordmap_options.copy()
                    merged_options.update(override_options)
                    final_piano_params["options"] = merged_options
                elif isinstance(override_options, dict) and override_options:
                    final_piano_params["options"] = override_options
                final_piano_params.update(override_dict)

            emotion_profile_this_block = blk_data.get("emotion_profile_applied", {})
            logger.debug(
                f"  Processing Block {blk_idx+1}/{len(processed_chord_stream)}: Offset={abs_block_offset:.2f}, QL={block_ql:.2f}, Chord='{original_label}' (Voicing='{chord_for_voicing_label}')"
            )
            logger.debug(f"    Final Piano Params: {final_piano_params}")
            logger.debug(f"    Emotion Profile: {emotion_profile_this_block}")

            cs_or_rest_for_block: Optional[music21.Music21Object] = None
            sanitized_chord_input = sanitize_chord_label(chord_for_voicing_label)
            if not sanitized_chord_input or sanitized_chord_input.lower() == "rest":
                cs_or_rest_for_block = note.Rest(quarterLength=block_ql)
            else:
                try:
                    cs_or_rest_for_block = harmony.ChordSymbol(sanitized_chord_input)
                    if not cs_or_rest_for_block.pitches:
                        logger.warning(
                            f"    ChordSymbol '{sanitized_chord_input}' parsed to no pitches. Treating as Rest."
                        )
                        cs_or_rest_for_block = note.Rest(quarterLength=block_ql)
                except Exception as e_cs:
                    logger.error(
                        f"    Error parsing ChordSymbol '{sanitized_chord_input}': {e_cs}. Treating as Rest.",
                        exc_info=True,
                    )
                    cs_or_rest_for_block = note.Rest(quarterLength=block_ql)

            rh_block_part_notes = self._generate_piano_hand_part_for_block(
                "RH",
                cs_or_rest_for_block,
                block_ql,
                abs_block_offset,
                blk_data,
                copy.deepcopy(final_piano_params),
                emotion_profile_this_block,
                self.rhythm_library,
            )
            lh_block_part_notes = self._generate_piano_hand_part_for_block(
                "LH",
                cs_or_rest_for_block,
                block_ql,
                abs_block_offset,
                blk_data,
                copy.deepcopy(final_piano_params),
                emotion_profile_this_block,
                self.rhythm_library,
            )

            for el_rh in rh_block_part_notes.flatten().notesAndRests:
                piano_rh_part.insert(
                    abs_block_offset + el_rh.getOffsetBySite(rh_block_part_notes.flat),
                    el_rh,
                )
            for el_lh in lh_block_part_notes.flatten().notesAndRests:
                piano_lh_part.insert(
                    abs_block_offset + el_lh.getOffsetBySite(lh_block_part_notes.flat),
                    el_lh,
                )

            apply_pedal_setting = final_piano_params.get(
                "piano_apply_pedal", final_piano_params.get("apply_pedal", True)
            )
            if not isinstance(cs_or_rest_for_block, note.Rest) and apply_pedal_setting:
                self._apply_pedal_to_part(piano_lh_part, abs_block_offset, block_ql)

        # (以降のHumanize処理、スコアへの追加、ログ出力は前回から変更なし)
        final_humanize_rh_opt = False
        final_humanize_lh_opt = False
        humanize_template = "default_subtle"
        humanize_custom = {}
        if (
            processed_chord_stream
            and "part_params" in processed_chord_stream[0]
            and "piano" in processed_chord_stream[0]["part_params"]
        ):
            first_block_piano_cfg = processed_chord_stream[0]["part_params"]["piano"]
            final_humanize_rh_opt = first_block_piano_cfg.get(
                "humanize_rh_opt", first_block_piano_cfg.get("humanize_opt", False)
            )
            final_humanize_lh_opt = first_block_piano_cfg.get(
                "humanize_lh_opt", first_block_piano_cfg.get("humanize_opt", False)
            )
            humanize_template = first_block_piano_cfg.get(
                "humanize_template_name",
                first_block_piano_cfg.get("template_name", "default_subtle"),
            )
            humanize_custom = first_block_piano_cfg.get(
                "humanize_custom_params", first_block_piano_cfg.get("custom_params", {})
            )
        if final_humanize_rh_opt and piano_rh_part.flatten().notes:
            logger.info(
                f"PianoGen.compose: Applying final touch humanization to Piano RH part (template: {humanize_template})."
            )
            piano_rh_part = apply_humanization_to_part(
                piano_rh_part,
                template_name=humanize_template,
                custom_params=humanize_custom,
            )
            piano_rh_part.id = "PianoRH_OtoKotoba_H"
        if final_humanize_lh_opt and piano_lh_part.flatten().notes:
            logger.info(
                f"PianoGen.compose: Applying final touch humanization to Piano LH part (template: {humanize_template})."
            )
            piano_lh_part = apply_humanization_to_part(
                piano_lh_part,
                template_name=humanize_template,
                custom_params=humanize_custom,
            )
            piano_lh_part.id = "PianoLH_OtoKotoba_H"
        piano_score.insert(0, piano_rh_part)
        piano_score.insert(0, piano_lh_part)
        logger.info(
            f"PianoGen.compose: Finished. Piano Score contains {len(list(piano_score.flatten().notesAndRests))} elements in total."
        )
        if piano_rh_part.flatten().notesAndRests:
            logger.info(
                f"  RH Part: {len(list(piano_rh_part.flatten().notesAndRests))} elements."
            )
        else:
            logger.info("  RH Part: is empty.")
        if piano_lh_part.flatten().notesAndRests:
            logger.info(
                f"  LH Part: {len(list(piano_lh_part.flatten().notesAndRests))} elements."
            )
        else:
            logger.info("  LH Part: is empty.")
        return piano_score


# --- END OF FILE generator/piano_generator.py ---
