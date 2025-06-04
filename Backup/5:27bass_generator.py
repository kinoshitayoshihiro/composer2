# --- START OF FILE generator/bass_generator.py (ボーカル協調フェーズ1対応) ---
from __future__ import annotations
"""bass_generator.py – streamlined rewrite with vocal context awareness (Phase 1)
Generates a **bass part** for the modular composer pipeline.
The heavy lifting (walking line, root-fifth, etc.) is delegated to
generator.bass_utils.generate_bass_measure so that this class
mainly decides **which style to use when** and applies rhythm.
Vocal note information is passed to bass_utils for consideration.
"""
from typing import Sequence, Dict, Any, Optional, List, Union, cast

import random
import logging

# music21 のサブモジュールを正しい形式でインポート
import music21.stream as stream
import music21.harmony as harmony
import music21.note as note
import music21.tempo as tempo
import music21.meter as meter
import music21.instrument as m21instrument
import music21.key as key
import music21.pitch as pitch
import music21.volume as m21volume

# ユーティリティのインポート
try:
    from .bass_utils import generate_bass_measure
    from utilities.core_music_utils import get_time_signature_object, sanitize_chord_label, MIN_NOTE_DURATION_QL
    from utilities.humanizer import apply_humanization_to_part, HUMANIZATION_TEMPLATES
except ImportError as e:
    logger_fallback = logging.getLogger(__name__ + ".fallback_utils")
    logger_fallback.error(f"BassGenerator: Failed to import required modules: {e}")
    def generate_bass_measure(*args, **kwargs) -> List[note.Note]: return [] # type: ignore
    def apply_humanization_to_part(part, *args, **kwargs) -> stream.Part: # type: ignore
        if isinstance(part, stream.Part):
            return part
        return stream.Part()
    def get_time_signature_object(ts_str: Optional[str]) -> meter.TimeSignature: return meter.TimeSignature("4/4") # type: ignore
    def sanitize_chord_label(label: Optional[str]) -> Optional[str]: # type: ignore
        if not label or label.strip().lower() in ["rest", "r", "n.c.", "nc", "none"]: return None
        return label.strip()
    MIN_NOTE_DURATION_QL = 0.125
    HUMANIZATION_TEMPLATES = {}


logger = logging.getLogger(__name__)

class BassGenerator:
    def __init__(
        self,
        rhythm_library: Optional[Dict[str, Dict]] = None,
        default_instrument = m21instrument.AcousticBass(),
        global_tempo: int = 100,
        global_time_signature: str = "4/4",
        global_key_tonic: str = "C",
        global_key_mode: str = "major",
        rng: Optional[random.Random] = None,
    ) -> None:
        self.rhythm_library = rhythm_library if rhythm_library is not None else {}
        self.default_instrument = default_instrument
        self.global_tempo = global_tempo
        self.global_time_signature_str = global_time_signature
        self.global_time_signature_obj = get_time_signature_object(global_time_signature)
        self.global_key_tonic = global_key_tonic
        self.global_key_mode = global_key_mode
        self.rng = rng or random.Random()

        if "bass_quarter_notes" not in self.rhythm_library:
            self.rhythm_library["bass_quarter_notes"] = {
                "description": "Default quarter note roots for bass.",
                "pattern": [
                    {"offset": 0.0, "duration": 1.0, "velocity_factor": 0.75, "type": "root"},
                    {"offset": 1.0, "duration": 1.0, "velocity_factor": 0.7, "type": "root"},
                    {"offset": 2.0, "duration": 1.0, "velocity_factor": 0.75, "type": "root"},
                    {"offset": 3.0, "duration": 1.0, "velocity_factor": 0.7, "type": "root"}
                ]
            }
            logger.info("BassGenerator: Added 'bass_quarter_notes' to rhythm_library as fallback.")


    def _select_style(self, bass_params: Dict[str, Any], blk_musical_intent: Dict[str, Any]) -> str:
        """
        Selects the bass style based on parameters and musical intent.
        """
        if "style" in bass_params and bass_params["style"]:
            return bass_params["style"]
        intensity = blk_musical_intent.get("intensity", "medium").lower()
        if intensity in {"low", "medium_low"}: return "root_only"
        if intensity in {"medium"}: return "root_fifth"
        return "walking" # Default for higher intensities

    def compose(self, processed_blocks: Sequence[Dict[str, Any]]) -> stream.Part:
        """
        Composes the bass part based on the processed chord blocks.

        Args:
            processed_blocks (Sequence[Dict[str, Any]]): A list of dictionaries,
                where each dictionary represents a chord block with its properties,
                including chord label, duration, musical intent, and potentially
                vocal notes occurring within that block under the key "vocal_notes_in_block".

        Returns:
            music21.stream.Part: The generated bass part.
        """
        bass_part = stream.Part(id="Bass")
        bass_part.insert(0, self.default_instrument)
        bass_part.insert(0, tempo.MetronomeMark(number=self.global_tempo))
        ts_copy_init = meter.TimeSignature(self.global_time_signature_obj.ratioString)
        bass_part.insert(0, ts_copy_init)

        first_block_tonic = processed_blocks[0].get("tonic_of_section", self.global_key_tonic) if processed_blocks else self.global_key_tonic
        first_block_mode = processed_blocks[0].get("mode", self.global_key_mode) if processed_blocks else self.global_key_mode
        try:
            bass_part.insert(0, key.Key(first_block_tonic, first_block_mode))
        except Exception as e_key:
            logger.warning(f"BassGenerator: Could not set initial key {first_block_tonic} {first_block_mode}: {e_key}. Using C Major.")
            bass_part.insert(0, key.Key("C", "major"))


        current_total_offset = 0.0

        for i, blk_data in enumerate(processed_blocks):
            bass_params = blk_data.get("part_params", {}).get("bass", {})
            block_q_length = blk_data.get("q_length", 4.0) # Default to 4.0 beats if not specified

            if not bass_params: # Parameters for bass might be missing for this block
                logger.debug(f"BassGenerator: No bass parameters for block {i+1}. Skipping bass for this block.")
                current_total_offset += block_q_length
                continue

            chord_label_str = blk_data.get("chord_label", "C")

            if chord_label_str.lower() == "rest":
                logger.info(f"BassGenerator: Block {i+1} ('{chord_label_str}') is a Rest. Skipping bass notes.")
                current_total_offset += block_q_length
                continue

            musical_intent = blk_data.get("musical_intent", {})
            selected_style = self._select_style(bass_params, musical_intent)

            cs_now_obj: Optional[harmony.ChordSymbol] = None
            sanitized_label = sanitize_chord_label(chord_label_str)
            if sanitized_label:
                try:
                    cs_now_obj = harmony.ChordSymbol(sanitized_label)
                    if not cs_now_obj.pitches: # ChordSymbol parsed but has no pitches (e.g., "N.C.")
                        cs_now_obj = None
                except Exception as e_parse:
                    logger.warning(f"BassGenerator: Error parsing chord '{chord_label_str}' (sanitized: '{sanitized_label}') for block {i+1}: {e_parse}. Will attempt to use a default C chord if needed.")
                    cs_now_obj = None # Explicitly set to None on error

            if cs_now_obj is None:
                logger.warning(f"BassGenerator: Could not create valid ChordSymbol for '{chord_label_str}' (Sanitized: '{sanitized_label}') in block {i+1}. Skipping notes for this block (or using fallback if style allows).")
                # Depending on style, generate_bass_measure might handle cs_now=None with a default.
                # For safety, we can skip or use a default C. Here, we'll let generate_bass_measure handle it or skip.
                # If we decide to always generate something, this is where a default C could be set:
                # cs_now_obj = harmony.ChordSymbol("C")

            # Get next chord for context (e.g., for walking bass approach notes)
            cs_next_obj: Optional[harmony.ChordSymbol] = None
            if i + 1 < len(processed_blocks):
                next_blk_data = processed_blocks[i+1]
                next_label_str = next_blk_data.get("chord_label")
                if next_label_str and next_label_str.lower() != "rest":
                    sanitized_next_label = sanitize_chord_label(next_label_str)
                    if sanitized_next_label:
                        try:
                            cs_next_obj = harmony.ChordSymbol(sanitized_next_label)
                            if not cs_next_obj.pitches:
                                cs_next_obj = None
                        except Exception:
                            cs_next_obj = None
            if cs_next_obj is None and cs_now_obj: # If no next chord or next is rest, use current chord as next for stability
                cs_next_obj = cs_now_obj
            elif cs_next_obj is None and cs_now_obj is None: # Both current and next are problematic
                logger.warning(f"BassGenerator: Both current and next chords are unparsable for block {i+1}. Attempting with default C for bass_utils.")
                cs_now_obj = harmony.ChordSymbol("C") # Provide a default
                cs_next_obj = cs_now_obj


            tonic = blk_data.get("tonic_of_section", self.global_key_tonic)
            mode = blk_data.get("mode", self.global_key_mode)
            target_octave = bass_params.get("octave", bass_params.get("bass_target_octave", 2))
            base_velocity = bass_params.get("velocity", bass_params.get("bass_velocity", 70))

            # Get vocal notes for this block
            vocal_notes_in_block = blk_data.get("vocal_notes_in_block", [])
            if vocal_notes_in_block:
                 logger.debug(f"BassGenerator: Block {i+1} (Chord: {chord_label_str}), {len(vocal_notes_in_block)} vocal notes for bass_utils consideration.")


            measure_pitches_template: List[pitch.Pitch] = []
            try:
                # Pass vocal_notes_in_block to generate_bass_measure
                # Also ensure cs_now_obj and cs_next_obj are not None before calling
                if cs_now_obj and cs_next_obj : # Ensure valid chord objects
                    temp_notes = generate_bass_measure(
                        style=selected_style,
                        cs_now=cs_now_obj,
                        cs_next=cs_next_obj,
                        tonic=tonic,
                        mode=mode,
                        octave=target_octave,
                        vocal_notes_in_block=vocal_notes_in_block
                    )
                    measure_pitches_template = [n.pitch for n in temp_notes if isinstance(n, note.Note)]
                else: # Fallback if cs_now_obj or cs_next_obj is still None
                    logger.warning(f"BassGenerator: cs_now_obj or cs_next_obj is None before calling generate_bass_measure for block {i+1}. Using default C root.")
                    default_root = pitch.Pitch('C3')
                    if cs_now_obj and cs_now_obj.root(): # Try to use current chord's root if available
                        default_root = cs_now_obj.root().transpose((target_octave - cs_now_obj.root().octave) * 12)
                    measure_pitches_template = [default_root] * 4


            except Exception as e_gbm:
                logger.error(f"BassGenerator: Error in generate_bass_measure for style '{selected_style}', chord '{chord_label_str}': {e_gbm}. Using root note.", exc_info=True)
                default_root_fallback = pitch.Pitch('C3')
                if cs_now_obj and cs_now_obj.root():
                     default_root_fallback = cs_now_obj.root().transpose((target_octave - cs_now_obj.root().octave) * 12)
                measure_pitches_template = [default_root_fallback] * 4


            rhythm_key = bass_params.get("rhythm_key", "bass_quarter_notes")
            rhythm_details = self.rhythm_library.get(rhythm_key)
            if rhythm_details is None:
                logger.warning(f"BassGenerator: Rhythm key '{rhythm_key}' not found in library. Falling back to 'bass_quarter_notes'.")
                rhythm_details = self.rhythm_library.get("bass_quarter_notes")
                if rhythm_details is None:
                    logger.error("BassGenerator: Fallback rhythm 'bass_quarter_notes' also not found! Using emergency fallback pattern.")
                    rhythm_details = { # Emergency fallback
                        "pattern": [{"offset": beat, "duration": 1.0, "velocity_factor": 0.7} for beat in range(4)],
                        "reference_duration_ql": 4.0
                    }

            pattern_events = rhythm_details.get("pattern", [])
            pattern_ref_duration = rhythm_details.get("reference_duration_ql", 4.0) # Assumed 4/4 bar if not specified

            pitch_idx = 0
            for event_data in pattern_events:
                event_offset_in_pattern = event_data.get("offset", 0.0)
                event_duration_from_pattern = event_data.get("duration", 1.0)

                # Scale rhythmic events to fit the actual duration of the chord block
                scale_factor = block_q_length / pattern_ref_duration if pattern_ref_duration > 0 else 1.0

                abs_event_offset_in_block = event_offset_in_pattern * scale_factor
                actual_event_duration = event_duration_from_pattern * scale_factor

                # Ensure note does not exceed block boundary or become too short
                if abs_event_offset_in_block >= block_q_length:
                    continue
                actual_event_duration = min(actual_event_duration, block_q_length - abs_event_offset_in_block)

                if actual_event_duration < MIN_NOTE_DURATION_QL / 2: # Avoid extremely short notes
                    continue

                current_pitch_obj: Optional[pitch.Pitch] = None
                if measure_pitches_template: # Use generated pitches if available
                    current_pitch_obj = measure_pitches_template[pitch_idx % len(measure_pitches_template)]
                    pitch_idx += 1
                else: # Fallback if measure_pitches_template is empty (should be rare now)
                    logger.warning(f"BassGenerator: measure_pitches_template is empty for block {i+1}, chord {chord_label_str}. Using default C3.")
                    current_pitch_obj = pitch.Pitch('C3')


                if current_pitch_obj is None: # Should ideally not happen
                    logger.error("BassGenerator: current_pitch_obj is None. This should not happen. Skipping note.")
                    continue

                n_bass = note.Note(current_pitch_obj)
                n_bass.quarterLength = actual_event_duration
                vel_factor = event_data.get("velocity_factor", 1.0)
                n_bass.volume = m21volume.Volume(velocity=int(base_velocity * vel_factor))
                bass_part.insert(current_total_offset + abs_event_offset_in_block, n_bass)

            current_total_offset += block_q_length

        # Apply humanization to the entire part once all notes are added
        global_bass_params = processed_blocks[0].get("part_params", {}).get("bass", {}) if processed_blocks else {}
        humanize_cfg = global_bass_params.get("humanization_settings", {}) # Assuming humanization params are nested
        if humanize_cfg.get("humanize_opt", False): # Check the specific opt flag
            h_template = humanize_cfg.get("template_name", "default_subtle")
            h_custom = humanize_cfg.get("custom_params", {})
            logger.info(f"BassGenerator: Applying humanization with template '{h_template}' and params {h_custom}")
            bass_part = apply_humanization_to_part(bass_part, template_name=h_template, custom_params=h_custom)
            bass_part.id = "Bass" # Ensure ID is reset after potential recreation by humanizer
            # Ensure essential elements are present after humanization
            if not bass_part.getElementsByClass(m21instrument.Instrument): bass_part.insert(0, self.default_instrument)
            if not bass_part.getElementsByClass(tempo.MetronomeMark): bass_part.insert(0, tempo.MetronomeMark(number=self.global_tempo))
            if not bass_part.getElementsByClass(meter.TimeSignature):
                ts_copy_humanize = meter.TimeSignature(self.global_time_signature_obj.ratioString)
                bass_part.insert(0, ts_copy_humanize)
            if not bass_part.getElementsByClass(key.Key):
                key_to_reinsert = key.Key(first_block_tonic, first_block_mode) # Re-fetch or use stored
                try:
                    bass_part.insert(0, key_to_reinsert)
                except Exception as e_key_reinsert:
                     logger.warning(f"BassGenerator: Could not re-insert key {first_block_tonic} {first_block_mode} after humanization: {e_key_reinsert}. Using C Major.")
                     bass_part.insert(0, key.Key("C", "major"))


        return bass_part
# --- END OF FILE generator/bass_generator.py (ボーカル協調フェーズ1対応) ---
