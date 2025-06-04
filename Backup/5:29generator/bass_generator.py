# --- START OF FILE generator/bass_generator.py (algorithmic_chord_tone_quarters強化 v1) ---
import music21
import logging
from music21 import stream, note, chord as m21chord, harmony, pitch, tempo, meter, instrument as m21instrument, key, interval, scale
import random
from typing import List, Dict, Optional, Any, Tuple, Union, cast, Sequence

try:
    from utilities.core_music_utils import get_time_signature_object, sanitize_chord_label, MIN_NOTE_DURATION_QL, _ROOT_RE_STRICT
    from utilities.humanizer import apply_humanization_to_part, HUMANIZATION_TEMPLATES
    from utilities.scale_registry import ScaleRegistry
except ImportError as e:
    print(f"BassGenerator: Warning - could not import all utilities: {e}")
    MIN_NOTE_DURATION_QL = 0.125 
    def get_time_signature_object(ts_str: Optional[str]) -> meter.TimeSignature: return meter.TimeSignature(ts_str or "4/4")
    def sanitize_chord_label(label: Optional[str]) -> Optional[str]: return label
    def apply_humanization_to_part(part, template_name=None, custom_params=None): return part
    HUMANIZATION_TEMPLATES = {}
    class ScaleRegistry: 
        @staticmethod
        def get(tonic_str: Optional[str], mode_str: Optional[str]) -> music21.scale.ConcreteScale:
            return music21.scale.MajorScale(tonic_str or "C")
    import re
    _ROOT_RE_STRICT = re.compile(r'^([A-G](?:[#b]{1,2}|[ns])?)(?![#b])')


class BassGenerator:
    def __init__(self, rhythm_library: Optional[Dict[str, Dict]] = None, default_instrument: m21instrument.Instrument = m21instrument.AcousticBass(), global_tempo: int = 120, global_time_signature: str = "4/4", global_key_tonic: str = "C", global_key_mode: str = "major", rng_seed: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        self.bass_rhythm_library = rhythm_library if rhythm_library else {}
        if not self.bass_rhythm_library: self.logger.warning("BassGenerator: No bass rhythm patterns provided in the rhythm library. Using internal defaults only.")
        self.default_instrument = default_instrument; self.global_tempo = global_tempo; self.global_time_signature_str = global_time_signature
        try: self.global_time_signature_obj = get_time_signature_object(global_time_signature)
        except Exception as e_ts_init: self.logger.error(f"BassGenerator init: Error initializing time signature from '{global_time_signature}': {e_ts_init}. Defaulting to 4/4.", exc_info=True); self.global_time_signature_obj = meter.TimeSignature("4/4")
        self.global_key_tonic = global_key_tonic; self.global_key_mode = global_key_mode
        if rng_seed is not None: self.rng = random.Random(rng_seed)
        else: self.rng = random.Random()
        if "basic_chord_tone_quarters" not in self.bass_rhythm_library:
            self.bass_rhythm_library["basic_chord_tone_quarters"] = {
                "description": "Basic quarter note pattern emphasizing root and fifth/third on strong beats. Supports weak_beat_style option.",
                "tags": ["bass", "default", "quarter_notes", "chord_tones", "algorithmic"],
                "pattern_type": "algorithmic_chord_tone_quarters", 
                "options": {
                    "base_velocity": 85,
                    "strong_beat_velocity_boost": 15, 
                    "off_beat_velocity_reduction": 5,  
                    "target_octave": 2,
                    "weak_beat_style": "root" # "root", "third_or_fifth", "eighth_roots"
                }
            }
            self.logger.info("BassGenerator: Added 'basic_chord_tone_quarters' algorithmic pattern to internal library.")
        if "bass_quarter_notes" not in self.bass_rhythm_library: self.bass_rhythm_library["bass_quarter_notes"] = {"description": "Default bass line - simple quarter notes on root.", "tags": ["bass", "default", "quarter_notes", "root"], "pattern": [{"offset": 0.0, "duration": 1.0, "velocity_factor": 0.75, "type": "root"}, {"offset": 1.0, "duration": 1.0, "velocity_factor": 0.7, "type": "root"}, {"offset": 2.0, "duration": 1.0, "velocity_factor": 0.75, "type": "root"}, {"offset": 3.0, "duration": 1.0, "velocity_factor": 0.7, "type": "root"}]}
        if "root_only" not in self.bass_rhythm_library: self.bass_rhythm_library["root_only"] = {"description": "Fallback whole-note roots only.", "tags": ["bass", "fallback", "long_notes", "root"], "pattern_type": "algorithmic_root_only", "pattern": [{"offset": 0.0, "duration": 4.0, "velocity_factor": 0.6}]}

    def _get_rhythm_pattern_details(self, rhythm_key: str) -> Dict[str, Any]:
        default_rhythm = self.bass_rhythm_library.get("basic_chord_tone_quarters"); details = self.bass_rhythm_library.get(rhythm_key)
        if not details: self.logger.warning(f"Bass rhythm key '{rhythm_key}' not found. Using 'basic_chord_tone_quarters' as fallback."); return default_rhythm if default_rhythm else {}
        if "pattern_type" in details and isinstance(details["pattern_type"], str):
            if "pattern" not in details: details["pattern"] = [] 
        elif "pattern" not in details or not isinstance(details["pattern"], list): self.logger.warning(f"Bass rhythm key '{rhythm_key}' has no 'pattern_type' and invalid or missing 'pattern'. Using fallback 'basic_chord_tone_quarters'."); return default_rhythm if default_rhythm else {}
        return details

    def _get_bass_pitch_in_octave(self, base_pitch_obj: Optional[pitch.Pitch], target_octave: int) -> int:
        if not base_pitch_obj: return pitch.Pitch(f"C{target_octave}").midi 
        p_new = pitch.Pitch(base_pitch_obj.name); p_new.octave = target_octave; min_bass_midi = 28; max_bass_midi = 60; current_midi = p_new.midi
        while current_midi < min_bass_midi: current_midi += 12
        while current_midi > max_bass_midi:
            if current_midi - 12 >= min_bass_midi: current_midi -= 12
            else: break 
        return max(min_bass_midi, min(current_midi, max_bass_midi))

    def _generate_notes_from_fixed_pattern( self, pattern: List[Dict[str, Any]], m21_cs: harmony.ChordSymbol, base_velocity: int, target_octave: int, block_offset_ignored: float, block_duration: float, current_scale: Optional[music21.scale.ConcreteScale] = None) -> List[Tuple[float, music21.note.Note]]:
        notes: List[Tuple[float, music21.note.Note]] = []
        if not m21_cs or not m21_cs.pitches: return notes
        root_pitch_obj = m21_cs.root(); third_pitch_obj = m21_cs.third; fifth_pitch_obj = m21_cs.fifth; chord_tones = [p for p in [root_pitch_obj, third_pitch_obj, fifth_pitch_obj] if p]
        for p_event in pattern:
            offset_in_block = p_event.get("offset", 0.0); duration_ql = p_event.get("duration", 1.0); vel_factor = p_event.get("velocity_factor", 1.0); note_type = p_event.get("type", "root").lower()
            final_velocity = max(1, min(127, int(base_velocity * vel_factor))); chosen_pitch_base: Optional[pitch.Pitch] = None
            if note_type == "root" and root_pitch_obj: chosen_pitch_base = root_pitch_obj
            elif note_type == "fifth" and fifth_pitch_obj: chosen_pitch_base = fifth_pitch_obj
            elif note_type == "third" and third_pitch_obj: chosen_pitch_base = third_pitch_obj
            elif note_type == "octave_root" and root_pitch_obj: chosen_pitch_base = root_pitch_obj
            elif note_type == "random_chord_tone" and chord_tones: chosen_pitch_base = self.rng.choice(chord_tones)
            elif note_type == "scale_tone" and current_scale:
                try: p_s = current_scale.pitchFromDegree(self.rng.choice([1,2,3,4,5,6,7]))
                except Exception: p_s = None
                if p_s: chosen_pitch_base = p_s
                elif root_pitch_obj: chosen_pitch_base = root_pitch_obj
            elif note_type == "approach" and root_pitch_obj: chosen_pitch_base = root_pitch_obj.transpose(-1)
            if not chosen_pitch_base and root_pitch_obj: chosen_pitch_base = root_pitch_obj
            if chosen_pitch_base:
                midi_pitch = self._get_bass_pitch_in_octave(chosen_pitch_base, target_octave); n = music21.note.Note(); n.pitch.midi = midi_pitch
                n.duration = music21.duration.Duration(duration_ql); n.volume.velocity = final_velocity
                if p_event.get("glide_to_next", False): n.tie = music21.tie.Tie("start")
                notes.append((offset_in_block, n))
        return notes

    def _generate_algorithmic_pattern(self, pattern_type: str, m21_cs: harmony.ChordSymbol, options: Dict[str, Any], base_velocity: int, target_octave: int, block_offset_ignored: float, block_duration: float, current_scale: music21.scale.ConcreteScale) -> List[Tuple[float, music21.note.Note]]:
        notes: List[Tuple[float, music21.note.Note]] = []
        if not m21_cs or not m21_cs.pitches: return notes
        root_note_obj = m21_cs.root()
        if not root_note_obj: return notes
        
        if pattern_type == "algorithmic_chord_tone_quarters":
            strong_beat_vel_boost = options.get("strong_beat_velocity_boost", 15)
            off_beat_vel_reduction = options.get("off_beat_velocity_reduction", 5)
            weak_beat_style = options.get("weak_beat_style", "root") # ★★★ 変更点: weak_beat_style を取得 ★★★
            next_chord_root: Optional[pitch.Pitch] = options.get("next_chord_root") # ★★★ 変更点: next_chord_root を取得 ★★★

            beats_per_measure_in_block = self.global_time_signature_obj.beatCount 
            num_beats_to_generate = int(block_duration)

            for beat_idx in range(num_beats_to_generate):
                current_rel_offset = beat_idx * 1.0 
                if current_rel_offset >= block_duration - MIN_NOTE_DURATION_QL / 2 : break
                
                chosen_pitch_base: Optional[pitch.Pitch] = None
                current_velocity = base_velocity 
                beat_in_measure = beat_idx % beats_per_measure_in_block
                
                note_duration_ql = 1.0 # 基本は四分音符

                if beat_in_measure == 0 : # 1拍目 (小節の頭)
                    chosen_pitch_base = root_note_obj
                    current_velocity = min(127, base_velocity + strong_beat_vel_boost)
                elif beats_per_measure_in_block >= 4 and beat_in_measure == (beats_per_measure_in_block // 2) : # 3拍目など
                    # ★★★ 変更点: 3拍目の音選択ロジック強化 ★★★
                    approach_candidates: List[pitch.Pitch] = []
                    if next_chord_root and current_scale:
                        # 半音下からのアプローチ (スケール内)
                        app_chrom_lower = next_chord_root.transpose(-1)
                        if current_scale.getScaleDegreeFromPitch(app_chrom_lower) is not None:
                            approach_candidates.append(app_chrom_lower)
                        # 全音下からのアプローチ (スケール内)
                        app_diatonic_lower = next_chord_root.transpose(-2)
                        if current_scale.getScaleDegreeFromPitch(app_diatonic_lower) is not None:
                             # 半音アプローチと重複しないように
                            if not any(p.nameWithOctave == app_diatonic_lower.nameWithOctave for p in approach_candidates):
                                approach_candidates.append(app_diatonic_lower)
                    
                    if approach_candidates:
                        chosen_pitch_base = self.rng.choice(approach_candidates)
                        self.logger.debug(f"Chord {m21_cs.figure}, Beat 3: Chose approach note {chosen_pitch_base.nameWithOctave} to next root {next_chord_root.nameWithOctave if next_chord_root else 'N/A'}")
                    elif m21_cs.fifth: chosen_pitch_base = m21_cs.fifth
                    elif m21_cs.third: chosen_pitch_base = m21_cs.third
                    else: chosen_pitch_base = root_note_obj
                    current_velocity = min(127, base_velocity + (strong_beat_vel_boost // 2))
                else: # 2拍目、4拍目などの弱拍
                    # ★★★ 変更点: weak_beat_style に応じた処理 ★★★
                    current_velocity = max(1, base_velocity - off_beat_vel_reduction)
                    if weak_beat_style == "third_or_fifth":
                        third = m21_cs.third
                        fifth = m21_cs.fifth
                        options_for_weak_beat = []
                        if third: options_for_weak_beat.append(third)
                        if fifth: options_for_weak_beat.append(fifth)
                        if options_for_weak_beat: chosen_pitch_base = self.rng.choice(options_for_weak_beat)
                        else: chosen_pitch_base = root_note_obj
                    elif weak_beat_style == "eighth_roots":
                        chosen_pitch_base = root_note_obj
                        note_duration_ql = 0.5 # 8分音符に変更
                        # 1つ目の8分音符
                        if chosen_pitch_base:
                            midi_pitch1 = self._get_bass_pitch_in_octave(chosen_pitch_base, target_octave)
                            n1 = music21.note.Note(); n1.pitch.midi = midi_pitch1
                            n1.duration.quarterLength = note_duration_ql
                            n1.volume.velocity = current_velocity
                            if current_rel_offset + note_duration_ql <= block_duration: # ブロック内に収まるか
                                notes.append((current_rel_offset, n1))
                        # 2つ目の8分音符
                        if chosen_pitch_base and (current_rel_offset + note_duration_ql < block_duration - MIN_NOTE_DURATION_QL / 2):
                            midi_pitch2 = self._get_bass_pitch_in_octave(chosen_pitch_base, target_octave)
                            n2 = music21.note.Note(); n2.pitch.midi = midi_pitch2
                            n2.duration.quarterLength = note_duration_ql
                            n2.volume.velocity = max(1, current_velocity - 5) # 少しベロシティ変化
                            notes.append((current_rel_offset + note_duration_ql, n2))
                        chosen_pitch_base = None # この後は通常のノート追加処理をスキップ
                    else: # "root" または未指定
                        chosen_pitch_base = root_note_obj
                
                if chosen_pitch_base: # eighth_roots以外の場合、またはeighth_rootsでノートがまだ追加されていない場合
                    midi_pitch = self._get_bass_pitch_in_octave(chosen_pitch_base, target_octave)
                    n = music21.note.Note(); n.pitch.midi = midi_pitch
                    actual_note_duration = min(note_duration_ql, block_duration - current_rel_offset)
                    if actual_note_duration < MIN_NOTE_DURATION_QL: continue
                    n.duration.quarterLength = actual_note_duration
                    n.volume.velocity = current_velocity
                    notes.append((current_rel_offset, n))
        
        # ... (他のアルゴリズムパターンは変更なし) ...
        elif pattern_type == "algorithmic_root_only":
            note_duration_ql = options.get("note_duration_ql", block_duration);
            if note_duration_ql <= 0: note_duration_ql = block_duration
            num_notes = int(block_duration / note_duration_ql) if note_duration_ql > 0 else 0
            for i in range(num_notes): current_rel_offset = i * note_duration_ql; midi_pitch = self._get_bass_pitch_in_octave(root_note_obj, target_octave); n = music21.note.Note(); n.pitch.midi = midi_pitch; n.duration.quarterLength = note_duration_ql; n.volume.velocity = base_velocity; notes.append((current_rel_offset, n))
        # (以下、他のアルゴリズムパターンの実装が続く)
        else: # フォールバック
            default_algo_options = self.bass_rhythm_library.get("basic_chord_tone_quarters",{}).get("options",{})
            # フォールバック時も next_chord_root を渡すようにする (あれば)
            default_algo_options["next_chord_root"] = options.get("next_chord_root")
            default_algo_options["weak_beat_style"] = options.get("weak_beat_style", "root") # weak_beat_styleも引き継ぐ
            notes.extend(self._generate_algorithmic_pattern("algorithmic_chord_tone_quarters", m21_cs, default_algo_options, base_velocity, target_octave, 0.0, block_duration, current_scale))
        return notes

    def compose(self, processed_blocks: Sequence[Dict[str, Any]], return_pretty_midi: bool = False) -> Union[stream.Part, Any]:
        bass_part = stream.Part(id="Bass"); bass_part.insert(0, self.default_instrument)
        if self.global_tempo: bass_part.insert(0, tempo.MetronomeMark(number=self.global_tempo))
        if self.global_time_signature_obj: bass_part.insert(0, meter.TimeSignature(self.global_time_signature_obj.ratioString))
        else: bass_part.insert(0, meter.TimeSignature("4/4"))
        if processed_blocks:
            first_block_tonic = processed_blocks[0].get("tonic_of_section", self.global_key_tonic); first_block_mode = processed_blocks[0].get("mode", self.global_key_mode)
            try:
                if not bass_part.getElementsByClass(key.Key).first(): bass_part.insert(0, key.Key(first_block_tonic, first_block_mode.lower()))
            except Exception:
                 if not bass_part.getElementsByClass(key.Key).first(): bass_part.insert(0, key.Key(self.global_key_tonic, self.global_key_mode.lower()))
        part_overall_humanize_params = None

        for blk_idx, blk_data in enumerate(processed_blocks):
            bass_params_for_block = blk_data.get("part_params", {}).get("bass", {})
            if not bass_params_for_block: bass_params_for_block = {"rhythm_key": "basic_chord_tone_quarters", "velocity": 70, "octave": 2}; self.logger.debug(f"BassGenerator: No bass params for block {blk_idx+1}. Using hardcoded defaults.")
            if blk_idx == 0: part_overall_humanize_params = {"humanize_opt": bass_params_for_block.get("humanize_opt", True), "template_name": bass_params_for_block.get("template_name", "default_subtle"), "custom_params": bass_params_for_block.get("custom_params", {})}
            
            chord_label_str = blk_data.get("chord_label", "C"); block_q_length = float(blk_data.get("q_length", 4.0)); block_abs_offset = float(blk_data.get("offset", 0.0))
            m21_cs_obj: Optional[harmony.ChordSymbol] = None; sanitized_label : Optional[str] = None; parse_failure_reason = "Unknown parse error"
            try:
                sanitized_label = sanitize_chord_label(chord_label_str)
                if sanitized_label and sanitized_label.lower() != "rest": m21_cs_obj = harmony.ChordSymbol(sanitized_label);
                elif sanitized_label and sanitized_label.lower() == "rest": self.logger.info(f"BassGen: Block {blk_idx+1} is a Rest."); continue
                else: parse_failure_reason = f"sanitize_chord_label returned None for '{chord_label_str}'"
            except harmony.HarmonyException as e_harm: parse_failure_reason = f"HarmonyException for '{sanitized_label}': {e_harm}"
            except Exception as e_chord_parse: parse_failure_reason = f"Unexpected parse error for '{sanitized_label}': {e_chord_parse}"
            if not m21_cs_obj and sanitized_label and sanitized_label.lower() != "rest":
                self.logger.warning(f"BassGen: Chord '{chord_label_str}' (sanitized: '{sanitized_label}') could not be fully parsed ({parse_failure_reason}). Attempting root-only fallback.")
                root_only_match = _ROOT_RE_STRICT.match(sanitized_label if sanitized_label else "")
                if root_only_match:
                    try: m21_cs_obj = harmony.ChordSymbol(root_only_match.group(0)); self.logger.info(f"BassGen: Fallback to root: '{m21_cs_obj.figure}' for original '{chord_label_str}'.")
                    except Exception as e_root_parse: self.logger.error(f"BassGen: Could not even parse root '{root_only_match.group(0)}' from '{chord_label_str}': {e_root_parse}. Skipping block."); self.logger.warning(f'Skipped block {blk_idx+1} in compose: chord="{chord_label_str}", reason="Root parse failed after sanitize failure ({parse_failure_reason})"') ; continue
                else: self.logger.error(f"BassGen: Could not extract root from '{sanitized_label}' after sanitize failure ({parse_failure_reason}). Skipping block."); self.logger.warning(f'Skipped block {blk_idx+1} in compose: chord="{chord_label_str}", reason="No root found after sanitize failure ({parse_failure_reason})"') ; continue
            elif not m21_cs_obj and (not sanitized_label or sanitized_label.lower() != "rest"): self.logger.error(f"BassGen: No valid chord or root for block {blk_idx+1} (Label: '{chord_label_str}', Sanitized: '{sanitized_label}', Reason: {parse_failure_reason}). Skipping."); self.logger.warning(f'Skipped block {blk_idx+1} in compose: chord="{chord_label_str}", reason="Final parse/root extraction failed ({parse_failure_reason})"') ; continue
            
            rhythm_key_from_params = bass_params_for_block.get("rhythm_key", bass_params_for_block.get("style", "basic_chord_tone_quarters")) 
            pattern_details = self._get_rhythm_pattern_details(rhythm_key_from_params)
            if not pattern_details: self.logger.warning(f'Skipped block {blk_idx+1} in compose: chord="{chord_label_str}", reason="No pattern_details for rhythm_key {rhythm_key_from_params}"'); continue
            base_vel = bass_params_for_block.get("velocity", 70); target_oct = bass_params_for_block.get("octave", 2)  
            section_tonic = blk_data.get("tonic_of_section", self.global_key_tonic); section_mode = blk_data.get("mode", self.global_key_mode)
            current_m21_scale = ScaleRegistry.get(section_tonic, section_mode)
            if not current_m21_scale: current_m21_scale = music21.scale.MajorScale("C")
            
            # ★★★ 変更点: 次のブロックのコード情報を取得 ★★★
            next_chord_root_pitch: Optional[pitch.Pitch] = None
            if blk_idx + 1 < len(processed_blocks):
                next_blk_data = processed_blocks[blk_idx + 1]
                next_chord_label_str = next_blk_data.get("chord_label")
                if next_chord_label_str:
                    next_sanitized_label = sanitize_chord_label(next_chord_label_str)
                    if next_sanitized_label and next_sanitized_label.lower() != "rest":
                        try:
                            next_m21_cs_obj = harmony.ChordSymbol(next_sanitized_label)
                            if next_m21_cs_obj.root():
                                next_chord_root_pitch = next_m21_cs_obj.root()
                        except Exception:
                            pass # 次のコードのパースに失敗しても、現在のブロックの処理は続行

            generated_notes_for_block: List[Tuple[float, music21.note.Note]] = []
            if m21_cs_obj: 
                if "pattern_type" in pattern_details and isinstance(pattern_details["pattern_type"], str):
                    algo_options = pattern_details.get("options", {}).copy() # ★★★ .copy() を追加 ★★★
                    algo_options["base_velocity"] = base_vel; algo_options["target_octave"] = target_oct
                    algo_options["next_chord_root"] = next_chord_root_pitch # ★★★ 次のコードのルートを渡す ★★★
                    # ★★★ weak_beat_style も chordmap から取得できるようにする (なければデフォルト) ★★★
                    algo_options["weak_beat_style"] = bass_params_for_block.get("weak_beat_style", algo_options.get("weak_beat_style", "root"))

                    generated_notes_for_block = self._generate_algorithmic_pattern( pattern_details["pattern_type"], m21_cs_obj, algo_options, base_vel, target_oct, 0.0, block_q_length, current_m21_scale)
                elif "pattern" in pattern_details and isinstance(pattern_details["pattern"], list):
                    generated_notes_for_block = self._generate_notes_from_fixed_pattern( pattern_details["pattern"], m21_cs_obj, base_vel, target_oct, 0.0, block_q_length, current_m21_scale)
            
            for rel_offset, note_obj in generated_notes_for_block: 
                abs_note_offset = block_abs_offset + rel_offset   
                end_of_note = abs_note_offset + note_obj.duration.quarterLength
                end_of_block = block_abs_offset + block_q_length
                if end_of_note > end_of_block + 0.001: 
                    new_dur = end_of_block - abs_note_offset
                    if new_dur > MIN_NOTE_DURATION_QL / 2: note_obj.duration.quarterLength = new_dur
                    else: continue
                
                log_chord_figure = m21_cs_obj.figure if m21_cs_obj else "N/A_or_Rest"
                self.logger.info(f"DEBUG Bass Insert: Chord='{log_chord_figure}', BlockOffset={block_abs_offset:.2f}, RelOffset={rel_offset:.2f}, NoteAbsOffset={abs_note_offset:.2f}, Pitch={note_obj.pitch.nameWithOctave if note_obj.pitch else 'N/A'}, Duration={note_obj.duration.quarterLength:.2f}")
                
                if note_obj.duration.quarterLength >= MIN_NOTE_DURATION_QL / 2:
                     bass_part.insert(abs_note_offset, note_obj)
                else: self.logger.debug(f"BassGen: Final note for {log_chord_figure} at {abs_note_offset:.2f} is too short. Skipping.")

        if part_overall_humanize_params and part_overall_humanize_params.get("humanize_opt", False):
            try:
                bass_part_humanized = apply_humanization_to_part(bass_part, template_name=part_overall_humanize_params.get("template_name"), custom_params=part_overall_humanize_params.get("custom_params"))
                if isinstance(bass_part_humanized, stream.Part): bass_part_humanized.id = "Bass";
            except Exception as e_hum: self.logger.error(f"BassGen: Error during bass part humanization: {e_hum}", exc_info=True)
        if return_pretty_midi: return None 
        return bass_part
# --- END OF FILE generator/bass_generator.py ---
