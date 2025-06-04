# --- START OF FILE generator/guitar_generator.py (ヒューマナイズ外部化・Rest対応版) ---
import music21
from typing import List, Dict, Optional, Tuple, Any, Sequence, Union, cast

# music21 のサブモジュールを正しい形式でインポート
import music21.stream as stream
import music21.note as note
import music21.harmony as harmony
import music21.pitch as pitch
import music21.meter as meter
import music21.duration as duration
import music21.instrument as m21instrument
import music21.scale as scale
import music21.interval as interval
import music21.tempo as tempo
import music21.key as key
import music21.chord      as m21chord
import music21.articulations as articulations
import music21.volume as m21volume
import music21.expressions as expressions

import random
import logging

# ユーティリティのインポート
try:
    from utilities.core_music_utils import MIN_NOTE_DURATION_QL, get_time_signature_object, sanitize_chord_label
    from utilities.humanizer import apply_humanization_to_part, HUMANIZATION_TEMPLATES
except ImportError:
    logger_fallback = logging.getLogger(__name__ + ".fallback_utils")
    logger_fallback.warning("GuitarGen: Could not import from utilities. Using fallbacks.")
    MIN_NOTE_DURATION_QL = 0.125
    def get_time_signature_object(ts_str: Optional[str]) -> meter.TimeSignature:
        if not ts_str: ts_str = "4/4"; return meter.TimeSignature(ts_str)
        try: return meter.TimeSignature(ts_str)
        except: return meter.TimeSignature("4/4")
    def sanitize_chord_label(label: Optional[str]) -> Optional[str]:
        if not label or label.strip().lower() in ["rest", "n.c.", "nc", "none"]: return None
        return label.strip()
    def apply_humanization_to_part(part, template_name=None, custom_params=None): return part
    HUMANIZATION_TEMPLATES = {}

logger = logging.getLogger(__name__)

DEFAULT_GUITAR_OCTAVE_RANGE: Tuple[int, int] = (2, 5)
GUITAR_STRUM_DELAY_QL: float = 0.02
MIN_STRUM_NOTE_DURATION_QL: float = 0.05
STYLE_BLOCK_CHORD = "block_chord"; STYLE_STRUM_BASIC = "strum_basic"; STYLE_ARPEGGIO = "arpeggio"
STYLE_POWER_CHORDS = "power_chords"; STYLE_MUTED_RHYTHM = "muted_rhythm"; STYLE_SINGLE_NOTE_LINE = "single_note_line"


class GuitarGenerator:
    def __init__(self,
                 rhythm_library: Optional[Dict[str, Dict]] = None,
                 default_instrument=m21instrument.AcousticGuitar(),
                 global_tempo: int = 120,
                 global_time_signature: str = "4/4"):
        self.rhythm_library = rhythm_library if rhythm_library is not None else {}
        if "guitar_default_quarters" not in self.rhythm_library:
             self.rhythm_library["guitar_default_quarters"] = {"description": "Default quarter note strums/hits", "pattern": [{"offset":i, "duration":1.0, "velocity_factor":0.8-(i%2*0.05)} for i in range(4)]}
             logger.info("GuitarGen: Added 'guitar_default_quarters' to rhythm_library.")
        self.default_instrument = default_instrument
        self.global_tempo = global_tempo
        self.global_time_signature_str = global_time_signature
        self.global_time_signature_obj = get_time_signature_object(global_time_signature)

    def _get_guitar_friendly_voicing(
        self, cs: harmony.ChordSymbol, num_strings: int = 6,
        preferred_octave_bottom: int = 2, max_octave_top: int = 5, # voicing_style 引数を削除 (未使用のため)
        # voicing_style: str = "standard" # この引数は内部で使用されていないため削除
    ) -> List[pitch.Pitch]:
        if not cs or not cs.pitches: return []
        original_pitches = list(cs.pitches); root = cs.root()
        voiced_pitches: List[pitch.Pitch] = []

        # voicing_style の代わりに直接的なロジックを使用、または外部から渡されることを想定
        # ここでは voicing_style == "power_chord_root_fifth" のような分岐は削除し、
        # より汎用的なボイシングを目指すか、このメソッドを呼び出す側でスタイルを考慮する

        try:
            # デフォルトのボイシングとしてクローズドポジションを使用
            temp_chord = cs.closedPosition(forceOctave=preferred_octave_bottom, inPlace=False)
            candidate_pitches = sorted(list(temp_chord.pitches), key=lambda p_sort: p_sort.ps)
        except Exception:
            candidate_pitches = sorted(original_pitches, key=lambda p_sort:p_sort.ps)

        if not candidate_pitches: return []

        # ギターの最低音E2 (MIDI 40) より低い音は調整
        guitar_min_ps = pitch.Pitch(f"E{DEFAULT_GUITAR_OCTAVE_RANGE[0]}").ps
        if candidate_pitches[0].ps < guitar_min_ps:
            oct_shift = math.ceil((guitar_min_ps - candidate_pitches[0].ps) / 12.0)
            candidate_pitches = [p_cand.transpose(oct_shift * 12) for p_cand in candidate_pitches]
            candidate_pitches.sort(key=lambda p_sort: p_sort.ps)
        
        selected_dict: Dict[str, pitch.Pitch] = {} # 重複する音名を除外するため辞書を使用
        for p_cand_select in candidate_pitches:
             # ギターの音域内に収まるように調整 (E2～G5程度を想定)
            if pitch.Pitch(f"E{DEFAULT_GUITAR_OCTAVE_RANGE[0]}").ps <= p_cand_select.ps <= pitch.Pitch(f"G{DEFAULT_GUITAR_OCTAVE_RANGE[1]}").ps:
                if p_cand_select.nameWithOctave not in selected_dict: # オクターブも含めてユニークにする
                     selected_dict[p_cand_select.nameWithOctave] = p_cand_select

        voiced_pitches = sorted(list(selected_dict.values()), key=lambda p_sort:p_sort.ps)
        return voiced_pitches[:num_strings] # 弦の数に合わせる


    def _create_notes_from_event(
        self, cs: harmony.ChordSymbol, guitar_params: Dict[str, Any],
        event_abs_offset: float, event_duration_ql: float, event_velocity: int
    ) -> List[Union[note.Note, m21chord.Chord]]:
        notes_for_event: List[Union[note.Note, m21chord.Chord]] = []
        style = guitar_params.get("guitar_style", STYLE_BLOCK_CHORD)

        num_strings = guitar_params.get("guitar_num_strings", 6)
        preferred_octave = guitar_params.get("guitar_target_octave", guitar_params.get("target_octave",3)) # target_octaveも考慮
        # voicing_style_name = guitar_params.get("guitar_voicing_style", "standard") # _get_guitar_friendly_voicingから削除したため

        # voicing_style は _get_guitar_friendly_voicing の内部ロジックとして一旦含めない
        chord_pitches = self._get_guitar_friendly_voicing(cs, num_strings, preferred_octave)
        if not chord_pitches: return []

        if style == STYLE_POWER_CHORDS and cs.root(): # STYLE_POWER_CHORDS を明示的に処理
            p_root = pitch.Pitch(cs.root().name)
            # 適切なオクターブに調整
            while p_root.octave < DEFAULT_GUITAR_OCTAVE_RANGE[0]: p_root.octave +=1
            while p_root.octave > DEFAULT_GUITAR_OCTAVE_RANGE[0]+1: p_root.octave -=1 # 低音弦を優先
            
            power_chord_pitches = [p_root, p_root.transpose(interval.PerfectFifth())]
            if num_strings > 2: # オクターブ上のルート音を追加 (3弦以上の場合)
                power_chord_pitches.append(p_root.transpose(interval.PerfectOctave()))
            
            ch = m21chord.Chord(power_chord_pitches[:num_strings], quarterLength=event_duration_ql * 0.95) # 持続を少し短く
            for n_in_ch_note in ch.notes: n_in_ch_note.volume.velocity = event_velocity
            ch.offset = event_abs_offset
            notes_for_event.append(ch)
            return notes_for_event


        if style == STYLE_BLOCK_CHORD:
            ch = m21chord.Chord(chord_pitches, quarterLength=event_duration_ql * 0.9)
            for n_in_ch_note in ch.notes: n_in_ch_note.volume.velocity = event_velocity
            ch.offset = event_abs_offset
            notes_for_event.append(ch)
        elif style == STYLE_STRUM_BASIC:
            is_down = guitar_params.get("strum_direction", "down").lower() == "down"
            play_order = list(reversed(chord_pitches)) if is_down else chord_pitches
            for i, p_obj_strum in enumerate(play_order):
                n_strum = note.Note(p_obj_strum)
                n_strum.duration = duration.Duration(quarterLength=max(MIN_STRUM_NOTE_DURATION_QL, event_duration_ql * 0.9))
                n_strum.offset = event_abs_offset + (i * GUITAR_STRUM_DELAY_QL)
                vel_adj_range = 10
                vel_adj = int(((len(play_order)-1-i)/(len(play_order)-1)*vel_adj_range)-(vel_adj_range/2)) if is_down and len(play_order)>1 else (int((i/(len(play_order)-1)*vel_adj_range)-(vel_adj_range/2)) if len(play_order)>1 else 0)
                n_strum.volume = m21volume.Volume(velocity=max(1, min(127, event_velocity + vel_adj)))
                notes_for_event.append(n_strum)
        elif style == STYLE_ARPEGGIO:
            arp_pattern_type = guitar_params.get("arpeggio_type", "up")
            arp_note_dur_ql = guitar_params.get("arpeggio_note_duration_ql", 0.5)

            ordered_arp_pitches: List[pitch.Pitch] = []
            if isinstance(arp_pattern_type, list): # 数値インデックスのリストの場合
                 ordered_arp_pitches = [chord_pitches[idx % len(chord_pitches)] for idx in arp_pattern_type if chord_pitches]
            elif arp_pattern_type == "down":
                ordered_arp_pitches = list(reversed(chord_pitches))
            elif arp_pattern_type == "up_down" and len(chord_pitches) > 2 :
                ordered_arp_pitches = chord_pitches + list(reversed(chord_pitches[1:-1]))
            else: # "up" or default
                ordered_arp_pitches = chord_pitches

            current_offset_in_event = 0.0; arp_idx = 0
            while current_offset_in_event < event_duration_ql and ordered_arp_pitches:
                p_play_arp = ordered_arp_pitches[arp_idx % len(ordered_arp_pitches)]
                actual_arp_dur = min(arp_note_dur_ql, event_duration_ql - current_offset_in_event)
                if actual_arp_dur < MIN_NOTE_DURATION_QL / 4: break
                n_arp = note.Note(p_play_arp, quarterLength=actual_arp_dur * 0.95)
                n_arp.volume = m21volume.Volume(velocity=event_velocity)
                n_arp.offset = event_abs_offset + current_offset_in_event
                notes_for_event.append(n_arp)
                current_offset_in_event += arp_note_dur_ql; arp_idx += 1
        elif style == STYLE_MUTED_RHYTHM:
            mute_note_dur = guitar_params.get("mute_note_duration_ql", 0.1)
            mute_interval = guitar_params.get("mute_interval_ql", 0.25)
            t_mute = 0.0
            if not chord_pitches: return [] # ミュート対象の音がない場合
            # ミュート音はルート音または最低音を使用
            mute_base_pitch = chord_pitches[0]
            while t_mute < event_duration_ql:
                actual_mute_dur = min(mute_note_dur, event_duration_ql - t_mute)
                if actual_mute_dur < MIN_NOTE_DURATION_QL / 8: break
                n_mute = note.Note(mute_base_pitch); n_mute.articulations = [articulations.Staccatissimo()]
                n_mute.duration.quarterLength = actual_mute_dur
                n_mute.volume = m21volume.Volume(velocity=int(event_velocity * 0.6) + random.randint(-5,5))
                n_mute.offset = event_abs_offset + t_mute
                notes_for_event.append(n_mute)
                t_mute += mute_interval
        return notes_for_event


    def compose(self, processed_chord_stream: List[Dict]) -> stream.Part:
        guitar_part = stream.Part(id="Guitar")
        guitar_part.insert(0, self.default_instrument)
        guitar_part.insert(0, tempo.MetronomeMark(number=self.global_tempo))
        ts_copy_init = meter.TimeSignature(self.global_time_signature_obj.ratioString)
        guitar_part.insert(0, ts_copy_init)

        if not processed_chord_stream: return guitar_part
        logger.info(f"GuitarGen: Starting for {len(processed_chord_stream)} blocks.")

        all_generated_elements_for_part: List[Union[note.Note, m21chord.Chord]] = []

        for blk_idx, blk_data in enumerate(processed_chord_stream):
            block_offset_ql = float(blk_data.get("offset", 0.0))
            block_duration_ql = float(blk_data.get("q_length", 4.0))
            chord_label_str = blk_data.get("chord_label", "C")
            guitar_params = blk_data.get("part_params", {}).get("guitar", {})
            if not guitar_params: continue

            # ★★★ "Rest" ラベルの明示的な処理 ★★★
            if chord_label_str.lower() == "rest":
                logger.info(f"GuitarGen: Block {blk_idx+1} ('{chord_label_str}') is a Rest. Skipping guitar notes for this block.")
                continue
            # ★★★ ここまで ★★★

            sanitized_label = sanitize_chord_label(chord_label_str)
            cs_object: Optional[harmony.ChordSymbol] = None
            if sanitized_label:
                try:
                    cs_object = harmony.ChordSymbol(sanitized_label)
                    if not cs_object.pitches: cs_object = None
                except Exception as e_parse_guitar:
                    logger.warning(f"GuitarGen: Error parsing chord '{chord_label_str}' (sanitized: '{sanitized_label}') for block {blk_idx+1}: {e_parse_guitar}")
                    cs_object = None

            if cs_object is None:
                logger.warning(f"GuitarGen: Could not create valid ChordSymbol for '{chord_label_str}' (Sanitized: '{sanitized_label}') in block {blk_idx+1}. Skipping notes for this block.")
                continue


            rhythm_key = guitar_params.get("guitar_rhythm_key", "guitar_default_quarters")
            rhythm_details = self.rhythm_library.get(rhythm_key, self.rhythm_library.get("guitar_default_quarters"))
            if not rhythm_details or "pattern" not in rhythm_details:
                logger.warning(f"GuitarGen: Rhythm key '{rhythm_key}' not found or invalid pattern for block {blk_idx+1}. Using default.")
                rhythm_details = self.rhythm_library.get("guitar_default_quarters") # 再度フォールバック
                if not rhythm_details or "pattern" not in rhythm_details: # それでもない場合
                    logger.error(f"GuitarGen: Default rhythm 'guitar_default_quarters' also missing or invalid. Skipping rhythm for block {blk_idx+1}.")
                    continue
            pattern_events = rhythm_details.get("pattern", [])


            for event_def in pattern_events:
                event_offset_in_pattern = float(event_def.get("offset", 0.0))
                event_duration_in_pattern = float(event_def.get("duration", 1.0))
                event_velocity_factor = float(event_def.get("velocity_factor", 1.0))
                abs_event_start_offset = block_offset_ql + event_offset_in_pattern
                max_possible_event_dur = block_duration_ql - event_offset_in_pattern
                actual_event_dur = min(event_duration_in_pattern, max_possible_event_dur)
                if actual_event_dur < MIN_NOTE_DURATION_QL / 2: continue
                event_base_velocity = int(guitar_params.get("guitar_velocity", 70) * event_velocity_factor)

                generated_elements = self._create_notes_from_event(
                    cs_object, guitar_params, abs_event_start_offset, actual_event_dur, event_base_velocity
                )
                all_generated_elements_for_part.extend(generated_elements)

        global_guitar_params = processed_chord_stream[0].get("part_params", {}).get("guitar", {}) if processed_chord_stream else {}
        if global_guitar_params.get("guitar_humanize", global_guitar_params.get("humanize", False)):
            h_template = global_guitar_params.get("guitar_humanize_style_template",
                                                 global_guitar_params.get("humanize_style_template", "default_guitar_subtle"))
            h_custom = {
                k.replace("default_guitar_humanize_", "").replace("guitar_humanize_", "").replace("humanize_", ""): v
                for k, v in global_guitar_params.items()
                if (k.startswith("guitar_humanize_") or k.startswith("default_guitar_humanize_") or k.startswith("humanize_"))
                   and not k.endswith("_template") and not k.endswith("humanize") and not k.endswith("_opt")
            }
            logger.info(f"GuitarGen: Humanizing guitar part (template: {h_template}, custom: {h_custom})")

            temp_part_for_humanize = stream.Part()
            for el_item_guitar in all_generated_elements_for_part:
                temp_part_for_humanize.insert(el_item_guitar.offset, el_item_guitar)

            guitar_part = apply_humanization_to_part(temp_part_for_humanize, template_name=h_template, custom_params=h_custom)
            guitar_part.id = "Guitar" # ID再設定
            if not guitar_part.getElementsByClass(m21instrument.Instrument).first():
                guitar_part.insert(0, self.default_instrument)
            if not guitar_part.getElementsByClass(tempo.MetronomeMark).first():
                guitar_part.insert(0, tempo.MetronomeMark(number=self.global_tempo))
            if not guitar_part.getElementsByClass(meter.TimeSignature).first():
                ts_copy_humanize = meter.TimeSignature(self.global_time_signature_obj.ratioString)
                guitar_part.insert(0, ts_copy_humanize)
        else:
            for el_item_guitar_final in all_generated_elements_for_part:
                guitar_part.insert(el_item_guitar_final.offset, el_item_guitar_final)

        logger.info(f"GuitarGen: Finished. Part has {len(list(guitar_part.flatten().notesAndRests))} elements.") # .flat -> .flatten()
        return guitar_part

# --- END OF FILE generator/guitar_generator.py ---
