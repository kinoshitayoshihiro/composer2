# --- START OF FILE generator/vocal_generator.py (同期特化・歌詞処理削除版 v6) ---
import music21
from typing import List, Dict, Optional, Any, Tuple, Union, cast

# music21 のサブモジュールを正しい形式でインポート
import music21.stream as stream
import music21.note as note
import music21.pitch as pitch
import music21.meter as meter
import music21.duration as duration
import music21.instrument as m21instrument
import music21.tempo as tempo

# import music21.key as key # このファイルでは直接使用していないためコメントアウト
# import music21.expressions as expressions # このファイルでは直接使用していないためコメントアウト
import music21.volume as m21volume
import music21.expressions as expressions

import music21.articulations as articulations
# import music21.dynamics as dynamics # このファイルでは直接使用していないためコメントアウト
from music21 import exceptions21
from utilities.vibrato_engine import generate_vibrato

import logging
import json
from pathlib import Path

# import re # 正規表現は歌詞処理で主に使用していたため不要に
import copy
import random

# import math # mathモジュールも現在のロジックでは不要に

# NumPy import attempt and flag
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logging.info(
        "VocalGen(Humanizer): NumPy found. Fractional noise generation is enabled."
    )
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    logging.warning(
        "VocalGen(Humanizer): NumPy not found. Fractional noise will use Gaussian fallback."
    )


logger = logging.getLogger(__name__)

PHONEME_DICT_PATH = Path(__file__).resolve().parents[1] / "utilities" / "phoneme_dict.json"
try:
    with PHONEME_DICT_PATH.open("r", encoding="utf-8") as f:
        PHONEME_DICT: Dict[str, str] = json.load(f)
except FileNotFoundError:
    logger.warning(f"Phoneme dictionary not found at {PHONEME_DICT_PATH}")
    PHONEME_DICT = {}


def text_to_phonemes(text: str) -> List[str]:
    """Convert Japanese characters to phoneme strings with multi-character support."""
    phonemes: List[str] = []
    if not text:
        return phonemes

    # Sort keys by length in descending order for greedy matching of longer keys
    keys = sorted(PHONEME_DICT.keys(), key=len, reverse=True)
    i = 0
    while i < len(text):
        matched = False
        for k in keys:
            if text.startswith(k, i):
                phonemes.append(PHONEME_DICT.get(k, k))
                i += len(k)
                matched = True
                break
        if not matched:
            phonemes.append(text[i])
            i += 1
    return phonemes


class PhonemeArticulation(articulations.Articulation):
    """Simple articulation storing a phoneme string."""

    def __init__(self, phoneme: str, **keywords):
        super().__init__(**keywords)
        self.phoneme = phoneme

    def _reprInternal(self):  # type: ignore[override]
        return self.phoneme

MIN_NOTE_DURATION_QL = 0.125
# DEFAULT_BREATH_DURATION_QL: float = 0.25 # 歌詞ベースのブレス挿入削除のため不要
# MIN_DURATION_FOR_BREATH_AFTER_NOTE_QL: float = 1.0 # 同上
# PUNCTUATION_FOR_BREATH: Tuple[str, ...] = ('、', '。', '！', '？', ',', '.', '!', '?') # 同上


try:
    from utilities.core_music_utils import get_time_signature_object
except ImportError:
    logger_fallback_util = logging.getLogger(__name__ + ".fallback_core_util")
    logger_fallback_util.warning(
        "VocalGen: Could not import get_time_signature_object from utilities. Using basic fallback."
    )

    def get_time_signature_object(ts_str: Optional[str]) -> meter.TimeSignature:
        if not ts_str:
            ts_str = "4/4"
            return meter.TimeSignature(ts_str)
        try:
            return meter.TimeSignature(ts_str)
        except:
            return meter.TimeSignature("4/4")


try:
    from utilities.humanizer import apply_humanization_to_element
except ImportError:
    logger_fallback_humanizer = logging.getLogger(
        __name__ + ".fallback_humanizer_vocal"
    )
    logger_fallback_humanizer.warning(
        "VocalGen: Could not import humanizer.apply_humanization_to_element. Humanization will be basic."
    )

    def apply_humanization_to_element(element, template_name=None, custom_params=None):
        return element


class VocalGenerator:
    def __init__(
        self,
        default_instrument=m21instrument.Vocalist(),
        global_tempo: int = 120,
        global_time_signature: str = "4/4",
    ):

        self.default_instrument = default_instrument
        self.global_tempo = global_tempo
        self.global_time_signature_str = global_time_signature
        try:
            self.global_time_signature_obj = get_time_signature_object(
                global_time_signature
            )
        except NameError:
            logger.error(
                "VocalGen init: get_time_signature_object not available. Using music21.meter directly."
            )
            self.global_time_signature_obj = meter.TimeSignature(global_time_signature)
        except Exception as e_ts_init:
            logger.error(
                f"VocalGen init: Error initializing time signature from '{global_time_signature}': {e_ts_init}. Defaulting to 4/4.",
                exc_info=True,
            )
            self.global_time_signature_obj = meter.TimeSignature("4/4")

    def _parse_midivocal_data(self, midivocal_data: List[Dict]) -> List[Dict]:
        parsed_notes = []
        for item_idx, item in enumerate(midivocal_data):
            try:
                offset = float(item.get("offset", item.get("Offset", 0.0)))
                pitch_name = str(item.get("pitch", item.get("Pitch", "")))
                length = float(item.get("length", item.get("Length", 0.0)))
                velocity = int(item.get("velocity", item.get("Velocity", 70)))

                if not pitch_name:
                    logger.warning(f"Vocal note #{item_idx+1} empty pitch. Skip.")
                    continue
                try:
                    pitch.Pitch(pitch_name)
                except Exception as e_p:
                    logger.warning(
                        f"Skip vocal #{item_idx+1} invalid pitch: '{pitch_name}' ({e_p})"
                    )
                    continue
                if length <= 0:
                    logger.warning(
                        f"Skip vocal #{item_idx+1} non-positive length: {length}"
                    )
                    continue

                parsed_notes.append(
                    {
                        "offset": offset,
                        "pitch_str": pitch_name,
                        "q_length": length,
                        "velocity": velocity,
                    }
                )
            except KeyError as ke:
                logger.error(
                    f"Skip vocal item #{item_idx+1} missing key: {ke} in {item}"
                )
            except ValueError as ve:
                logger.error(
                    f"Skip vocal item #{item_idx+1} ValueError: {ve} in {item}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error parsing vocal item #{item_idx+1}: {e} in {item}",
                    exc_info=True,
                )

        parsed_notes.sort(key=lambda x: x["offset"])
        logger.info(f"Parsed {len(parsed_notes)} valid notes from midivocal_data.")
        return parsed_notes

    def _split_into_syllables(self, lyric_words: List[str]) -> List[str]:
        """Naively split words into syllables. Japanese characters are treated as syllables."""
        syllables: List[str] = []
        for word in lyric_words:
            syllables.extend(list(word))
        return syllables

    def _assign_syllables_to_part(self, part: stream.Stream, syllables: List[str]) -> None:
        """Assign syllables sequentially to notes of the part via ``note.lyric``."""
        notes = list(part.flatten().notes)
        for n, syl in zip(notes, syllables):
            n.lyric = syl

    def _assign_phonemes_to_part(self, part: stream.Stream, phonemes: List[str]) -> None:
        """Attach phoneme articulations sequentially to notes of the part."""
        notes = list(part.flatten().notes)
        for n, ph in zip(notes, phonemes):
            n.articulations.append(PhonemeArticulation(ph))

    def _apply_vibrato_to_part(self, part: stream.Stream, phonemes: List[str]) -> None:
        """Embed vibrato events into each note's expressions based on phoneme."""
        notes = list(part.flatten().notes)
        for n, ph in zip(notes, phonemes):
            if n.quarterLength < 0.5:
                continue
            depth = 0.5 if ph and ph[0].lower() in "aeiou" else 0.25
            events = generate_vibrato(n.quarterLength, depth, 5.0)
            n.expressions.append(expressions.TextExpression("vibrato"))
            n.editorial.vibrato_events = events

    def _get_section_for_note_offset(
        self, note_offset: float, processed_stream: List[Dict]
    ) -> Optional[str]:
        """
        指定された音符オフセットがどのセクションに属するかを返します。
        processed_stream は modular_composer.py で生成されるブロック情報のリストです。
        """
        for block in processed_stream:
            block_start = block.get("offset", 0.0)
            block_end = block_start + block.get("q_length", 0.0)
            # 厳密な比較 (< block_end) を行う
            if block_start <= note_offset < block_end:
                return block.get("section_name")
        logger.debug(
            f"VocalGen: No section found in processed_stream for note offset {note_offset:.2f}"
        )  # ログレベルをdebugに変更
        return None

    def compose(
        self,
        midivocal_data: List[Dict],
        # kasi_rist_data: Dict[str, List[str]], # 歌詞データは不要に
        processed_chord_stream: List[Dict],  # 将来的な拡張のため、引数としては残す
        # insert_breaths_opt: bool = True, # ブレス挿入オプションは削除
        # breath_duration_ql_opt: float = DEFAULT_BREATH_DURATION_QL, # 同上
        humanize_opt: bool = True,
        humanize_template_name: Optional[str] = "vocal_ballad_smooth",
        humanize_custom_params: Optional[Dict[str, Any]] = None,
        lyrics_words: Optional[List[str]] = None,
    ) -> stream.Part:

        vocal_part = stream.Part(id="Vocal")
        vocal_part.insert(0, self.default_instrument)
        vocal_part.append(tempo.MetronomeMark(number=self.global_tempo))

        if self.global_time_signature_obj:
            ts_copy = meter.TimeSignature(self.global_time_signature_obj.ratioString)
            vocal_part.append(ts_copy)
        else:
            logger.warning(
                "VocalGen compose: global_time_signature_obj is None. Defaulting to 4/4."
            )
            vocal_part.append(meter.TimeSignature("4/4"))

        parsed_vocal_notes_data = self._parse_midivocal_data(midivocal_data)
        if not parsed_vocal_notes_data:
            logger.warning(
                "VocalGen: No valid notes parsed from midivocal_data. Returning empty part."
            )
            return vocal_part

        final_elements: List[Union[note.Note, note.Rest]] = (
            []
        )  # note.Restも型ヒントに残すが、現状はNoteのみ

        for note_data_item in parsed_vocal_notes_data:
            note_offset = note_data_item["offset"]
            note_pitch_str = note_data_item["pitch_str"]
            note_q_length = note_data_item["q_length"]
            note_velocity = note_data_item.get("velocity", 70)

            # section_for_this_note = self._get_section_for_note_offset(note_offset, processed_chord_stream) # 歌詞割り当てには不要

            try:
                m21_n_obj = note.Note(note_pitch_str, quarterLength=note_q_length)
                m21_n_obj.volume = m21volume.Volume(velocity=note_velocity)
                m21_n_obj.offset = note_offset  # オフセットを設定
                final_elements.append(m21_n_obj)  # 直接 final_elements に追加
            except Exception as e:
                logger.error(
                    f"VocalGen: Failed to create Note for {note_pitch_str} at {note_offset}: {e}"
                )
                continue

        if humanize_opt:
            temp_humanized_elements = []
            try:
                for (
                    el_item
                ) in (
                    final_elements
                ):  # この時点ではfinal_elementsはNoteオブジェクトのみのはず
                    if isinstance(el_item, note.Note):
                        humanized_el = apply_humanization_to_element(
                            el_item,
                            template_name=humanize_template_name,
                            custom_params=humanize_custom_params,
                        )
                        temp_humanized_elements.append(humanized_el)
                    # else: # Rest の場合はそのまま追加するが、現状はRestはfinal_elementsに入らない
                    #     temp_humanized_elements.append(el_item)
                final_elements = temp_humanized_elements
            except (
                NameError
            ):  # apply_humanization_to_element がインポート失敗した場合のフォールバック
                logger.warning(
                    "VocalGen: apply_humanization_to_element not available, skipping humanization for vocal notes."
                )
            except Exception as e_hum:
                logger.error(
                    f"VocalGen: Error during vocal note humanization: {e_hum}",
                    exc_info=True,
                )

        for el_item_final in final_elements:
            vocal_part.insert(el_item_final.offset, el_item_final)

        if lyrics_words:
            syllables = self._split_into_syllables(lyrics_words)
            phonemes = text_to_phonemes("".join(syllables))
            self._assign_syllables_to_part(vocal_part, syllables)
            self._assign_phonemes_to_part(vocal_part, phonemes)
            self._apply_vibrato_to_part(vocal_part, phonemes)

        logger.info(
            f"VocalGen: Finished. Final part has {len(list(vocal_part.flatten().notesAndRests))} elements."
        )
        return vocal_part

    def extract_phonemes(self, part: stream.Part) -> List[str]:
        """Return phoneme articulations found on ``part``."""
        phonemes: List[str] = []
        for n in part.flatten().notes:
            for a in n.articulations:
                if isinstance(a, PhonemeArticulation):
                    phonemes.append(a.phoneme)
        return phonemes

    def synthesize_with_tts(self, midi_file: Path, phoneme_file: Path) -> bytes:
        """Load phonemes from ``phoneme_file`` and synthesize audio with a TTS model.

        Returns empty bytes if ``tts_model`` cannot be imported or if synthesis fails.
        """
        try:
            from tts_model import synthesize  # type: ignore
        except ImportError as e:
            logger.error("Failed to import tts_model: %s", e, exc_info=True)
            return b""

        try:
            with phoneme_file.open("r", encoding="utf-8") as f:
                phonemes = json.load(f)
            return synthesize(midi_file, phonemes)
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e, exc_info=True)
            return b""


from .base_part_generator import BasePartGenerator


class PianoGenerator(BasePartGenerator):
    def __init__(self, *args, main_cfg=None, **kwargs):
        self.main_cfg = main_cfg
        super().__init__(*args, **kwargs)
        # ...他の初期化処理...


# --- END OF FILE generator/vocal_generator.py ---
