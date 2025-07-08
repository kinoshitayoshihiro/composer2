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
from utilities.vibrato_engine import (
    generate_vibrato,
    generate_gliss,
    generate_trill,
)
from utilities.cc_tools import merge_cc_events

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


def synthesize_with_onnx(model_path: Path, midi: Path, phonemes: List[str]) -> bytes:
    """Synthesize *phonemes* using an ONNX TTS model.

    Always returns ``bytes`` and logs failures instead of raising.
    """

    logger.info("Starting ONNX synthesis using %s", model_path)
    try:
        import numpy as np
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover - optional
        logger.error("Failed to import onnxruntime: %s", exc, exc_info=True)
        return b""

    try:
        session = ort.InferenceSession(str(model_path))
        tokens = np.fromiter((ord(p[0]) for p in phonemes), dtype=np.float32)
        output = session.run(None, {"input": tokens})[0]
        if not isinstance(output, (bytes, bytearray)):
            output = bytes(output)
        logger.info("Finished ONNX synthesis")
        return bytes(output)
    except Exception as exc:  # pragma: no cover - runtime
        logger.error("ONNX synthesis failed: %s", exc, exc_info=True)
        return b""


def text_to_phonemes(
    text: str, phoneme_dict: Optional[Dict[str, str]] = None
) -> List[Tuple[str, str, float]]:
    """Convert Japanese characters to phoneme tuples.

    Accent alternates ``"H"`` and ``"L"`` for demonstration purposes. Duration
    defaults to ``MIN_NOTE_DURATION_QL``; ``compose`` updates this with each
    note's real length.
    """
    mapping = phoneme_dict or PHONEME_DICT
    phonemes: List[Tuple[str, str, float]] = []
    if not text:
        return phonemes

    # Sort keys by length in descending order for greedy matching of longer keys
    keys = sorted(mapping.keys(), key=len, reverse=True)
    i = 0
    toggle_high = True
    while i < len(text):
        matched = False
        for k in keys:
            if text.startswith(k, i):
                accent = "H" if toggle_high else "L"
                phonemes.append((mapping.get(k, k), accent, MIN_NOTE_DURATION_QL))
                toggle_high = not toggle_high
                i += len(k)
                matched = True
                break
        if not matched:
            accent = "H" if toggle_high else "L"
            phonemes.append((text[i], accent, MIN_NOTE_DURATION_QL))
            toggle_high = not toggle_high
            i += 1
    return phonemes


class PhonemeArticulation(articulations.Articulation):
    """Articulation storing phoneme, accent and note duration."""

    def __init__(self, phoneme: str, accent: str = "L", duration_qL: float = 0.0, **keywords):
        super().__init__(**keywords)
        self.phoneme = phoneme
        self.accent = accent
        self.duration_qL = duration_qL

    def _reprInternal(self):  # type: ignore[override]
        return f"{self.phoneme}:{self.accent}:{self.duration_qL}"

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
        phoneme_dict_path: Optional[Path] = None,
        *,
        vibrato_depth: float = 0.5,
        vibrato_rate: float = 5.0,
        enable_articulation: bool = True,
    ):

        self.default_instrument = default_instrument
        self.global_tempo = global_tempo
        self.global_time_signature_str = global_time_signature
        self.phoneme_dict_path = phoneme_dict_path or PHONEME_DICT_PATH
        self.vibrato_depth = float(vibrato_depth)
        self.vibrato_rate = float(vibrato_rate)
        self.enable_articulation = bool(enable_articulation)

        try:
            with self.phoneme_dict_path.open("r", encoding="utf-8") as fh:
                self.phoneme_dict = json.load(fh)
        except FileNotFoundError:
            logger.warning(f"Phoneme dictionary not found at {self.phoneme_dict_path}")
            self.phoneme_dict = PHONEME_DICT
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
        """Naively split words into syllables.

        Special markers like ``"[gliss]"`` or ``"[trill]"`` are kept intact.
        Japanese characters are treated as individual syllables.
        """
        syllables: List[str] = []
        for word in lyric_words:
            if word in {"[gliss]", "[trill]"}:
                syllables.append(word)
            else:
                syllables.extend(list(word))
        return syllables

    def _assign_syllables_to_part(self, part: stream.Stream, syllables: List[str]) -> None:
        """Assign syllables sequentially to notes of the part via ``note.lyric``."""
        notes = sorted(part.flatten().notes, key=lambda n: n.offset)
        if not syllables:
            logger.warning("VocalGen: lyrics_words is empty. Skipping lyric assignment.")
            return
        if len(syllables) != len(notes):
            logger.warning(
                "VocalGen: Number of syllables (%d) does not match number of notes (%d).",
                len(syllables),
                len(notes),
            )
        for n, syl in zip(notes, syllables):
            n.lyric = syl

    def _assign_phonemes_to_part(
        self, part: stream.Stream, phonemes: List[Tuple[str, str, float]]
    ) -> None:
        """Attach phoneme articulations sequentially to notes of the part."""
        notes = sorted(part.flatten().notes, key=lambda n: n.offset)
        for n, (ph, accent, _dur) in zip(notes, phonemes):
            n.articulations.append(
                PhonemeArticulation(ph, accent=accent, duration_qL=n.quarterLength)
            )

    def _apply_vibrato_to_part(
        self, part: stream.Stream, phonemes: List[Tuple[str, str, float]]
    ) -> None:
        """Embed vibrato events and store CC/pitch bend messages."""
        if not self.enable_articulation:
            return

        notes = sorted(part.flatten().notes, key=lambda n: n.offset)
        cc_events: list[tuple[float, int, int]] = []
        bends: list[tuple[float, int]] = []
        for n, (ph, _accent, _dur) in zip(notes, phonemes):
            if n.quarterLength < 0.5:
                continue
            depth = (
                self.vibrato_depth
                if ph and ph[0].lower() in "aeiou"
                else self.vibrato_depth * 0.5
            )
            events = generate_vibrato(
                n.quarterLength, depth, self.vibrato_rate
            )
            n.expressions.append(expressions.TextExpression("vibrato"))
            n.editorial.vibrato_events = events
            for kind, t, val in events:
                abs_t = float(n.offset) + t
                if kind == "aftertouch":
                    cc_events.append((abs_t, 74, val))
                else:
                    bends.append((abs_t, val))
        if cc_events:
            existing = [
                (e["time"], e["cc"], e["val"]) if isinstance(e, dict) else e
                for e in getattr(part, "extra_cc", [])
            ]
            part.extra_cc = merge_cc_events(set(existing), set(cc_events))
        if bends:
            data = getattr(part, "pitch_bends", [])
            if data and isinstance(data[0], dict):
                base = [(d["time"], d["pitch"]) for d in data]
            else:
                base = data
            merged: dict[float, int] = {float(t): int(v) for t, v in base}
            for t, v in bends:
                merged[float(t)] = int(v)
            part.pitch_bends = [
                {"time": t, "pitch": v} for t, v in sorted(merged.items())
            ]

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
        lyrics_words: List[str] | None = None,
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
            phonemes = text_to_phonemes("".join(syllables), self.phoneme_dict)
            self._assign_syllables_to_part(vocal_part, syllables)
            self._assign_phonemes_to_part(vocal_part, phonemes)
            self._apply_vibrato_to_part(vocal_part, phonemes)

            # articulation markers
            if self.enable_articulation:
                notes = sorted(vocal_part.flatten().notes, key=lambda n: n.offset)
                for idx, n in enumerate(notes):
                    lyr = n.lyric
                    if lyr == "[gliss]" and idx + 1 < len(notes):
                        nxt = notes[idx + 1]
                        events = generate_gliss(
                            int(n.pitch.midi), int(nxt.pitch.midi), n.quarterLength
                        )
                        bends = [
                            (float(n.offset) + t, (p - n.pitch.midi) * 64)
                            for p, t in events
                        ]
                        data = getattr(vocal_part, "pitch_bends", [])
                        for t, v in bends:
                            data.append({"time": t, "pitch": int(v)})
                        vocal_part.pitch_bends = sorted(data, key=lambda x: x["time"])
                    elif lyr == "[trill]":
                        events = generate_trill(int(n.pitch.midi), n.quarterLength)
                        data = getattr(vocal_part, "pitch_bends", [])
                        cc_events = getattr(vocal_part, "extra_cc", [])
                        for p, t, vel in events:
                            data.append({"time": float(n.offset) + t, "pitch": 0})
                            cc_events.append(
                                {"time": float(n.offset) + t, "cc": 74, "val": vel}
                            )
                        vocal_part.pitch_bends = sorted(data, key=lambda x: x["time"])
                        vocal_part.extra_cc = merge_cc_events(cc_events, [])
        else:
            logger.warning(
                "VocalGen compose: lyrics_words not provided. Skipping lyric assignment."
            )

        logger.info(
            f"VocalGen: Finished. Final part has {len(list(vocal_part.flatten().notesAndRests))} elements."
        )
        return vocal_part

    def extract_phonemes(self, part: stream.Part) -> List[Tuple[str, str, float]]:
        """Return phoneme tuples found on ``part``."""
        phonemes: List[Tuple[str, str, float]] = []
        for n in part.flatten().notes:
            for a in n.articulations:
                if isinstance(a, PhonemeArticulation):
                    phonemes.append((a.phoneme, a.accent, a.duration_qL))
        return phonemes

    def synthesize_with_tts(
        self, midi_file: Path, phoneme_file: Path, onnx_model: Optional[Path] = None
    ) -> bytes:
        """Load phonemes from ``phoneme_file`` and synthesize audio.

        If ``onnx_model`` is provided, :func:`synthesize_with_onnx` is used.
        Returns empty bytes on failure.
        """
        logger.info("Starting TTS synthesis for %s", midi_file)
        try:
            with phoneme_file.open("r", encoding="utf-8") as f:
                phonemes = json.load(f)
            if onnx_model:
                audio = synthesize_with_onnx(onnx_model, midi_file, phonemes)
            else:
                from tts_model import synthesize  # type: ignore

                audio = synthesize(midi_file, phonemes)
            logger.info("Finished TTS synthesis")
            return audio
        except ImportError as e:
            logger.error("Failed to import tts_model: %s", e, exc_info=True)
            return b""
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
