# --- START OF FILE generators/chord_voicer.py (修正版) ---
from __future__ import annotations
from typing import List, Dict, Optional

# music21 のサブモジュールを正しい形式でインポート
import music21.stream as stream
import music21.note as note
import music21.harmony as harmony
import music21.pitch as pitch
import music21.meter as meter
import music21.instrument as m21instrument
import music21.tempo as tempo
import music21.chord as m21chord
import music21.volume as m21volume
from .base_part_generator import BasePartGenerator
from music21 import articulations  # 明示的にインポート
import re  # 正規表現を使用する場合
import logging

logger = logging.getLogger(__name__)

# --- core_music_utils からのインポート試行 ---
try:
    from utilities.core_music_utils import (
        get_time_signature_object,
    )

    logger.info(
        "ChordVoicer: Successfully imported get_time_signature_object from utilities.core_music_utils."
    )
except ImportError:
    logger.warning(
        "ChordVoicer: Could not import get_time_signature_object from utilities.core_music_utils. Using basic fallback."
    )

    def get_time_signature_object(ts_str: Optional[str]) -> meter.TimeSignature:
        if not ts_str:
            ts_str = "4/4"
        try:
            return meter.TimeSignature(ts_str)
        except:  # noqa E722
            return meter.TimeSignature("4/4")


# ChordVoicer内で sanitize_chord_label を定義 (Haruさん提供版)
def sanitize_chord_label(label: Optional[str]) -> Optional[str]:
    """
    入力されたコードラベルをmusic21が解釈しやすい形式に正規化する。
    Restや無効なラベルの場合は "Rest" を返す。
    """
    if label is None:
        return "Rest"

    s = str(label).strip()

    if s.upper() in ["NC", "N.C.", "NOCHORD", "NO CHORD", "SILENCE", "-", "REST", "R"]:
        return "Rest"
    if not s:
        return "Rest"

    s = (
        s.replace("Bb", "B-")
        .replace("Eb", "E-")
        .replace("Ab", "A-")
        .replace("Db", "D-")
        .replace("Gb", "G-")
    )
    s = re.sub(r"[△Δ]", "maj7", s)
    s = s.replace("Ma7", "maj7").replace("Maj7", "maj7").replace("MA7", "maj7")
    s = s.replace("M7", "maj7")
    s = s.replace("min7", "m7").replace("-7", "m7")
    s = s.replace("dim7", "dim").replace("°7", "dim")
    s = s.replace("ø", "m7b5").replace("Φ", "m7b5").replace("Ø", "m7b5")
    s = s.replace("dim", "dim")
    s = s.replace("aug", "+")
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"add\s*9", "add9", s, flags=re.IGNORECASE)
    s = re.sub(r"sus\s*2", "sus2", s, flags=re.IGNORECASE)

    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            chord_part = parts[0]
            bass_part = parts[1]
            bass_part = (
                bass_part.replace("Bb", "B-")
                .replace("Eb", "E-")
                .replace("Ab", "A-")
                .replace("Db", "D-")
                .replace("Gb", "G-")
            )
            s = f"{chord_part}/{bass_part}"

    s = s.replace("majmaj7", "maj7").replace("majmaj", "maj")
    s = " ".join(s.split())

    if not s:
        return "Rest"

    logger.debug(f"Sanitized chord label: '{label}' -> '{s}'")
    return s


def parse_chord_symbol(symbol: str, base_octave: int = 4) -> List[pitch.Pitch]:
    """Parse a chord symbol string into a list of ``music21.pitch.Pitch``.

    This lightweight parser recognises basic triad qualities and a few
    extensions such as ``add9``, ``sus2``, ``m7`` and ``6``.  It is meant as
    a minimal fallback when ``music21`` cannot correctly interpret the chord
    symbol on its own.

    Parameters
    ----------
    symbol:
        Chord symbol such as ``"Cadd9"`` or ``"Am7"``.
    base_octave:
        Octave for the root note.  Other tones are built relative to this.

    Returns
    -------
    List[pitch.Pitch]
        Pitches describing the chord.  Empty list if the symbol could not be
        parsed.
    """

    sanitized = sanitize_chord_label(symbol)
    if sanitized is None or sanitized.lower() == "rest":
        return []

    m_root = re.match(r"^([A-G](?:[#-])?)(.*)$", sanitized)
    if not m_root:
        return []

    root_str, remainder = m_root.group(1), m_root.group(2)

    base_intervals = [0, 4, 7]  # major triad by default
    extra_intervals: List[int] = []

    # Quality detection
    if remainder.startswith("m7b5"):
        base_intervals = [0, 3, 6]
        extra_intervals.append(10)
        remainder = remainder[4:]
    elif remainder.startswith("maj7"):
        extra_intervals.append(11)
        remainder = remainder[4:]
    elif remainder.startswith("m7"):
        base_intervals = [0, 3, 7]
        extra_intervals.append(10)
        remainder = remainder[2:]
    elif remainder.startswith("dim"):
        base_intervals = [0, 3, 6]
        remainder = remainder[3:]
    elif remainder.startswith("aug") or remainder.startswith("+"):
        base_intervals = [0, 4, 8]
        remainder = remainder[3:] if remainder.startswith("aug") else remainder[1:]
    elif remainder.startswith("sus2"):
        base_intervals = [0, 2, 7]
        remainder = remainder[4:]
    elif remainder.startswith("sus4") or remainder.startswith("sus"):
        base_intervals = [0, 5, 7]
        remainder = remainder[4:] if remainder.startswith("sus4") else remainder[3:]
    elif remainder.startswith("m"):
        base_intervals = [0, 3, 7]
        remainder = remainder[1:]

    # Tensions
    if "add9" in remainder:
        extra_intervals.append(14)
        remainder = remainder.replace("add9", "")
    if "6" in remainder:
        extra_intervals.append(9)
        remainder = remainder.replace("6", "")
    if "7" in remainder:
        extra_intervals.append(10)
        remainder = remainder.replace("7", "")

    root_pitch = pitch.Pitch(root_str)
    root_pitch.octave = base_octave
    intervals = base_intervals + sorted(extra_intervals)
    return [root_pitch.transpose(i) for i in intervals]


DEFAULT_CHORD_TARGET_OCTAVE_BOTTOM: int = 3
VOICING_STYLE_CLOSED = "closed"
VOICING_STYLE_OPEN = "open"
VOICING_STYLE_SEMI_CLOSED = "semi_closed"
VOICING_STYLE_DROP2 = "drop2"
VOICING_STYLE_DROP3 = "drop3"
VOICING_STYLE_DROP24 = "drop2and4"
VOICING_STYLE_FOUR_WAY_CLOSE = "four_way_close"
DEFAULT_VOICING_STYLE = VOICING_STYLE_CLOSED


class ChordVoicer:
    def __init__(
        self,
        default_instrument=m21instrument.KeyboardInstrument(),
        global_tempo: int = 120,
        global_time_signature: str = "4/4",
    ):
        self.default_instrument = default_instrument
        self.global_tempo = global_tempo
        try:
            self.global_time_signature_obj = get_time_signature_object(
                global_time_signature
            )
            if self.global_time_signature_obj is None:
                logger.warning(
                    "ChordVoicer __init__: get_time_signature_object returned None. Defaulting to 4/4."
                )
                self.global_time_signature_obj = meter.TimeSignature("4/4")
        except Exception as e_ts_init:
            logger.error(
                f"ChordVoicer __init__: Error initializing time signature from '{global_time_signature}': {e_ts_init}. Defaulting to 4/4.",
                exc_info=True,
            )
            self.global_time_signature_obj = meter.TimeSignature("4/4")

    def _apply_voicing_style(
        self,
        cs_obj: harmony.ChordSymbol,
        style_name: str,
        target_octave_for_bottom_note: int = DEFAULT_CHORD_TARGET_OCTAVE_BOTTOM,
        num_voices_target: Optional[int] = None,
    ) -> List[pitch.Pitch]:
        if not cs_obj.pitches:
            # 指定ベース音だけでも処理する余地はあるが、一旦は空を返す
            # if cs_obj.bass():
            #     bass_pitch = pitch.Pitch(cs_obj.bass().name)
            #     bass_pitch.octave = target_octave_for_bottom_note
            #     return [bass_pitch]
            logger.debug(
                f"CV._apply_style: ChordSymbol '{cs_obj.figure}' has no pitches (and no explicit bass handling here). Returning empty list."
            )
            return []
        try:
            temp_chord_for_closed = m21chord.Chord(cs_obj.pitches)
            original_closed_pitches = sorted(
                list(temp_chord_for_closed.closedPosition(inPlace=False).pitches),
                key=lambda p: p.ps,
            )
        except Exception as e_closed:
            logger.warning(
                f"CV._apply_style: Could not get closed position for '{cs_obj.figure}': {e_closed}. Using raw pitches."
            )
            original_closed_pitches = sorted(list(cs_obj.pitches), key=lambda p: p.ps)

        if not original_closed_pitches:
            return []

        current_pitches_for_voicing = list(original_closed_pitches)
        voiced_pitches_list: List[pitch.Pitch] = []

        try:
            m21_chord_obj = m21chord.Chord(current_pitches_for_voicing)

            if style_name == VOICING_STYLE_OPEN:
                voiced_pitches_list = list(
                    m21_chord_obj.openPosition(inPlace=False).pitches
                )
            elif style_name == VOICING_STYLE_DROP2:
                if len(m21_chord_obj.pitches) >= 2:
                    temp_pitches = list(m21_chord_obj.pitches)
                    if len(temp_pitches) >= 2:  # 念のため再確認
                        idx_to_drop = len(temp_pitches) - 2
                        note_to_drop = temp_pitches.pop(idx_to_drop)
                        note_to_drop_oct_down = note_to_drop.transpose(-12)
                        voiced_pitches_list = sorted(
                            temp_pitches + [note_to_drop_oct_down], key=lambda p: p.ps
                        )
                    else:
                        voiced_pitches_list = list(m21_chord_obj.pitches)
                else:
                    voiced_pitches_list = list(m21_chord_obj.pitches)
            elif style_name == VOICING_STYLE_DROP3:
                if len(m21_chord_obj.pitches) >= 3:
                    temp_pitches = list(m21_chord_obj.pitches)
                    idx_to_drop = len(temp_pitches) - 3
                    note_to_drop = temp_pitches.pop(idx_to_drop)
                    note_to_drop_oct_down = note_to_drop.transpose(-12)
                    voiced_pitches_list = sorted(
                        temp_pitches + [note_to_drop_oct_down], key=lambda p: p.ps
                    )
                else:
                    voiced_pitches_list = list(m21_chord_obj.pitches)
            elif style_name == VOICING_STYLE_DROP24:
                # music21 には Drop2and4 の直接的なメソッドはないため、手動実装
                # (上から2番目と4番目を下げる。4声以上必要)
                if len(m21_chord_obj.pitches) >= 4:
                    temp_pitches = sorted(
                        list(m21_chord_obj.pitches), key=lambda p: p.ps, reverse=True
                    )  # 高い順にソート
                    note_to_drop_2nd = temp_pitches[1].transpose(-12)
                    note_to_drop_4th = temp_pitches[3].transpose(-12)
                    # 元のリストから2番目と4番目を削除し、オクターブ下げたものを追加
                    remaining_pitches = [
                        p for i, p in enumerate(temp_pitches) if i not in [1, 3]
                    ]
                    voiced_pitches_list = sorted(
                        remaining_pitches + [note_to_drop_2nd, note_to_drop_4th],
                        key=lambda p: p.ps,
                    )
                else:
                    voiced_pitches_list = list(m21_chord_obj.pitches)

            elif style_name == VOICING_STYLE_FOUR_WAY_CLOSE:
                if len(m21_chord_obj.pitches) == 4:
                    # music21のfourWayCloseはChordSymbolに直接適用できないのでChordオブジェクト経由
                    chord_for_4way = m21chord.Chord(
                        cs_obj.pitches
                    )  # 元のChordSymbolのピッチで
                    if len(chord_for_4way.pitches) == 4:  # 再確認
                        chord_for_4way.fourWayClose(inPlace=True)
                        voiced_pitches_list = list(chord_for_4way.pitches)
                    else:
                        voiced_pitches_list = list(original_closed_pitches)
                else:
                    logger.debug(
                        f"CV: fourWayClose expects 4 voices, got {len(m21_chord_obj.pitches)} for {cs_obj.figure}. Using closed position."
                    )
                    voiced_pitches_list = list(original_closed_pitches)
            elif style_name == VOICING_STYLE_SEMI_CLOSED:
                if len(original_closed_pitches) >= 1:
                    # ベース音(指定があればそれ、なければルート)を target_octave に配置
                    # 残りの音をその上でクローズに
                    bass_or_root_for_bottom = (
                        cs_obj.bass() if cs_obj.bass() else cs_obj.root()
                    )
                    if bass_or_root_for_bottom:
                        bottom_pitch_obj = pitch.Pitch(bass_or_root_for_bottom.name)
                        bottom_pitch_obj.octave = target_octave_for_bottom_note

                        upper_structure_source_pitches = [
                            p
                            for p in original_closed_pitches
                            if p.name != bottom_pitch_obj.name
                        ]
                        # もしベース音がルートと異なり、かつルート音が上部構造に含まれていなかったら追加を試みる
                        if (
                            cs_obj.bass()
                            and cs_obj.root()
                            and cs_obj.bass().name != cs_obj.root().name
                        ):
                            if not any(
                                p.name == cs_obj.root().name
                                for p in upper_structure_source_pitches
                            ):
                                # original_closed_pitches から探す
                                original_root = next(
                                    (
                                        p
                                        for p in original_closed_pitches
                                        if p.name == cs_obj.root().name
                                    ),
                                    None,
                                )
                                if original_root:
                                    upper_structure_source_pitches.append(original_root)

                        if (
                            not upper_structure_source_pitches
                            and len(original_closed_pitches) > 1
                        ):
                            # ベース音以外が全て同じ音名だった場合(例: C/C で C のみ)、
                            # original_closed_pitches からベース音以外の音を選ぶ
                            upper_structure_source_pitches = [
                                p
                                for p in original_closed_pitches
                                if p.ps != bottom_pitch_obj.ps
                            ]

                        if upper_structure_source_pitches:
                            temp_upper_chord = m21chord.Chord(
                                upper_structure_source_pitches
                            )
                            closed_upper_pitches = sorted(
                                list(
                                    temp_upper_chord.closedPosition(
                                        inPlace=False
                                    ).pitches
                                ),
                                key=lambda p: p.ps,
                            )
                            if closed_upper_pitches:
                                lowest_upper = min(
                                    closed_upper_pitches, key=lambda p: p.ps
                                )
                                while lowest_upper.ps <= bottom_pitch_obj.ps + 1:
                                    lowest_upper.transpose(12, inPlace=True)
                                shift_amount = (
                                    lowest_upper.ps
                                    - min(closed_upper_pitches, key=lambda p: p.ps).ps
                                )  # 最初のlowest_upperからの実際のシフト量

                                final_upper_voices = [
                                    p.transpose(shift_amount)
                                    for p in closed_upper_pitches
                                ]
                                voiced_pitches_list = sorted(
                                    [bottom_pitch_obj] + final_upper_voices,
                                    key=lambda p: p.ps,
                                )
                            else:
                                voiced_pitches_list = [bottom_pitch_obj]
                        else:
                            voiced_pitches_list = [bottom_pitch_obj]
                    else:  # ベースもルートもない（通常ありえない）
                        voiced_pitches_list = list(original_closed_pitches)
                else:
                    voiced_pitches_list = list(original_closed_pitches)
            else:  # VOICING_STYLE_CLOSED or unknown
                if style_name != VOICING_STYLE_CLOSED:
                    logger.debug(
                        f"CV: Unknown voicing style '{style_name}'. Defaulting to closed for '{cs_obj.figure}'."
                    )
                voiced_pitches_list = list(original_closed_pitches)
        except Exception as e_style_app:
            logger.error(
                f"CV._apply_style: Error applying voicing style '{style_name}' to '{cs_obj.figure}': {e_style_app}. Defaulting to closed.",
                exc_info=True,
            )
            voiced_pitches_list = list(original_closed_pitches)

        if not voiced_pitches_list:  # スタイル適用で空になった場合
            voiced_pitches_list = list(original_closed_pitches)  # フォールバック

        # 声部数ターゲットの適用
        if num_voices_target is not None and voiced_pitches_list:
            if len(voiced_pitches_list) > num_voices_target:
                temp_final_pitches = []
                root_of_chord = cs_obj.root()
                # 指定ベース音があればそれを優先的に含める
                bass_of_chord = cs_obj.bass()
                priority_note_name = (
                    bass_of_chord.name
                    if bass_of_chord
                    else (root_of_chord.name if root_of_chord else None)
                )

                if priority_note_name:
                    priority_in_voiced = next(
                        (
                            p
                            for p in voiced_pitches_list
                            if p.name == priority_note_name
                        ),
                        None,
                    )
                    if priority_in_voiced:
                        temp_final_pitches.append(priority_in_voiced)
                    # もしボイシング結果に優先音が含まれていなかったら、元のコード構成音から探して追加
                    elif len(temp_final_pitches) < num_voices_target:
                        original_priority_pitch = next(
                            (
                                p
                                for p in original_closed_pitches
                                if p.name == priority_note_name
                            ),
                            None,
                        )
                        if original_priority_pitch:
                            temp_final_pitches.append(original_priority_pitch)

                remaining_pitches_for_num_target = sorted(
                    [p for p in voiced_pitches_list if p not in temp_final_pitches],
                    key=lambda x: x.ps,
                    reverse=True,  # 高い音から取るか、低い音から取るか、あるいは音楽的役割で？一旦高い方から
                )
                while (
                    len(temp_final_pitches) < num_voices_target
                    and remaining_pitches_for_num_target
                ):
                    temp_final_pitches.append(remaining_pitches_for_num_target.pop(0))
                voiced_pitches_list = sorted(temp_final_pitches, key=lambda p: p.ps)
                logger.debug(
                    f"CV: Reduced voices to {len(voiced_pitches_list)} for {cs_obj.figure} (target: {num_voices_target})."
                )

            elif len(voiced_pitches_list) < num_voices_target:
                can_add_voices = num_voices_target - len(voiced_pitches_list)
                pitches_to_add_from = list(original_closed_pitches)  # 元の構成音から
                added_count = 0
                # オクターブ重複を避けるため、既存の音名を記録
                existing_names_in_voiced = {p.name for p in voiced_pitches_list}

                for p_orig in sorted(
                    pitches_to_add_from, key=lambda p: p.ps
                ):  # 低い音から試す
                    if added_count >= can_add_voices:
                        break
                    # 追加候補の音と同じ音名が既にボイシングになければオクターブ上を追加
                    if p_orig.name not in existing_names_in_voiced:
                        p_oct_up = p_orig.transpose(12)
                        # さらにオクターブを調整して、既存の音域と近くなるように
                        if voiced_pitches_list:
                            avg_ps = sum(p.ps for p in voiced_pitches_list) / len(
                                voiced_pitches_list
                            )
                            while p_oct_up.ps < avg_ps - 18:  # 平均より低すぎたら上げる
                                p_oct_up.transpose(12, inPlace=True)
                            while p_oct_up.ps > avg_ps + 18:  # 平均より高すぎたら下げる
                                p_oct_up.transpose(-12, inPlace=True)

                        voiced_pitches_list.append(p_oct_up)
                        existing_names_in_voiced.add(p_oct_up.name)
                        added_count += 1
                voiced_pitches_list = sorted(voiced_pitches_list, key=lambda p: p.ps)
                logger.debug(
                    f"CV: Attempted to add voices for {cs_obj.figure}. Result: {len(voiced_pitches_list)} (target: {num_voices_target})."
                )

        # --- 最終的なオクターブ調整 (指定ベース音を最優先) ---
        if voiced_pitches_list:
            # cs_obj.bass() が存在し、それがルート音と異なる場合、そのベース音を最優先で配置
            has_specified_bass = cs_obj.bass() is not None
            is_bass_different_from_root = False
            if has_specified_bass and cs_obj.root():
                if (
                    cs_obj.bass().nameWithOctave != cs_obj.root().nameWithOctave
                ):  # オクターブも含めて比較
                    is_bass_different_from_root = (
                        cs_obj.bass().name != cs_obj.root().name
                    )

            # ケース1: 指定ベース音があり、それがルート音と異なる音名の場合
            if has_specified_bass and is_bass_different_from_root:
                specified_bass_pitch_name = cs_obj.bass().name
                authoritative_bass_pitch = pitch.Pitch(specified_bass_pitch_name)
                authoritative_bass_pitch.octave = target_octave_for_bottom_note
                logger.debug(
                    f"  CV: Prioritizing specified bass {authoritative_bass_pitch.nameWithOctave} for chord {cs_obj.figure}"
                )

                # ボイシングされたピッチから指定ベース音(音名で比較)を除いたものを上部構造とする
                upper_structure_pitches = [
                    p
                    for p in voiced_pitches_list
                    if p.name != specified_bass_pitch_name
                ]

                # もし元のvoiced_pitches_listに指定ベース音の音名が含まれていなかった場合
                # (例: C/Bb で voiced_pitches_list が C,E,G のみだった場合)、
                # upper_structure_pitches は voiced_pitches_list と同じになる。
                # この場合、上部構造は cs_obj.pitches (C,E,G) から構成されるべき。
                if not any(
                    p.name == specified_bass_pitch_name for p in voiced_pitches_list
                ):
                    logger.debug(
                        f"  CV: Specified bass name {specified_bass_pitch_name} was not found in initially voiced list for {cs_obj.figure}. Upper structure based on remaining notes."
                    )
                    # このケースでは、cs_obj.pitches (コード本来の構成音) を上部構造の基礎とする方が適切かもしれない
                    # ただし、前のボイシング処理で意図的に音が省かれている可能性もあるため、ここでは voiced_pitches_list を尊重

                upper_structure_pitches = sorted(
                    upper_structure_pitches, key=lambda p: p.ps
                )

                if upper_structure_pitches:
                    # 上部構造の最低音が、確定ベース音より必ず高くなるようにオクターブ調整
                    lowest_upper_pitch = min(
                        upper_structure_pitches, key=lambda p: p.ps
                    )
                    # 厳密に「より高い」ので、半音でも高い状態 (ps > authoritative_bass_pitch.ps) を目指す
                    # 同時に、離れすぎないようにも調整する（例：ベースのオクターブ+1以内など）
                    # ここでは単純にベースより高ければOKとする
                    transposed_upper_structure = []

                    # まず、最低音がベースより低ければ、最低音がベースのすぐ上に来るように全体を上げる
                    shift_to_get_above_bass = 0
                    temp_lowest_upper_for_calc = pitch.Pitch(
                        lowest_upper_pitch.nameWithOctave
                    )  # copy
                    while temp_lowest_upper_for_calc.ps <= authoritative_bass_pitch.ps:
                        temp_lowest_upper_for_calc.transpose(12, inPlace=True)
                        shift_to_get_above_bass += 12

                    if shift_to_get_above_bass > 0:
                        transposed_upper_structure = [
                            p.transpose(shift_to_get_above_bass)
                            for p in upper_structure_pitches
                        ]
                        logger.debug(
                            f"  CV: Transposed upper structure for {cs_obj.figure} by {shift_to_get_above_bass} semitones to be above bass {authoritative_bass_pitch.nameWithOctave}"
                        )
                    else:
                        transposed_upper_structure = list(
                            upper_structure_pitches
                        )  # コピーを作成

                    # それでもまだ最低音がベース音以下なら（ありえないはずだが）、さらに調整
                    if (
                        transposed_upper_structure
                        and min(transposed_upper_structure, key=lambda p: p.ps).ps
                        <= authoritative_bass_pitch.ps
                    ):
                        logger.warning(
                            f"  CV: Upper structure still not above bass for {cs_obj.figure}. Re-adjusting forcefully."
                        )
                        shift_again = (
                            authoritative_bass_pitch.ps
                            + 2
                            - min(transposed_upper_structure, key=lambda p: p.ps).ps
                        )  # ベース+全音上くらいに
                        transposed_upper_structure = [
                            p.transpose(shift_again) for p in transposed_upper_structure
                        ]

                    upper_structure_pitches = transposed_upper_structure

                voiced_pitches_list = sorted(
                    [authoritative_bass_pitch] + upper_structure_pitches,
                    key=lambda p: p.ps,
                )
                logger.debug(
                    f"  CV: Final pitches with specified bass {authoritative_bass_pitch.nameWithOctave}: {[p.nameWithOctave for p in voiced_pitches_list]}"
                )

            # ケース2: 指定ベースがない、またはベースがルートと同じ場合 (従来の全体オクターブ調整)
            else:
                if voiced_pitches_list:  # voiced_pitches_listが空でないことを再確認
                    current_bottom_actual_pitch = min(
                        voiced_pitches_list, key=lambda p: p.ps
                    )

                    reference_pitch_name_for_octave = "C"  # デフォルト
                    if cs_obj.root():
                        reference_pitch_name_for_octave = cs_obj.root().name
                    elif (
                        cs_obj.bass()
                    ):  # ルートがなくてもベースがあるならそれを使う（この分岐ではルートと同じはず）
                        reference_pitch_name_for_octave = cs_obj.bass().name

                    target_bottom_octave_pitch = pitch.Pitch(
                        reference_pitch_name_for_octave
                    )
                    target_bottom_octave_pitch.octave = target_octave_for_bottom_note

                    # ボイシング結果の最低音と、目標オクターブの基準音との差からシフト量を決定
                    semitones_to_shift_final = (
                        round(
                            (
                                target_bottom_octave_pitch.ps
                                - current_bottom_actual_pitch.ps
                            )
                            / 12.0
                        )
                        * 12
                    )

                    if semitones_to_shift_final != 0:
                        logger.debug(
                            f"  CV: Final octave shift for {cs_obj.figure} (no distinct bass or bass is root): {semitones_to_shift_final} semitones to bring bottom near {target_bottom_octave_pitch.nameWithOctave}"
                        )
                        voiced_pitches_list = [
                            p.transpose(semitones_to_shift_final)
                            for p in voiced_pitches_list
                        ]

        return sorted(voiced_pitches_list, key=lambda p: p.ps)

    def compose(self, processed_chord_events: List[Dict]) -> stream.Part:
        chord_part = stream.Part(id="ChordsVoiced")
        try:
            chord_part.insert(0, self.default_instrument)
            if self.global_tempo:
                chord_part.append(tempo.MetronomeMark(number=self.global_tempo))
            if self.global_time_signature_obj and hasattr(
                self.global_time_signature_obj, "ratioString"
            ):
                chord_part.append(
                    meter.TimeSignature(self.global_time_signature_obj.ratioString)
                )
            else:
                chord_part.append(meter.TimeSignature("4/4"))  # フォールバック
        except Exception as e_init_part:
            logger.error(
                f"CV.compose: Error setting up initial part elements: {e_init_part}",
                exc_info=True,
            )

        if not processed_chord_events:
            logger.info("CV.compose: Received empty processed_chord_events.")
            return chord_part
        logger.info(
            f"CV.compose: Processing {len(processed_chord_events)} chord events."
        )

        for event_idx, event_data in enumerate(processed_chord_events):
            abs_offset = event_data.get(
                "humanized_offset_beats",
                event_data.get("absolute_offset", event_data.get("offset")),
            )
            humanized_duration = event_data.get(
                "humanized_duration_beats", event_data.get("q_length")
            )
            chord_symbol_str_original = event_data.get(  # サニタイズ前の元ラベルも保持
                "original_chord_label"
            )
            chord_symbol_str_for_voicing = event_data.get(
                "chord_symbol_for_voicing", chord_symbol_str_original
            )
            specified_bass_str_for_voicing = event_data.get(
                "specified_bass_for_voicing"
            )

            emotion_params = event_data.get("emotion_profile_applied", {})
            humanized_velocity = emotion_params.get("velocity", 64)
            humanized_articulation_str = emotion_params.get("articulation")

            part_specific_params = event_data.get("part_params", {})
            voicing_params = part_specific_params.get(
                "chords", part_specific_params.get("piano", {})  # pianoをフォールバック
            )
            voicing_style_name = voicing_params.get(
                "voicing_style", DEFAULT_VOICING_STYLE
            )
            target_oct = voicing_params.get(
                "target_octave", DEFAULT_CHORD_TARGET_OCTAVE_BOTTOM
            )
            num_voices = voicing_params.get("num_voices")  # Noneも許容

            # サニタイズ処理
            final_chord_symbol_to_parse = sanitize_chord_label(
                chord_symbol_str_for_voicing
            )

            if (
                final_chord_symbol_to_parse is None
                or final_chord_symbol_to_parse.lower() == "rest"
            ):
                logger.debug(
                    f"  CV Event {event_idx+1}: '{chord_symbol_str_original}' (parsed as '{final_chord_symbol_to_parse}') is Rest."
                )
                if abs_offset is not None and humanized_duration is not None:
                    rest_note = note.Rest(quarterLength=humanized_duration)
                    chord_part.insert(float(abs_offset), rest_note)
                    logger.debug(
                        f"  CV: Added Rest at {abs_offset:.2f} dur:{humanized_duration:.2f}"
                    )
                else:
                    logger.warning(
                        f"  CV Event {event_idx+1}: Skipping Rest due to None offset or duration. Offset: {abs_offset}, Duration: {humanized_duration}"
                    )
                continue

            try:
                cs = harmony.ChordSymbol(final_chord_symbol_to_parse)
                # 指定ベース音があれば適用 (ベース音自体もサニタイズを試みる)
                if specified_bass_str_for_voicing:
                    # ベース音は単純な音名であることが多いので、サニタイズは限定的に
                    # Bb -> B- のような変換は sanitize_chord_label が行う
                    final_bass_to_set = sanitize_chord_label(
                        specified_bass_str_for_voicing
                    )
                    if final_bass_to_set and final_bass_to_set.lower() != "rest":
                        try:
                            # music21のChordSymbol.bass()は音名文字列またはPitchオブジェクトを受け付ける
                            cs.bass(final_bass_to_set)
                            logger.debug(
                                f"  CV Event {event_idx+1}: Set bass to '{final_bass_to_set}' for chord '{cs.figure}'"
                            )
                        except Exception as e_bass:
                            logger.warning(
                                f"  CV Event {event_idx+1}: Failed to set bass '{final_bass_to_set}' (original bass: '{specified_bass_str_for_voicing}') for chord '{cs.figure}': {e_bass}"
                            )
                    elif final_bass_to_set and final_bass_to_set.lower() == "rest":
                        logger.debug(
                            f"  CV Event {event_idx+1}: Specified bass '{specified_bass_str_for_voicing}' sanitized to Rest. Ignoring bass for '{cs.figure}'."
                        )

            except exceptions21.ChordException as e_cs_harmony:  # music21固有の例外
                logger.error(
                    f"  CV Event {event_idx+1}: music21.harmony.ChordSymbol creation failed for '{final_chord_symbol_to_parse}' (original: '{chord_symbol_str_original}', bass: '{specified_bass_str_for_voicing}'): {e_cs_harmony}. Skipping."
                )
                continue
            except Exception as e_cs_general:  # その他の予期せぬ例外
                logger.error(
                    f"  CV Event {event_idx+1}: Unexpected error creating ChordSymbol from '{final_chord_symbol_to_parse}' (original: '{chord_symbol_str_original}', bass: '{specified_bass_str_for_voicing}'): {e_cs_general}. Skipping.",
                    exc_info=True,
                )
                continue

            if not cs.pitches:  # ChordSymbolが作れてもピッチがない場合
                logger.warning(
                    f"  CV Event {event_idx+1}: ChordSymbol '{cs.figure}' (parsed from '{final_chord_symbol_to_parse}') has no pitches after parsing. Skipping."
                )
                continue

            voiced_pitches = self._apply_voicing_style(
                cs,  # ベース音設定済みのChordSymbolを渡す
                voicing_style_name,
                target_octave_for_bottom_note=target_oct,
                num_voices_target=num_voices,
            )

            if not voiced_pitches:
                logger.warning(
                    f"  CV Event {event_idx+1}: No pitches after voicing for '{cs.figure}'. Skipping."
                )
                continue

            notes_for_final_chord = []
            for p_obj in voiced_pitches:
                n = note.Note(p_obj)
                # music21.volume.Volume オブジェクトを介してベロシティを設定
                vol = m21volume.Volume(velocity=humanized_velocity)
                n.volume = vol  # Noteオブジェクトに直接代入
                if humanized_articulation_str:
                    if humanized_articulation_str.lower() == "staccato":
                        n.articulations.append(articulations.Staccato())
                    elif humanized_articulation_str.lower() == "tenuto":
                        n.articulations.append(articulations.Tenuto())
                    elif (
                        humanized_articulation_str.lower() == "accent"
                    ):  # "accented" -> "accent"
                        n.articulations.append(articulations.Accent())
                    elif (
                        humanized_articulation_str.lower() == "legato"
                    ):  # Legatoはスラーで表現されることが多いが、ここでは明確な指示がないので何もしないか、ノート長で表現
                        pass  # humanized_duration で表現されていると期待
                notes_for_final_chord.append(n)

            if notes_for_final_chord:
                final_chord_obj = m21chord.Chord(
                    notes_for_final_chord, quarterLength=humanized_duration
                )
                if abs_offset is not None and humanized_duration is not None:
                    chord_part.insert(float(abs_offset), final_chord_obj)
                    logger.debug(
                        f"  CV: Added {final_chord_obj.pitchedCommonName} ({[p.nameWithOctave for p in final_chord_obj.pitches]}) vel:{humanized_velocity} art:'{humanized_articulation_str}' at {abs_offset:.2f} dur:{humanized_duration:.2f}"
                    )
                else:
                    logger.warning(
                        f"  CV Event {event_idx+1}: Skipping chord due to None offset or duration. Offset: {abs_offset}, Duration: {humanized_duration}"
                    )
            else:
                logger.warning(
                    f"  CV Event {event_idx+1}: No notes to form a chord for '{cs.figure}'."
                )

        logger.info(
            f"CV.compose: Finished. Part '{chord_part.id}' contains {len(list(chord_part.flatten().notesAndRests))} elements."
        )
        return chord_part

# --- END OF FILE generators/chord_voicer.py ---
