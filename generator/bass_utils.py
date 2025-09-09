# --- START OF FILE generator/bass_utils.py (修正版) ---
from __future__ import annotations
"""bass_utils.py
Low-level helpers for *bass line generation*.
"""

from typing import List, Sequence, Optional, Any, Dict, Tuple, Union
import random as _rand
import logging

from music21 import note, pitch, harmony, interval, scale as m21_scale

try:
    from utilities.scale_registry import ScaleRegistry as SR
except ImportError:
    logger_fallback_sr_bass = logging.getLogger(__name__ + ".fallback_sr_bass")
    logger_fallback_sr_bass.error("BassUtils: Could not import ScaleRegistry from utilities. Scale-aware functions might fail.")
    class SR:
        @staticmethod
        def get(tonic_str: Optional[str], mode_str: Optional[str]) -> m21_scale.ConcreteScale:
            logger_fallback_sr_bass.warning("BassUtils: Using dummy ScaleRegistry.get().")
            effective_tonic = tonic_str if tonic_str else "C"
            try: return m21_scale.MajorScale(pitch.Pitch(effective_tonic))
            except Exception: return m21_scale.MajorScale(pitch.Pitch("C"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
def mirror_pitches(
    vocal_notes: List[note.Note],
    tonic: pitch.Pitch,
    target_octave: int = 2,
) -> List[pitch.Pitch]:
    """Return mirrored pitches of the given vocal melody.

    Each note is reflected around ``tonic`` within a diatonic major scale and
    placed near ``target_octave``.
    """

    if not vocal_notes:
        return []

    tonic_for_scale = pitch.Pitch(tonic.name)
    tonic_for_scale.octave = max(0, target_octave - 1)
    scl = m21_scale.MajorScale(tonic_for_scale)

    mirrored: List[pitch.Pitch] = []
    for vn in vocal_notes:
        if not isinstance(vn, note.Note):
            continue
        v_pitch = vn.pitch
        deg = scl.getScaleDegreeFromPitch(v_pitch)
        if deg is None:
            mp = pitch.Pitch()
            mp.ps = 2 * tonic.ps - v_pitch.ps
        else:
            mirrored_deg = ((1 - deg) % 7) + 1
            mp = scl.pitchFromDegree(mirrored_deg)
        if mp.step == scl.tonic.step and mp.octave < target_octave:
            mp.octave = target_octave
        mirrored.append(mp)

    return mirrored

# ---------------------------------------------------------------------------
# --- 定義: music21.scale.nextPitch() の direction 引数用 ---
# bass_generator.py と同じ定数を定義
DIRECTION_UP = 1
DIRECTION_DOWN = -1
# ---------------------------------------------------------

def get_approach_note(
    from_pitch: pitch.Pitch,
    to_pitch: pitch.Pitch,
    scale_obj: Optional[m21_scale.ConcreteScale],
    approach_style: str = "chromatic_or_diatonic", # "chromatic_优先", "diatonic_only", "chromatic_only", "subdom_dom"
    max_step: int = 2, # 半音単位での最大距離 (2なら全音まで)
    preferred_direction: Optional[str] = None # "above", "below", None (近い方)
) -> Optional[pitch.Pitch]:
    """
    指定された2音間を繋ぐのに適したアプローチノート（1音）を提案します。
    (この関数の内部ロジックは変更ありません)
    """
    if not from_pitch or not to_pitch:
        return None

    candidates: List[Tuple[int, pitch.Pitch]] = [] # (優先度, ピッチ) - 数値が小さいほど高優先度

    if approach_style == "subdom_dom":
        if scale_obj:
            try:
                return scale_obj.pitchFromDegree(2)
            except Exception:
                pass
        return None

    for step in range(1, max_step + 1):
        # 下からのアプローチ
        p_below = to_pitch.transpose(-step)
        is_diatonic_below = scale_obj and scale_obj.getScaleDegreeFromPitch(p_below) is not None

        # 上からのアプローチ
        p_above = to_pitch.transpose(step)
        is_diatonic_above = scale_obj and scale_obj.getScaleDegreeFromPitch(p_above) is not None

        if approach_style == "diatonic_only":
            if is_diatonic_below: candidates.append((step, p_below))
            if is_diatonic_above: candidates.append((step, p_above))
        elif approach_style == "chromatic_only":
            if step == 1: # 半音のみ
                candidates.append((1, p_below))
                candidates.append((1, p_above))
        elif approach_style == "chromatic_or_diatonic": # デフォルト
            priority_below = 0
            if step == 1: priority_below = 1 if is_diatonic_below else 3
            elif step == 2 and is_diatonic_below: priority_below = 2

            priority_above = 0
            if step == 1: priority_above = 1 if is_diatonic_above else 3
            elif step == 2 and is_diatonic_above: priority_above = 2

            if priority_below > 0: candidates.append((priority_below, p_below))
            if priority_above > 0: candidates.append((priority_above, p_above))

    if not candidates:
        return None

    # 優先方向と現在の音からの距離でソート
    def sort_key(candidate_tuple):
        priority, p_cand = candidate_tuple # p -> p_cand に変更（引数名との衝突回避）
        distance_from_current = abs(p_cand.ps - from_pitch.ps)
        direction_score = 0
        if preferred_direction == "above" and p_cand.ps < to_pitch.ps: direction_score = 100 # ペナルティ
        if preferred_direction == "below" and p_cand.ps > to_pitch.ps: direction_score = 100 # ペナルティ
        return (priority, direction_score, distance_from_current)

    candidates.sort(key=sort_key)

    logger.debug(f"BassUtils (get_approach): From={from_pitch.name}, To={to_pitch.name}, Style='{approach_style}', PrefDir='{preferred_direction}', Candidates: {[(c[0], c[1].nameWithOctave, abs(c[1].ps - from_pitch.ps)) for c in candidates]}")

    return candidates[0][1] if candidates else None


def get_walking_note(
    prev_pitch: pitch.Pitch,
    next_root: pitch.Pitch,
    scale_pitches: Sequence[pitch.Pitch],
) -> pitch.Pitch:
    """Return a suitable walking note between prev_pitch and next_root."""
    candidates = [p for p in scale_pitches if p]
    if not candidates:
        candidates = [prev_pitch]
    best = min(
        candidates,
        key=lambda p: (abs(p.ps - next_root.ps), abs(p.ps - prev_pitch.ps)),
    )
    if best.ps == prev_pitch.ps:
        for step in [1, -1, 2, -2]:
            alt = prev_pitch.transpose(step)
            if abs(alt.ps - next_root.ps) <= abs(best.ps - next_root.ps):
                best = alt
                break
    return best


# --- 既存の関数 (walking_quarters, root_fifth_half, STYLE_DISPATCH, generate_bass_measure) は変更なし ---
# (ただし、STYLE_DISPATCH内のlambda関数でのlogger呼び出しは、このファイルスコープのloggerを使うように修正を推奨)
def walking_quarters(cs_now: harmony.ChordSymbol, cs_next: harmony.ChordSymbol, tonic: str, mode: str, octave: int = 3, vocal_notes_in_block: Optional[List[Dict]] = None) -> List[pitch.Pitch]:
    if vocal_notes_in_block: logger.debug(f"walking_quarters for {cs_now.figure if cs_now else 'N/A'}: Received {len(vocal_notes_in_block)} vocal notes.")
    scl = SR.get(tonic, mode)
    if not cs_now or not cs_now.root(): return [pitch.Pitch(f"C{octave}")] * 4
    root_now_init = cs_now.root(); root_now = root_now_init.transpose((octave - root_now_init.octave) * 12)
    effective_cs_next_root = cs_next.root() if cs_next and cs_next.root() else root_now_init
    root_next = effective_cs_next_root.transpose((octave - effective_cs_next_root.octave) * 12)

    beat1 = root_now
    options_b2_pitches = []
    if cs_now.third: options_b2_pitches.append(cs_now.third)
    if cs_now.fifth: options_b2_pitches.append(cs_now.fifth)
    if not options_b2_pitches: options_b2_pitches.append(root_now_init)

    beat2_candidate_pitch = _rand.choice(options_b2_pitches)
    beat2 = beat2_candidate_pitch.transpose((octave - beat2_candidate_pitch.octave) * 12)

    beat3_candidate_pitch = beat2.transpose(_rand.choice([-2, -1, 1, 2]))
    if scl.getScaleDegreeFromPitch(beat3_candidate_pitch) is None:
        temp_options_b3 = [p for p in options_b2_pitches if p.nameWithOctave != beat2_candidate_pitch.nameWithOctave]
        if not temp_options_b3: temp_options_b3 = [root_now_init]
        beat3_candidate_pitch = _rand.choice(temp_options_b3)
        beat3 = beat3_candidate_pitch.transpose((octave - beat3_candidate_pitch.octave) * 12)
    else: beat3 = beat3_candidate_pitch

    # ★★★ 修正点1: approach_note を get_approach_note に変更 ★★★
    # スタイルは diatonic_only を指定し、結果がなければ root_next を使用
    beat4 = get_approach_note(beat3, root_next, scl, approach_style="diatonic_only") or root_next

    if scl.getScaleDegreeFromPitch(beat4) is None: # get_approach_noteがスケール外を返した場合、またはroot_nextがスケール外の場合のフォールバック
        if scl.getScaleDegreeFromPitch(root_next) is not None:
            beat4 = root_next
        else:
            # ★★★ 修正点2: direction を定数に変更 ★★★
            direction_to_next = DIRECTION_UP if root_next.ps > beat3.ps else DIRECTION_DOWN
            beat4_alt_candidate = scl.nextPitch(beat3, direction=direction_to_next)

            actual_beat4_alt: Optional[pitch.Pitch] = None
            if isinstance(beat4_alt_candidate, pitch.Pitch):
                actual_beat4_alt = beat4_alt_candidate
            elif isinstance(beat4_alt_candidate, list) and beat4_alt_candidate and isinstance(beat4_alt_candidate[0], pitch.Pitch):
                actual_beat4_alt = beat4_alt_candidate[0]

            if actual_beat4_alt and (scl.getScaleDegreeFromPitch(actual_beat4_alt) is not None):
                beat4 = actual_beat4_alt
            else:
                beat4 = root_next # 最終フォールバック
    return [beat1, beat2, beat3, beat4]

def root_fifth_half(cs_now: harmony.ChordSymbol, cs_next: harmony.ChordSymbol, tonic: str, mode: str, octave: int = 3, vocal_notes_in_block: Optional[List[Dict]] = None) -> List[pitch.Pitch]:
    if vocal_notes_in_block: logger.debug(f"root_fifth_half for {cs_now.figure if cs_now else 'N/A'}: Received {len(vocal_notes_in_block)} vocal notes.")
    if not cs_now or not cs_now.root(): return [pitch.Pitch(f"C{octave}")] * 4
    root_init = cs_now.root(); root = root_init.transpose((octave - root_init.octave) * 12)
    fifth_init = cs_now.fifth
    if fifth_init is None: fifth_init = root_init.transpose(12)
    fifth = fifth_init.transpose((octave - fifth_init.octave) * 12)
    return [root, fifth, root, fifth]

STYLE_DISPATCH: Dict[str, Any] = {
    "root_only": lambda cs_now, cs_next, tonic, mode, octave, vocal_notes_in_block, **k: ([cs_now.root().transpose((octave - cs_now.root().octave) * 12)] * 4 if cs_now and cs_now.root() else [pitch.Pitch(f"C{octave}")]*4),
    "simple_roots": lambda cs_now, cs_next, tonic, mode, octave, vocal_notes_in_block, **k: ([cs_now.root().transpose((octave - cs_now.root().octave) * 12)] * 4 if cs_now and cs_now.root() else [pitch.Pitch(f"C{octave}")]*4),
    "root_fifth": root_fifth_half,
    "walking": walking_quarters,
}
def generate_bass_measure(style: str, cs_now: harmony.ChordSymbol, cs_next: harmony.ChordSymbol, tonic: str, mode: str, octave: int = 3, vocal_notes_in_block: Optional[List[Dict]] = None) -> List[note.Note]:
    func = STYLE_DISPATCH.get(style)
    if not func:
        logger.warning(f"BassUtils: Style '{style}' not found in STYLE_DISPATCH. Falling back to 'root_only'.")
        style = "root_only"; func = STYLE_DISPATCH[style]

    # cs_now が None または root を持たない場合のフォールバック処理を強化
    if cs_now is None or not hasattr(cs_now, 'root') or not cs_now.root():
        logger.warning(f"BassUtils: Invalid ChordSymbol '{cs_now}' for style '{style}'. Using C major as fallback.")
        cs_now = harmony.ChordSymbol("C") # デフォルトのコードを設定

    initial_pitches: List[pitch.Pitch]
    try:
        initial_pitches = func(cs_now=cs_now, cs_next=cs_next, tonic=tonic, mode=mode, octave=octave, vocal_notes_in_block=vocal_notes_in_block)
    except Exception as e_dispatch:
        logger.error(f"BassUtils: Error dispatching style '{style}' for chord '{cs_now.figure if cs_now else 'N/A'}': {e_dispatch}", exc_info=True)
        root_p_obj = cs_now.root() if cs_now and cs_now.root() else pitch.Pitch("C") # cs_now.root() の存在確認
        initial_pitches = [root_p_obj.transpose((octave - root_p_obj.octave) * 12)] * 4 if root_p_obj else [pitch.Pitch(f"C{octave}")] * 4

    if not initial_pitches or len(initial_pitches) != 4: # ピッチリストの整形
        root_p_obj = cs_now.root() if cs_now and cs_now.root() else pitch.Pitch("C")
        fill_pitch = root_p_obj.transpose((octave - root_p_obj.octave) * 12) if root_p_obj else pitch.Pitch(f"C{octave}")
        if not initial_pitches: initial_pitches = [fill_pitch] * 4
        elif len(initial_pitches) < 4: initial_pitches.extend([fill_pitch] * (4 - len(initial_pitches)))
        else: initial_pitches = initial_pitches[:4]

    final_adjusted_pitches: List[pitch.Pitch] = []
    scl_obj = SR.get(tonic, mode)
    for beat_idx, p_bass_initial in enumerate(initial_pitches):
        adjusted_pitch_current_beat = p_bass_initial
        vocal_notes_on_this_beat = [vn for vn in (vocal_notes_in_block or []) if beat_idx <= vn.get("block_relative_offset", -999.0) < (beat_idx + 1.0)]
        if vocal_notes_on_this_beat and cs_now and cs_now.root(): # cs_now と root の存在確認
            root_pc = cs_now.root().pitchClass
            fifth_pc = cs_now.fifth.pitchClass if cs_now.fifth else (root_pc + 7) % 12
            for vn_data in vocal_notes_on_this_beat:
                if not p_bass_initial: continue
                try:
                    vocal_pitch_obj = pitch.Pitch(vn_data["pitch_str"])
                    if vocal_pitch_obj.pitchClass == adjusted_pitch_current_beat.pitchClass:
                        current_bass_is_root = (adjusted_pitch_current_beat.pitchClass == root_pc)
                        candidate_pitch: Optional[pitch.Pitch] = None
                        if current_bass_is_root and cs_now.fifth:
                            candidate_pitch = cs_now.fifth.transpose((octave - cs_now.fifth.octave) * 12)
                            if scl_obj.getScaleDegreeFromPitch(candidate_pitch) is not None:
                                adjusted_pitch_current_beat = candidate_pitch
                            elif cs_now.third: # fifth がダメなら third を試す
                                candidate_pitch = cs_now.third.transpose((octave - cs_now.third.octave) * 12)
                                if scl_obj.getScaleDegreeFromPitch(candidate_pitch) is not None:
                                    adjusted_pitch_current_beat = candidate_pitch
                        elif cs_now.root() and adjusted_pitch_current_beat.pitchClass == fifth_pc: # fifth だったら root に変更
                             adjusted_pitch_current_beat = cs_now.root().transpose((octave - cs_now.root().octave) * 12)
                        break
                except Exception: pass # Ignore errors in vocal note processing
        final_adjusted_pitches.append(adjusted_pitch_current_beat)

    notes_out: List[note.Note] = []
    for p_obj_final in final_adjusted_pitches:
        if p_obj_final is None: p_obj_final = pitch.Pitch(f"C{octave}") # None チェック
        n = note.Note(p_obj_final); n.quarterLength = 1.0; notes_out.append(n)
    return notes_out
# --- END OF FILE generator/bass_utils.py ---