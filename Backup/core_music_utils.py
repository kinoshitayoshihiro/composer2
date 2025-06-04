# --- START OF FILE utilities/core_music_utils.py (o3さん提案適用 + Haruさん既存コードベース) ---
import music21
import logging
from music21 import meter, harmony, pitch, scale
from typing import Optional, Dict, Any, List
import re

logger = logging.getLogger(__name__)

MIN_NOTE_DURATION_QL: float = 0.125 # o3さん指摘: 設定ファイル化検討

def get_time_signature_object(ts_str: Optional[str]) -> meter.TimeSignature:
    ts_str = ts_str or "4/4"
    try:
        return meter.TimeSignature(ts_str)
    except meter.MeterException:
        logger.warning(f"CoreUtils (GTSO): Invalid TS '{ts_str}'. Default 4/4.")
        return meter.TimeSignature("4/4")
    except Exception as e_ts:
        logger.error(f"CoreUtils (GTSO): Error for TS '{ts_str}': {e_ts}. Default 4/4.", exc_info=True)
        return meter.TimeSignature("4/4")

def build_scale_object(mode_str: Optional[str], tonic_str: Optional[str]) -> scale.ConcreteScale:
    mode_key = (mode_str or "major").lower()
    tonic_val = tonic_str or "C"
    try:
        tonic_p = pitch.Pitch(tonic_val)
    except Exception:
        logger.error(f"CoreUtils (BuildScale): Invalid tonic '{tonic_val}'. Defaulting to C.")
        tonic_p = pitch.Pitch("C")
    
    mode_map: Dict[str, Any] = {
        "ionian": scale.MajorScale, "major": scale.MajorScale,
        "dorian": scale.DorianScale, "phrygian": scale.PhrygianScale,
        "lydian": scale.LydianScale, "mixolydian": scale.MixolydianScale,
        "aeolian": scale.MinorScale, "minor": scale.MinorScale,
        "locrian": scale.LocrianScale,
        "harmonicminor": scale.HarmonicMinorScale,
        "melodicminor": scale.MelodicMinorScale,
    }
    scl_cls = mode_map.get(mode_key, scale.MajorScale)
    if scl_cls is scale.MajorScale and mode_key not in mode_map :
        logger.warning(f"CoreUtils (BuildScale): Unknown mode '{mode_key}'. Using MajorScale with {tonic_p.name}.")
    try:
        return scl_cls(tonic_p)
    except Exception as e_create:
        logger.error(f"CoreUtils (BuildScale): Error creating '{scl_cls.__name__}' for {tonic_p.name}: {e_create}. Fallback C Major.", exc_info=True)
        return scale.MajorScale(pitch.Pitch("C"))

_ROOT_RE_STRICT = re.compile(r'^([A-G](?:[#b]{1,2}|[ns])?)(?![#b])') # 厳密なルート音 (ダブルシャープ/フラット、ナチュラル許容)
_ROOT_RE_SIMPLE = re.compile(r'^[A-G][#b]?') # シンプルなルート音 (シングルシャープ/フラットのみ)


def _expand_tension_block_final_polish(seg: str) -> str:
    seg = seg.strip().lower()
    if not seg: return ""
    if seg.startswith(("#", "b")): return seg 
    if seg.startswith("add"):
        match_add_num = re.match(r'add(\d+)', seg)
        if match_add_num: return f"add{match_add_num.group(1)}"
        logger.debug(f"CoreUtils (_ETB): Invalid 'add' format '{seg}', treating as empty.")
        return "" 
    if seg.isdigit(): return f"add{seg}" 
    if seg in ["omit3", "omit5", "omitroot"]: return seg
    logger.debug(f"CoreUtils (_ETB): Unknown tension '{seg}', passing as is.")
    return seg

def _addify_callback_final_polish(match: re.Match) -> str:
    prefix = match.group(1) or ""
    number = match.group(2)
    # 品質やテンションキーワードで終わる場合は add を付けない (より多くのキーワードを考慮)
    if prefix.lower().endswith(('sus', 'sus4', 'sus2', 'add', 'maj', 'm', 'min', 'dim', 'aug', 'ø', 'alt',
                                '7', '9', '11', '13', 'b5', '#5', 'b9', '#9', '#11', 'b13',
                                'th', 'nd', 'rd', 'st')): # 13thなども考慮
        return match.group(0) 
    return f'{prefix}add{number}'

def sanitize_chord_label(label: Optional[str]) -> Optional[str]:
    """
    コードラベルをmusic21が解釈しやすい形式にサニタイズし、
    解釈不能な場合でもルート音だけでも抽出を試みる。(o3さん提案ベースで強化)

    Args:
        label (Optional[str]): サニタイズするコードラベル文字列。

    Returns:
        Optional[str]: サニタイズされたコードラベル文字列。解釈不能な場合はルート音、
                       それも無理ならNone (Restとして扱われることを意図)。
    """
    if not label or not isinstance(label, str):
        logger.debug(f"CoreUtils (sanitize): Label '{label}' is None or not a string. Interpreting as Rest (returning None).")
        return None 
    
    original_label = label
    sanitized = label.strip()

    if not sanitized or sanitized.lower() in {"rest", "r", "nc", "n.c.", "silence", "-"}:
        logger.debug(f"CoreUtils (sanitize): Label '{original_label}' is a direct Rest match. Returning None.")
        return None

    # 0a. ワードベースの品質変換
    word_map = {
        r'(?i)\b([A-Ga-g][#b\-]*)\s+minor\b': r'\1m',
        r'(?i)\b([A-Ga-g][#b\-]*)\s+major\b': r'\1maj', # majはmusic21が解釈できる
        r'(?i)\b([A-Ga-g][#b\-]*)\s+diminished\b':   r'\1dim',
        r'(?i)\b([A-Ga-g][#b\-]*)\s+augmented\b':   r'\1aug',
        r'(?i)\b([A-Ga-g][#b\-]*)\s+dominant\s*7\b': r'\17', # dominant 7 -> 7
    }
    for pat, rep in word_map.items(): sanitized = re.sub(pat, rep, sanitized)
    
    # 0b. ルート音の先頭文字を大文字化、それ以外を小文字化 (例: cMI -> Cmi)
    # ただし、"m" や "M" は品質を示すため、単純な小文字化は避ける。ルート音のみ対象。
    root_match_for_case = _ROOT_RE_STRICT.match(sanitized)
    if root_match_for_case:
        root_part = root_match_for_case.group(1)
        rest_part = sanitized[len(root_part):]
        sanitized = root_part[0].upper() + root_part[1:].lower() + rest_part
    else: # ルートが取れない場合は全体を一度大文字化してmusic21に任せることも検討したが、ノイズが多いので一旦そのまま
        logger.debug(f"CoreUtils (sanitize): Could not identify clear root for case normalization in '{sanitized}'.")


    # 1. フラット/シャープ正規化 & SUS正規化
    sanitized = re.sub(r'^([A-G])bb', r'\1--', sanitized) # ダブルフラット
    sanitized = re.sub(r'^([A-G])b(?![#b-])', r'\1-', sanitized) # シングルフラット (bの後に記号が続かない場合)
    sanitized = re.sub(r'/([A-G])bb', r'/\1--', sanitized)
    sanitized = re.sub(r'/([A-G])b(?![#b-])', r'/\1-', sanitized)
    
    sanitized = re.sub(r'(?i)([A-G][#\-]?(?:\d+)?)(sus)(?![24\d])', r'\g<1>sus4', sanitized) 
    sanitized = re.sub(r'(?i)(sus)([24])', r'sus\2', sanitized) 

    # 2. 括弧の不均衡修正 (o3さん提案のロジックをベースに)
    open_paren_count = sanitized.count('(')
    close_paren_count = sanitized.count(')')
    if open_paren_count > close_paren_count and open_paren_count == 1 : # 開き括弧が1つ多く、閉じ括弧がない場合
        logger.info(f"CoreUtils (sanitize): Detected unclosed parenthesis in '{original_label}'. Attempting recovery.")
        parts = sanitized.split('(', 1)
        if len(parts) == 2:
            base_part, content_after_paren = parts
            if content_after_paren.strip(): # 括弧の後に内容がある
                # 括弧内のテンションらしきものを展開
                recovered_tensions = "".join(_expand_tension_block_final_polish(p) for p in re.split(r'[,\s]+', content_after_paren) if p)
                if recovered_tensions:
                    sanitized = base_part + recovered_tensions
                    logger.info(f"CoreUtils (sanitize): Recovered from unclosed: '{content_after_paren}' -> '{recovered_tensions}', result: '{sanitized}'")
                else: # 有効なテンションがなければ括弧以降を削除
                    sanitized = base_part
                    logger.info(f"CoreUtils (sanitize): No valid tensions from unclosed content '{content_after_paren}', kept base: '{sanitized}'")
            else: # 括弧の後に内容がない場合は括弧を削除
                sanitized = base_part
                logger.info(f"CoreUtils (sanitize): Empty content after unclosed parenthesis, kept base: '{sanitized}'")
    elif open_paren_count < close_paren_count: # 閉じ括弧が多い場合 (単純に削除)
        sanitized = sanitized.replace(')', '', open_paren_count - close_paren_count)
        logger.info(f"CoreUtils (sanitize): Removed excess closing parentheses from '{original_label}' -> '{sanitized}'")


    # 3. altコード展開
    sanitized = re.sub(r'([A-Ga-g][#\-]?)(?:7)?alt(?:ered)?', r'\g<1>7#9b13', sanitized, flags=re.IGNORECASE) # altered も考慮
    sanitized = sanitized.replace('badd13', 'b13').replace('#add13', '#13') # add表記をシンプルに

    # 4. 括弧の平坦化 (再帰的な括弧や複雑なネストには限定的)
    # (o3さんのループ方式を参考に、より安全な単一パス処理に)
    match_paren = re.match(r'^(.*?)\(([^)]+)\)(.*)$', sanitized)
    if match_paren:
        base, inner, suf = match_paren.groups()
        # 括弧内をカンマやスペースで区切り、各要素をテンションとして展開
        expanded_inner_parts = [_expand_tension_block_final_polish(p) for p in re.split(r'[,\s]+', inner) if p.strip()]
        expanded_inner = "".join(expanded_inner_parts)
        sanitized = base + expanded_inner + suf
        if sanitized != original_label:
             logger.debug(f"CoreUtils (sanitize): Parentheses flattened: '{original_label}' -> '{sanitized}'")


    # 5. 品質関連の正規化 (順序が重要になる場合がある)
    qual_map = {
        # より具体的なものから先に
        r'(?i)half-diminished7': 'm7b5', r'(?i)halfdim7': 'm7b5', r'(?i)ø7': 'm7b5',
        r'(?i)diminished7': 'dim7', r'(?i)dim7': 'dim7',
        r'(?i)minor-major7': 'mM7', r'(?i)minmaj7': 'mM7', r'(?i)m\(maj7\)': 'mM7',
        r'(?i)major7': 'maj7', r'(?i)maj7': 'maj7', r'(?i)M7': 'maj7',
        r'(?i)minor7': 'm7', r'(?i)min7': 'm7',
        r'(?i)dominant7': '7', r'(?i)dom7': '7',
        # 一般的な品質
        r'(?i)ø': 'm7b5', # ø単体はm7b5として扱うことが多い
        r'(?i)diminished': 'dim', r'(?i)dim': 'dim',
        r'(?i)augmented': 'aug', r'(?i)aug': 'aug', r'(?i)\+': 'aug', # + も aug として扱う
        r'(?i)minor': 'm', r'(?i)min': 'm',
        r'(?i)major(?![\d#b])': '', # "Cmajor" -> "C", ただし "Cmajor7" は残す
        r'(?i)maj(?![\d#b])': '',   # "Cmaj" -> "C"
        r'(?i)M(?!aj|\d|[#b])': '', # "CM" -> "C" (ただし CM7 は残す)
    }
    for pat, rep in qual_map.items():
        sanitized = re.sub(pat, rep, sanitized)
                                                       
    # 6. "add"補完 (o3さんコールバック方式を参考に、より限定的に適用)
    # 例: C9 -> C7add9, Am11 -> Am7add11add9 (music21はC9をC dominant ninthと解釈するが、add9を明示したい場合)
    # ただし、これは音楽的解釈に踏み込むため、慎重に。今回はシンプルなadd補完に留める。
    # C(9) -> Cadd9, C(11) -> Cadd11
    sanitized = re.sub(r'([A-G][#b\-]*(?:m|M|maj|dim|aug|sus\d*)?)(\d{2,})', _addify_callback_final_polish, sanitized)


    # 7. maj9(#...) -> maj7(#...)add9 のような変換 (music21の解釈に合わせるか、明示的表記にするかのトレードオフ)
    # 今回はmusic21の解釈力を信じ、過度な変換は避ける方向で。
    # sanitized = re.sub(r'(maj)9(#\d+)', r'\g<1>7\g<2>add9', sanitized, flags=re.IGNORECASE)

    # 8. susコードの重複等最終修正
    sanitized = re.sub(r'(?i)sus4sus4', 'sus4', sanitized) # 重複susの修正
    sanitized = re.sub(r'(?i)sus2sus2', 'sus2', sanitized)
    sanitized = re.sub(r'(?i)sus42', 'sus4', sanitized) # Dsus42 -> Dsus4
    sanitized = re.sub(r'(?i)sus24', 'sus2', sanitized) # Dsus24 -> Dsus2
    
    # 8b. 重複addの圧縮
    sanitized = re.sub(r'(?i)(add)(add)+', r'\1', sanitized) # addadd -> add

    # 9. 全体的なスペース・カンマの最終除去
    sanitized = re.sub(r'[,\s]', '', sanitized)
    
    # 10. 末尾に残った可能性のある不要な文字や記号の除去
    sanitized = re.sub(r'[^a-zA-Z0-9#\-/\u00f8\+]+$', '', sanitized) # +（aug）は残す
    if sanitized.lower().endswith("add") and not sanitized.lower()[-4:-3].isdigit(): # 数字が続くaddは残す(add13など)
        logger.info(f"CoreUtils (sanitize): Removing trailing 'add' from '{sanitized}' (orig: '{original_label}')")
        sanitized = sanitized[:-3]

    if not sanitized: 
        logger.info(f"CoreUtils (sanitize): Label '{original_label}' resulted in empty string after sanitization. Interpreting as Rest (returning None).")
        return None

    # ログ出力はget_music21_chord_object側で行うため、ここでは最終的なsanitized文字列を返す
    if sanitized != original_label:
        logger.debug(f"CoreUtils (sanitize): Sanitized '{original_label}' -> '{sanitized}'")
    
    return sanitized


def get_music21_chord_object(chord_label_str: Optional[str], current_key: Optional[scale.ConcreteScale] = None) -> Optional[harmony.ChordSymbol]:
    """
    サニタイズされたコードラベル文字列からmusic21のChordSymbolオブジェクトを生成します。
    パース失敗時はNoneを返します。
    """
    if not isinstance(chord_label_str, str) or not chord_label_str.strip():
        logger.debug(f"CoreUtils (get_obj): Input '{chord_label_str}' is empty or not a string. Interpreting as Rest (returning None).")
        return None

    sanitized_label = sanitize_chord_label(chord_label_str) 

    if not sanitized_label: 
        logger.info(f"CoreUtils (get_obj): sanitize_chord_label returned None for '{chord_label_str}'. Interpreting as Rest.")
        return None # sanitize_chord_labelがNoneを返したら、それはRest扱い

    cs = None
    try:
        cs = harmony.ChordSymbol(sanitized_label) 
        if not cs.pitches:
            # パースは成功したが音がない場合 (例: "Rest"という名前のChordSymbolや、解釈不能だがエラーにならないケース)
            logger.warning(f"CoreUtils (get_obj): Parsed '{sanitized_label}' (from '{chord_label_str}') but it has NO PITCHES (figure: {cs.figure}). Interpreting as Rest (None).")
            return None
        
        # ルート音で始まっているか最終確認 (例: "add9"だけなどは無効)
        # music21.harmony.ChordSymbolはルートなしでもエラーにならないことがあるため
        if not cs.root():
             logger.warning(f"CoreUtils (get_obj): Parsed '{sanitized_label}' (from '{chord_label_str}') but NO ROOT found (figure: {cs.figure}). Interpreting as Rest (None).")
             return None

        logger.info(f"CoreUtils (get_obj): Successfully parsed '{sanitized_label}' (from '{chord_label_str}') as '{cs.figure}' with pitches: {[p.nameWithOctave for p in cs.pitches]}")
        return cs
    except harmony.HarmonyException as e_harm:
        logger.warning(f"CoreUtils (get_obj): music21.harmony.HarmonyException for '{sanitized_label}' (from '{chord_label_str}'): {e_harm}. Attempting root-only fallback.")
        # ルート音だけでも試す (o3さん提案)
        root_match = _ROOT_RE_STRICT.match(sanitized_label) # より厳密なルート抽出
        if not root_match: root_match = _ROOT_RE_SIMPLE.match(sanitized_label) # シンプルなルート抽出も試す

        if root_match:
            root_str = root_match.group(0)
            logger.info(f"CoreUtils (get_obj): Falling back to root-only: '{root_str}' for original '{chord_label_str}'.")
            try:
                cs_root_only = harmony.ChordSymbol(root_str)
                if cs_root_only.pitches and cs_root_only.root():
                    logger.info(f"CoreUtils (get_obj): Successfully parsed root-only fallback '{root_str}' as '{cs_root_only.figure}'.")
                    return cs_root_only
                else:
                    logger.error(f"CoreUtils (get_obj): Root-only fallback '{root_str}' parsed but no pitches/root. Original '{chord_label_str}' -> Rest.")
                    return None
            except Exception as e_root_parse:
                logger.error(f"CoreUtils (get_obj): Exception on root-only fallback parse for '{root_str}' (from '{chord_label_str}'): {e_root_parse}. -> Rest.")
                return None
        else:
            logger.error(f"CoreUtils (get_obj): Could not extract root for fallback from '{sanitized_label}' (orig: '{chord_label_str}'). -> Rest.")
            return None
    except Exception as e_other: # その他の予期せぬエラー
        logger.error(f"CoreUtils (get_obj): Unexpected Exception for '{sanitized_label}' (from '{chord_label_str}'): {type(e_other).__name__}: {e_other}. -> Rest.", exc_info=True)
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s.%(funcName)s: %(message)s')
    
    # 「Masterpiece Edition - Final Polish」の期待値 + o3さん提案のテストケース
    final_expected_outcomes_masterpiece_final_polish = {
        "E7(b9)": "E7b9", "C7(#9,b13)": "C7#9b13", "C7(b9,#11,add13)": "C7b9#11add13",
        "C7alt": "C7#9b13", "Fmaj7(add9)": "Fmaj7add9", "Fmaj7(add9,13)": "Fmaj7add13", 
        "Bbmaj7(#11)": "B-maj7#11", 
        "Cø7": "Cm7b5", "Cm7b5": "Cm7b5", "Cø": "Cm7b5",
        "Am7(add11)": "Am7add11", "Am7(add11": "Am7add11", # 括弧閉じ不足のリカバリケース
        "Dsus": "Dsus4", "Gsus4": "Gsus4", "Asus2": "Asus2", "Csus44": "Csus4",
        "Bbmaj9(#11)": "B-maj7#11add9", # music21は B-maj9#11 と解釈するが、add9を明示する形も許容
        "F#7": "F#7", "Calt": "C7#9b13", "silence": None,
        "Cminor7": "Cm7", "Gdominant7": "G7",
        "Bb": "B-", "Ebm": "E-m", "F#": "F#", "Dbmaj7": "D-maj7",
        "G7SUS": "G7sus4", "G7sus": "G7sus4", "Gsus": "Gsus4",
        "d minor": "Dm", "e dim": "Edim", "C major7": "Cmaj7", "G AUG": "Gaug", "C major": "C", # Cmajor -> C
        "bad(input": None, "": None, "N.C.": None, "  Db  ": "D-",
        "GM7(": "GM7", "G diminished": "Gdim", 
        "C/Bb": "C/B-", "CbbM7": "C--M7", "C##M7": "C##M7", 
        "C(omit3)": "Comit3", "Fmaj7(add9": "Fmaj7add9",
        "Am7addadd11": "Am7add11", 
        "Gadd": "Gadd", # G(add) -> Gadd (music21はGメジャーとして解釈するが、addを保持)
        "Fmaj7add": "Fmaj7", # 末尾のaddは削除される
        "Dm7addadd13": "Dm7add13",
        "Cadd": "Cadd",
        "Fmaj7addaddadd13": "Fmaj7add13",
        "B-maj9#11add": "B-maj7#11add9",
        # o3さん提案のテストケース
        "Gsus2/D": "Gsus2/D", # music21はこれを正しくパースできるはず
        "Cadd9": "Cadd9",
        "Am(add9)": "Amadd9",
        "Csus4/G": "Csus4/G",
        "Am7add11": "Am7add11",
        "G13": "G13",
        "Bo7": "Bdim7", # B diminished seventh
        "E7b9": "E7b9",
        "Fmaj7add9": "Fmaj7add9",
        "C/E": "C/E",
        "Gm7b5": "Gm7b5",
        "C7#9b13": "C7#9b13",
        "Dm7add13": "Dm7add13",
        "Em7b5": "Em7b5",
        "A7b9": "A7b9",
        "C13": "C13",
        "Bbmaj9#11": "B-maj7#11add9", # 既存のテストと重複するが確認のため
        "C/Bb": "C/B-",
        "D7sus4": "D7sus4",
        "Fmaj7add9add13": "Fmaj7add13", # add9とadd13が共存
        "Ebmaj7(#11)": "E-maj7#11",
        "Bbmaj7/D": "B-maj7/D",
    }
    
    other_tests_masterpiece_final_polish = [ "Rest", "Dsus2", "Dsus4", "Csu" ] # Csuのような独自表記もテスト
    
    all_labels_to_test_masterpiece_final_polish = sorted(list(set(list(final_expected_outcomes_masterpiece_final_polish.keys()) + other_tests_masterpiece_final_polish)))

    print("\n--- Running sanitize_chord_label Test Cases (Harugoro x o3 Masterpiece Final Polish - Enhanced) ---")
    s_parses_mfp = 0; f_parses_mfp = 0; r_count_mfp = 0; exp_match_mfp = 0; exp_mismatch_mfp = 0

    for label_orig in all_labels_to_test_masterpiece_final_polish:
        expected_val = final_expected_outcomes_masterpiece_final_polish.get(label_orig)
        sanitized_res = sanitize_chord_label(label_orig) 
        
        eval_str = ""
        if expected_val is None: # NoneはRest扱い
            if sanitized_res is None: eval_str = "✔ (OK, Interpreted as Rest)"; exp_match_mfp +=1
            else: eval_str = f"✘ (Exp: Rest (None), Got: '{sanitized_res}')"; exp_mismatch_mfp +=1
        elif sanitized_res == expected_val:
            eval_str = "✔ (OK, Expected Sanitized Form)"; exp_match_mfp +=1
        else: 
            eval_str = f"✘ (Exp Sanitized: '{expected_val}', Got: '{sanitized_res}')"; exp_mismatch_mfp +=1
        
        print(f"Original: '{label_orig:<20}' -> Sanitized: '{str(sanitized_res):<25}' {eval_str}")

        cs_obj_mfp = get_music21_chord_object(label_orig) # オリジナルラベルでオブジェクト取得を試す
        
        if cs_obj_mfp:
            try: fig_disp = cs_obj_mfp.figure
            except: fig_disp = "[Error Retrieving Figure]"
            # pitches_str = ", ".join([p.nameWithOctave for p in cs_obj_mfp.pitches]) if cs_obj_mfp.pitches else "No Pitches"
            pitches_str = ", ".join(sorted(list(set(p.name for p in cs_obj_mfp.pitches)))) if cs_obj_mfp.pitches else "No Pitches" # ピッチクラスでソートして重複排除
            print(f"  └─> music21 obj: '{fig_disp:<25}' (Unique Pitch Classes: {pitches_str})")
            s_parses_mfp += 1
        elif sanitized_res is None : # sanitize_chord_label が None を返した場合 (Rest扱い)
            print(f"  └─> Interpreted as Rest (sanitize_chord_label returned None).")
            r_count_mfp += 1
        else: # music21がパースできなかったか、ピッチがなかった場合
            print(f"  └─> music21 FAILED or NO PITCHES for sanitized '{sanitized_res}' (from original '{label_orig}')")
            f_parses_mfp += 1

    print(f"\n--- Test Summary (Harugoro x o3 Masterpiece Final Polish - Enhanced) ---")
    total_labels_mfp = len(all_labels_to_test_masterpiece_final_polish)
    attempted_to_parse_mfp = total_labels_mfp - r_count_mfp # Restとして扱われたものを除く

    print(f"Total unique labels processed: {total_labels_mfp}")
    if exp_match_mfp + exp_mismatch_mfp > 0:
      print(f"Matches with expected sanitization outcome (incl. Rest as None): {exp_match_mfp}")
      print(f"Mismatches with expected sanitization outcome: {exp_mismatch_mfp}")
    print(f"Successfully parsed by music21 (Chord obj with pitches): {s_parses_mfp} / {attempted_to_parse_mfp} non-Rest attempts")
    print(f"Failed to parse by music21 (or no pitches) after sanitization: {f_parses_mfp}")
    print(f"Interpreted as 'Rest' (sanitize_chord_label returned None): {r_count_mfp}")
    
    # music21が解釈できたもの + Restとして正しく処理されたもの を成功とみなす
    functional_success_count_mfp = s_parses_mfp + r_count_mfp
    overall_success_rate_mfp = (functional_success_count_mfp / total_labels_mfp * 100) if total_labels_mfp > 0 else 0
    print(f"Estimated overall functional success (parsed by music21 + correctly identified as Rest): {overall_success_rate_mfp:.2f}%")

# --- END OF FILE utilities/core_music_utils.py ---
