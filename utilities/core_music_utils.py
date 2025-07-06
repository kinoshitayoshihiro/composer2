try:
    from music21 import pitch, harmony, key, meter, interval
except ModuleNotFoundError as e:  # pragma: no cover - dependency check
    raise ModuleNotFoundError(
        "music21 is required. Please run 'pip install -r requirements.txt'."
    ) from e
import re
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)
_ROOT_RE_STRICT = re.compile(r'^([A-G](?:[#b]{1,2}|[ns])?)(?![#b])')
MIN_NOTE_DURATION_QL = 0.0625 # 64分音符程度を最小音価とする

def get_time_signature_object(ts_str: Optional[str]) -> Optional[meter.TimeSignature]:
    if not ts_str:
        logger.debug("CoreUtils (get_ts): Received None or empty string for time signature.")
        return None
    try:
        return meter.TimeSignature(ts_str)
    except Exception as e:
        logger.error(f"CoreUtils (get_ts): Invalid time signature string: '{ts_str}'. Error: {e}")
        return None

def get_key_signature_object(tonic: Optional[str], mode: Optional[str] = 'major') -> Optional[key.Key]:
    if not tonic:
        logger.debug("CoreUtils (get_key): Received None or empty string for tonic.")
        return None
    try:
        actual_mode = mode.lower() if mode else 'major'
        return key.Key(tonic, actual_mode)
    except Exception as e:
        logger.error(f"CoreUtils (get_key): Invalid key signature: Tonic='{tonic}', Mode='{mode}'. Error: {e}")
        return None

def calculate_note_times(current_beat: float, duration_beats: float, bpm: float) -> Tuple[float, float]:
    if bpm <= 0:
        logger.warning("CoreUtils (calc_times): BPM must be positive. Returning (0,0).")
        return 0.0, 0.0
    start_time_seconds = (current_beat / bpm) * 60.0
    duration_seconds = (duration_beats / bpm) * 60.0
    end_time_seconds = start_time_seconds + duration_seconds
    return start_time_seconds, end_time_seconds

def get_pitch_object_with_octave(name: str, oct: int) -> Optional[pitch.Pitch]:
    """指定された音名とオクターブでmusic21.pitch.Pitchオブジェクトを作成する"""
    try:
        p = pitch.Pitch(name)
        p.octave = oct
        return p
    except Exception as e:
        logger.error(f"CoreUtils (get_pitch_oct): Error creating pitch '{name}{oct}': {e}")
        return None

def sanitize_chord_label(label: Optional[str]) -> Optional[str]:
    """
    入力されたコードラベルを music21 が解釈しやすい形式に正規化する。
    - 全角英数を半角に
    - 不要な空白削除
    - ルート音のフラットを'-'に (例: Bb -> B-)
    - '△'や'M'を'maj'に (ただし M7 は maj7)
    - 'ø'や'Φ'を'm7b5'に (ハーフディミニッシュ)
    - 'NC', 'N.C.', 'Rest' などは "Rest" に統一
    - 括弧やカンマは削除
    - ルート音を大文字化
    - スラッシュコードでベース音がルートの短7度下などの場合、コードに '7' を自動付与 (例: C/Bb -> C7/Bb)
    """
    if label is None:
        logger.debug("CoreUtils (sanitize): Label is None, returning 'Rest'.")
        return "Rest"

    original_label_for_log = str(label) # ログ用に元のラベルを保持
    s = str(label).strip()

    if not s:
        logger.debug(f"CoreUtils (sanitize): Label '{original_label_for_log}' is empty after strip, returning 'Rest'.")
        return "Rest"

    # 1. 全角を半角に変換
    s = s.translate(str.maketrans(
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ＃♭＋－／．（）０１２３４５６７８9',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#b+-/.()0123456789'
    ))

    # 2. No Chord / Rest 系の早期判定と統一 (大文字・小文字を区別しない)
    if s.upper() in ["NC", "N.C.", "NOCHORD", "NO CHORD", "SILENCE", "-", "REST", "R"]:
        logger.debug(f"CoreUtils (sanitize): Label '{original_label_for_log}' identified as Rest by common alias.")
        return "Rest"

    # 3. 一般的な記号の置換 (music21フレンドリーな形式へ)
    #    フラット記号の置換はルート音とベース音に対して個別に行うため、ここでは広範囲には行わない。
    #    B- のような表記はmusic21が解釈できる。

    # 4. メジャーセブンス、マイナー、ディミニッシュ、オーグメント等のエイリアス変換
    s = re.sub(r"[△Δ]", "maj7", s)  # △ や Δ を maj7 に
    s = s.replace("Ma7", "maj7").replace("Maj7", "maj7").replace("MA7", "maj7")
    s = s.replace("M7", "maj7") # M7 も maj7 に

    # 単独の 'M' (メジャーを示す) は、'maj' に変換するか music21 の解釈に任せるか。
    # C, Cm, CM7 (Cmaj7) があるため、CM (Cmaj) は Cmaj と書くのが明確。
    # ただし、Am の 'm' と区別するため、正規表現で大文字Mのみを対象とする。
    # 例: CM -> Cmaj, FM -> Fmaj (ただしAmはそのまま)
    # s = re.sub(r'(?<![a-zA-Z0-9])M(?![0-9a-z])', 'maj', s) # M単独 -> maj (前後が文字や数字でない)
    # より安全には、ルート音抽出後に品質部分だけ見て置換。
    # 現状では M7 -> maj7 の変換があるので、M 単独は music21 のデフォルト解釈に任せるのも手。
    # C, F だけでもメジャーと解釈される。

    s = s.replace("min7", "m7").replace("-7", "m7") # マイナーセブンス
    # s = s.replace("min", "m") # マイナー (Cm のような形を推奨)

    s = s.replace("dim7", "dim") # dim7 は dim と同じことが多い
    s = s.replace("°7", "dim")
    s = s.replace("ø", "m7b5").replace("Φ", "m7b5").replace("Ø", "m7b5") # ハーフディミニッシュ

    s = s.replace("aug", "+") # オーグメント

    # 5. テンションノートの括弧除去 (例: C7(b9) -> C7b9)
    s = s.replace("(", "").replace(")", "")

    # 6. スラッシュコードの処理 と 短7度ベースの自動7th付与
    processed_slash_code = False
    if '/' in s:
        parts = s.split('/', 1) # 最初のスラッシュで分割
        if len(parts) == 2:
            chord_part_original = parts[0].strip()
            bass_part_original = parts[1].strip()

            if chord_part_original and bass_part_original:
                # まずコード部分とベース部分のルート音を正規化（大文字化、フラット記号）
                temp_chord_root_info = _extract_and_normalize_root(chord_part_original)
                temp_bass_root_info = _extract_and_normalize_root(bass_part_original)

                if temp_chord_root_info and temp_bass_root_info:
                    normalized_chord_part = temp_chord_root_info['normalized_figure']
                    normalized_bass_part = temp_bass_root_info['normalized_figure'] # ベースは通常ルート音のみ

                    try:
                        # music21で解釈してインターバルを計算
                        cs_base_for_interval = harmony.ChordSymbol(normalized_chord_part)
                        # ベース音は単独の音として解釈させる (例: "B-", "G#")
                        # "Gmaj7"のようなベース指定は通常ありえないので、ルート音のみ取り出す
                        p_bass_for_interval = pitch.Pitch(temp_bass_root_info['root_name_normalized'])

                        if cs_base_for_interval.root() and p_bass_for_interval:
                            iv = interval.Interval(noteStart=cs_base_for_interval.root(), noteEnd=p_bass_for_interval)
                            semitones_from_root = iv.semitones % 12 # オクターブ差を無視

                            # ベース音がルートの短7度下(長2度上と同じ、10半音上) or 長2度下(短7度上と同じ、2半音上)
                            # music21 の interval は noteStart から noteEnd への方向。
                            # C に対する Bb は、C.transpose(-2) なので semitones = -2 (または +10)
                            if semitones_from_root == 10 or semitones_from_root == 2: # 10 (C->Bb), 2 (C->D)
                                # 正規化されたコード部分に '7' がまだ含まれていなければ付与
                                # (maj7, m7, dim7 などを考慮し、単純な '7' の存在だけでなく、
                                #  既にセブンスコードの種類が明示されているか確認)
                                if not re.search(r'(maj7|m7|dim7|dom7|7)', normalized_chord_part, re.IGNORECASE) and \
                                   not re.search(r'7', normalized_chord_part): # 最後の '7' はドミナントセブンス用
                                    # 既存の品質の直後、またはルートの直後に '7' を挿入
                                    # 例: C -> C7, Cm -> Cm7, Cmaj -> Cmaj7 ではなく C7
                                    #     Caug -> Caug7
                                    # ここでは単純に末尾に追加するが、より洗練されたロジックも可能
                                    # 7thが付くことでドミナントになることが多いので、
                                    # Cm/Bb -> Cm7/Bb のように、元の品質を保ちつつ7thを付加したい
                                    
                                    # ルート音と品質部分を分離する試み
                                    match_root_quality = re.match(r"([A-G][-#bxs]*)(\S*)", normalized_chord_part)
                                    if match_root_quality:
                                        root_only = match_root_quality.group(1)
                                        quality_only = match_root_quality.group(2)
                                        if 'sus' in quality_only: # Csus/Bb -> C7sus/Bb
                                            normalized_chord_part = root_only + "7" + quality_only
                                        elif quality_only and not re.search(r'7', quality_only):
                                             # Cm/Bb -> Cm7/Bb, Cmaj/Bb -> Cmaj7/Bb (ただしMはmaj7に変換済みが多い)
                                             # C+/Bb -> C+7/Bb
                                             if quality_only.endswith(('m', 'maj', '+', 'dim', 'sus')):
                                                normalized_chord_part = root_only + quality_only + '7'
                                             else: # C -> C7
                                                normalized_chord_part = root_only + '7' + quality_only
                                        elif not quality_only : # ルート音のみの場合 C/Bb -> C7/Bb
                                            normalized_chord_part += '7'
                                        # else: 既に7が含まれる場合は何もしない

                                    logger.info(f"CoreUtils (sanitize): Auto-added '7' to chord part for slash bass '{original_label_for_log}'. New chord part: '{normalized_chord_part}'")
                        
                        s = f"{normalized_chord_part}/{normalized_bass_part}"
                        processed_slash_code = True
                    except Exception as e_interval:
                        logger.warning(f"CoreUtils (sanitize): Error processing slash chord '{original_label_for_log}' for 7th auto-add: {e_interval}. Using original parts.")
                        # エラー時は元の正規化されたパーツで再構成 (7th付与なし)
                        s = f"{temp_chord_root_info['normalized_figure']}/{temp_bass_root_info['normalized_figure']}"
                        processed_slash_code = True # 処理は試みた
                else: # ルート抽出失敗
                    logger.warning(f"CoreUtils (sanitize): Could not extract root from chord part or bass part of '{original_label_for_log}'. Skipping 7th auto-add.")
                    # フォールバック: 単純に空白除去と大文字化
                    s = f"{chord_part_original.strip().capitalize()}/{bass_part_original.strip().capitalize()}" # 極簡易的
                    processed_slash_code = True
        
    # 7. スラッシュコードでない場合、またはスラッシュコード処理でsが更新されなかった場合、全体のルート音を正規化
    if not processed_slash_code:
        root_info = _extract_and_normalize_root(s)
        if root_info:
            s = root_info['normalized_figure']
        else: # ルート抽出失敗の場合、これ以上の正規化は困難
            logger.warning(f"CoreUtils (sanitize): Could not extract root from '{s}' (original: '{original_label_for_log}').")
            # この時点で music21 がパースできなければ None になる可能性が高い

    # 8. 不要な文字の最終クリーニング (例: majmaj7 -> maj7)
    s = s.replace("majmaj7", "maj7").replace("majmaj", "maj")
    s = " ".join(s.split()) # 連続する空白を一つに (ほぼ不要なはずだが念のため)

    if not s: # サニタイズの結果、空文字列になった場合もRest
        logger.debug(f"CoreUtils (sanitize): Sanitized label for '{original_label_for_log}' became empty. Returning 'Rest'.")
        return "Rest"

    # 9. 最終チェック: music21でパース試行
    try:
        cs_test = harmony.ChordSymbol(s)
        # music21が解釈できても、ルート音が取れない場合がある (例: "major" だけなど)
        # また、ChordSymbolがエラーを出さなくても、意図しない解釈をしている場合もある
        if cs_test.root():
            final_sanitized_label = cs_test.figure # music21 が解釈した標準的な表記を取得
            # ルート音のフラットが 'b' に戻っている場合があるので '-' に再統一
            final_sanitized_label = final_sanitized_label.replace(f"{cs_test.root().name[0]}b", f"{cs_test.root().name[0]}-") if 'b' in cs_test.root().name else final_sanitized_label
            if cs_test.bass() and 'b' in cs_test.bass().name:
                 final_sanitized_label = final_sanitized_label.replace(f"{cs_test.bass().name[0]}b", f"{cs_test.bass().name[0]}-", 1) # ベース音のフラットも


            logger.info(f"CoreUtils (sanitize): Original='{original_label_for_log}' -> SanitizedTo='{s}' -> FinalMusic21Figure='{final_sanitized_label}'. Root: {cs_test.root().nameWithOctave if cs_test.root() else 'N/A'}, Bass: {cs_test.bass().nameWithOctave if cs_test.bass() else 'N/A'}")
            # music21 の figure は Bb を B- ではなく Bb と表示することがあるので、
            # s (こちらで B- に変換済み) を返す方が一貫性がある場合も。
            # ただし、music21 が解釈した結果が最も信頼できる形であることも多い。
            # ここでは、自前で正規化した s を返す。
            return s
        else:
            logger.warning(f"CoreUtils (sanitize): Sanitized form '{s}' (from '{original_label_for_log}') parsed by music21 but NO ROOT. Treating as invalid. Returning None.")
            return None
    except Exception as e_parse:
        logger.warning(f"CoreUtils (sanitize): Final form '{s}' (from '{original_label_for_log}') FAILED music21 parsing ({type(e_parse).__name__}: {e_parse}). Returning None.")
        return None

def _extract_and_normalize_root(figure_str: str) -> Optional[Dict[str, str]]:
    """
    コード表記文字列からルート音を抽出し、大文字化とフラット記号('-')への統一を行う。
    Returns:
        A dictionary {'root_name_original': str, 'root_name_normalized': str, 'normalized_figure': str} or None
    """
    if not figure_str:
        return None

    # 簡易的なルート音候補の抽出 (A-G とそれに続く #, b, -, x, s)
    # music21 の Pitch オブジェクトを使うのが最も確実
    try:
        # 一旦 ChordSymbol に通してルート音を取得しようと試みる
        # これが最も堅牢だが、完全でない入力ではエラーになる
        temp_cs = harmony.ChordSymbol(figure_str)
        root_pitch_obj = temp_cs.root()
        if not root_pitch_obj: # "maj7" のような入力ではルートが取れない
            raise ValueError("No root found by ChordSymbol")
            
        original_root_name = root_pitch_obj.name # C, B-, F# など
        # music21のnameは C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B (シャープは F#, C#, G#, D#, A#)
        # これを C, C#, D, D-, E, F, F#, G, G#, A, A-, B に近づける

        normalized_root_name = original_root_name.replace('b', '-') # Db -> D-, Gb -> G- etc.
        
        # figure全体で、元のルート表記を正規化されたルート表記に置換する
        # (ただし、大文字・小文字を区別して置換する必要がある)
        # 例: dbmaj7 -> D-maj7, gbm7b5 -> G-m7b5
        # ルート音が figure の先頭にあると仮定する単純な置換
        normalized_figure = figure_str # 初期値
        if figure_str.lower().startswith(original_root_name.lower()):
            prefix_len = len(original_root_name)
            normalized_figure = normalized_root_name.capitalize() + figure_str[prefix_len:]
        else: # ルートが先頭にない複雑なケースや、元の名前と一致しない場合は、より高度な解析が必要
              # ここでは、figure全体を大文字化する程度に留めるか、あるいはtemp_cs.figureを使う
            normalized_figure = temp_cs.figure # music21が解釈した形
            # 再度、ルート音のフラットを '-' に統一
            if 'b' in normalized_figure: # Bb -> B-
                match_b_root = re.match(r"([A-G])b", normalized_figure)
                if match_b_root:
                    normalized_figure = match_b_root.group(1) + "-" + normalized_figure[len(match_b_root.group(0)):]
            normalized_figure = normalized_figure[0].upper() + normalized_figure[1:]


        return {
            "root_name_original": original_root_name,
            "root_name_normalized": normalized_root_name.capitalize(), # C, D-, F#
            "normalized_figure": normalized_figure
        }

    except Exception as e_root_extract:
        # music21.harmony.ChordSymbol が失敗した場合のフォールバック
        # (より単純な正規表現ベースのルート抽出)
        logger.debug(f"CoreUtils (_extract_root): ChordSymbol failed for '{figure_str}' ({e_root_extract}). Using regex fallback.")
        
        # A-G の後に #, b, bb, x, ##, - が続くパターン
        # (大文字・小文字を区別しないが、出力は常に大文字ルート)
        match = re.match(r"([a-gA-G])([#b\-xs]{0,2})(.*)", figure_str)
        if match:
            original_root_char = match.group(1)
            original_accidental = match.group(2) if match.group(2) else ""
            quality_and_tension = match.group(3) if match.group(3) else ""

            normalized_root_char = original_root_char.upper()
            
            # フラット 'b' を '-' に統一
            normalized_accidental = original_accidental.replace('b', '-')
            # music21 は 's' (sharp) や 'x' (double sharp) を直接は好まない場合があるので注意
            # ここでは入力されたものを尊重しつつ、フラットのみ '-' に
            
            normalized_root_name = normalized_root_char + normalized_accidental
            normalized_figure = normalized_root_name + quality_and_tension
            
            return {
                "root_name_original": original_root_char + original_accidental,
                "root_name_normalized": normalized_root_name,
                "normalized_figure": normalized_figure
            }
        else:
            logger.warning(f"CoreUtils (_extract_root): Regex fallback also failed to extract root from '{figure_str}'.")
            return None

# Example Usage (for testing within this file if run directly)
if __name__ == '__main__':
    test_labels = [
        "C/Bb", "Cm/Bb", "Cmaj7/Bb", "Caug/G#", "c/g", "Am/G", "Dm7/G", "G7sus/C",
        "NC", "N.C.", None, "", "  ", "C", "c", "CM", "Cm", "C#m7", "Dbmaj7",
        "Ｆ♯dim7", "Ｂ♭7(b9,#11)", "A♭M7", "BM7", "Ebm", "Ａ△７", "Ｇø",
        "Cmaj7", "Cmin7", "Cdim", "Caug", "C7sus4", "C(add9)",
        "C/G", "Cmaj7/G", "Am/C", "G/F#", "G/F", # G/F は G7/F になるべき
        "c/b-", "c/bb", # c/bb (C/Bb) -> C7/Bb
        "Cdim/A", # AはCdimの構成音ではないが、Aが短7度下なら... (これは複雑)
        "nonsense", "maj7only"
    ]
    print("Sanitization Test Results:")
    print("--------------------------")
    for label in test_labels:
        sanitized = sanitize_chord_label(label)
        print(f"Original: '{label}'\t -> Sanitized: '{sanitized}'")
        if sanitized and sanitized != "Rest":
            try:
                cs = harmony.ChordSymbol(sanitized)
                print(f"\tMusic21 Parsed: Root='{cs.root().name if cs.root() else 'N/A'}', Bass='{cs.bass().name if cs.bass() else 'N/A'}', Pitches='{[p.name for p in cs.pitches]}'")
            except Exception as e:
                print(f"\tError parsing sanitized '{sanitized}' with music21: {e}")
    print("--------------------------")