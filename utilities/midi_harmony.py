from __future__ import annotations
from typing import List, Tuple, Optional, IO, Union

# 依存は music21 があればベスト。無ければ控えめにフォールバック。
try:
    from music21 import converter, chord, key as m21key, roman as m21roman, stream
except Exception:
    converter = chord = m21key = m21roman = stream = None  # type: ignore


def extract_progression_from_midi(
    midi_data: Union[str, bytes, IO[bytes]],
    *,
    key_hint: Optional[str] = None,
    beats_per_bar: float = 4.0,
    max_bars: Optional[int] = None,
) -> List[Tuple[float, str]]:
    """
    MIDI(ファイルパス/バイナリ/バッファ)からシンプルなコード進行を推定し、
    [(bar_start_beat, "Am"), ...] 形式で返す。
    - 1小節に1コード（多数決/和音重心）を基本とする。
    - music21 がなければ空配列を返す（UI側でメッセージ表示）。
    """
    if converter is None:
        return []
    try:
        score = converter.parse(midi_data)
    except Exception:
        return []

    # キー推定（ヒントがあれば優先）
    if key_hint:
        try:
            k = m21key.Key(key_hint)
        except Exception:
            k = score.analyze('key')
    else:
        try:
            k = score.analyze('key')
        except Exception:
            k = m21key.Key('C')

    # chordify で和音化 → 1小節ずつ代表コードを採用
    ch = score.chordify()
    # 4/4想定で小節境界を使う（異拍子は要改善だが最小版として）
    bars: List[Tuple[float, str]] = []
    if not hasattr(ch, 'measure'):
        return []

    mno = 1
    bar_start_beat = 0.0
    taken = 0
    while True:
        meas = ch.measure(mno)
        if meas is None:
            break
        # 小節内の最も長く鳴っている和音 or 最初の和音を採用
        chosen = None
        max_dur = 0.0
        for e in meas.recurse().notesAndRests:
            if getattr(e, 'isChord', False):
                dur = float(e.duration.quarterLength or 0.0)
                if dur > max_dur:
                    max_dur = dur
                    chosen = e
        if chosen is None:
            # 休符だけの小節→トニック置き
            sym = k.tonic.name
            if k.mode == 'minor':
                sym += 'm'
        else:
            # roman → chordSymbol に変換して記述を安定させる
            try:
                rn = m21roman.romanNumeralFromChord(chosen, k)
                sym = rn.figure  # 例: i, V, IV6 等
                # 実用のため triad 化し簡素表示へ（例: i → Am, V → E）
                triad = rn.root().name
                if rn.quality == 'minor':
                    sym = f'{triad}m'
                elif rn.quality in ('major', 'dominant'):
                    sym = triad
                else:
                    # dim/aug/sus等は簡素化（最小実装）
                    sym = triad
            except Exception:
                # 失敗時は音高の根音名から推定
                p = chosen.root()
                sym = p.name
                if k.mode == 'minor' and sym.lower().startswith(k.tonic.name.lower()):
                    sym += 'm'

        bars.append((bar_start_beat, sym))
        bar_start_beat += beats_per_bar
        mno += 1
        taken += 1
        if max_bars and taken >= max_bars:
            break

    return bars
