from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import pretty_midi

try:
    from music21 import chord as m21chord, pitch as m21pitch
except Exception:
    m21chord = None
    m21pitch = None

Grid = float  # beats
ChordSeq = List[Tuple[float, str]]  # (bar_start_beat, chord_symbol)

@dataclass
class RiffFromVocalConfig:
    genre: str = "ballad"        # "ballad" or "rock"
    grid: Grid = 0.5             # 0.5=8th, 0.25=16th
    register: Tuple[int, int] = (40, 76)  # guitar-ish
    harmony: Optional[List[str]] = None    # ["power5"] or ["triad","add9"] etc.
    avoid_overlap: bool = True   # vocalが鳴ってる瞬間は避ける
    base_velocity: int = 84
    note_beats: float = 0.5      # 1発の長さ(拍)
    bars: int = 8                # 生成長（bar単位）

    def __post_init__(self):
        if self.harmony is None:
            self.harmony = ["triad","add9"] if self.genre == "ballad" else ["power5","octave"]
        if self.genre == "rock" and self.grid > 0.25:
            self.grid = 0.25

def load_vocal_pm(vocal: str | Path | pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    if isinstance(vocal, pretty_midi.PrettyMIDI):
        return vocal
    return pretty_midi.PrettyMIDI(str(vocal))

def vocal_active_mask(vocal_pm: pretty_midi.PrettyMIDI, total_beats: float, sec_per_beat: float, grid: Grid) -> List[bool]:
    """grid刻みでボーカルが鳴っているかどうかのブール配列を返す"""
    steps = int(round(total_beats / grid))
    mask = [False] * steps
    # 最初のモノフォニックトラックを使う（なければ全無効）
    mel = None
    for inst in vocal_pm.instruments:
        if not inst.is_drum:
            mel = inst
            break
    if mel is None:
        return mask
    for n in mel.notes:
        start_b = n.start / sec_per_beat
        end_b = n.end / sec_per_beat
        s_idx = max(0, int(start_b / grid))
        e_idx = min(steps, int(end_b / grid) + 1)
        for i in range(s_idx, e_idx):
            mask[i] = True
    return mask

def chord_root_midi(symbol: str, instrument: str = "guitar") -> int:
    if m21pitch is not None:
        try:
            root = m21chord.ChordSymbol(symbol).root()
            return int(root.midi)  # type: ignore[attr-defined]
        except Exception:
            pass
    pcs = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    name = symbol.strip()
    semi = 0
    if name:
        base = pcs.get(name[0].upper(), 0)
        rest = name[1:]
        if rest.startswith("#"): semi = 1
        elif rest.startswith("b"): semi = -1
        pc = (base + semi) % 12
    else:
        pc = 0
    base_oct = 3 if instrument != "bass" else 2
    return 12 * base_oct + pc

def harmony_to_pitches(root_midi: int, tags: List[str], minor_like: bool) -> List[int]:
    out: List[int] = []
    for tag in tags:
        if tag == "power5":
            out.extend([root_midi, root_midi + 7])
        elif tag == "octave":
            out.extend([root_midi, root_midi + 12])
        elif tag == "triad":
            third = 3 if minor_like else 4
            out.extend([root_midi, root_midi + third, root_midi + 7])
        elif tag == "add9":
            third = 3 if minor_like else 4
            out.extend([root_midi, root_midi + third, root_midi + 7, root_midi + 14])
        elif tag == "sus2":
            out.extend([root_midi, root_midi + 2, root_midi + 7])
        elif tag == "root":
            out.append(root_midi)
        elif tag == "fifth":
            out.append(root_midi + 7)
        else:
            out.append(root_midi)
    # order-preserving dedup
    seen=set(); uniq=[]
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def is_minor_symbol(symbol: str) -> bool:
    cs = symbol.lower().replace(" ", "")
    return ("m" in cs and not cs.startswith("maj")) or "min" in cs

def clamp_register(midi: int, lo: int, hi: int) -> int:
    while midi < lo: midi += 12
    while midi > hi: midi -= 12
    return midi

def generate_riff_from_vocal(
    vocal: str | Path | pretty_midi.PrettyMIDI,
    chord_seq: ChordSeq,
    tempo: float,
    config: Optional[RiffFromVocalConfig] = None,
    instrument_name: str = "Riff:guitar",
) -> pretty_midi.PrettyMIDI:
    """ボーカルMIDIの“鳴っていない隙間”を優先して、ジャンル基調のグリッドに沿うリフを生成"""
    cfg = config or RiffFromVocalConfig()
    pm_v = load_vocal_pm(vocal)
    sec_per_beat = 60.0 / float(tempo or 120.0)
    total_beats = (chord_seq[-1][0] + 4.0) if chord_seq else cfg.bars * 4.0

    mask = vocal_active_mask(pm_v, total_beats, sec_per_beat, cfg.grid) if cfg.avoid_overlap else None
    steps = int(round(total_beats / cfg.grid))

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo or 120.0))
    inst = pretty_midi.Instrument(program=29, name=instrument_name)  # Overdriven Guitar
    lo, hi = cfg.register

    # barごとに対応するコードを引き当て
    chord_by_bar = []
    for b in range(int(total_beats // 4)):
        chord_by_bar.append(chord_seq[b % len(chord_seq)][1] if chord_seq else "Am")

    for i in range(steps):
        beat = i * cfg.grid
        bar_idx = int(beat // 4.0)
        if cfg.avoid_overlap and mask and mask[i]:
            continue  # ボーカルが鳴ってるグリッドは避ける
        # できるだけ強拍優先（8分なら 0,1,2,3 拍頭／16分なら 0,0.5,1,1.5 ... のうち整数拍を優先）
        if (beat % 1.0) != 0.0 and cfg.genre == "ballad":
            # バラードは“拍頭重視”にして情報量を抑える
            continue

        symbol = chord_by_bar[min(bar_idx, len(chord_by_bar)-1)] if chord_by_bar else "Am"
        root = chord_root_midi(symbol, instrument="guitar")
        minor_like = is_minor_symbol(symbol)

        for p in harmony_to_pitches(root, cfg.harmony, minor_like):
            p = clamp_register(p, lo, hi)
            start = beat * sec_per_beat
            end = (beat + cfg.note_beats) * sec_per_beat
            inst.notes.append(pretty_midi.Note(velocity=cfg.base_velocity, pitch=p, start=start, end=end))

    pm.instruments.append(inst)
    return pm
