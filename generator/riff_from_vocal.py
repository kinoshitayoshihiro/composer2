from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import math
import io
import pretty_midi

try:
    from music21 import chord as m21chord, pitch as m21pitch
except Exception:
    m21chord = None
    m21pitch = None

Grid = float  # beats
ChordSeq = List[Tuple[float, str]]  # (bar_start_beat, chord_symbol)

@dataclass
class Dials:
    intensity: float = 0.5  # 0..1
    drive: float = 0.5      # 0..1 （0=8分寄り / 1=16分寄り）
    warmth: float = 0.5     # 0..1 （柔らかさ・add9系の比率）

@dataclass
class RiffFromVocalConfig:
    genre: str = "ballad"        # "ballad" or "rock"
    grid: Grid = 0.5             # 初期グリッド（dials.drive で上書き）
    register: Tuple[int, int] = (40, 76)  # guitar-ish
    harmony: List[str] | None = None      # ["power5"] / ["triad","add9"] etc.
    avoid_overlap: bool = True   # ボーカルが鳴ってる瞬間は避ける
    base_velocity: int = 84
    note_beats: float = 0.5      # 1発の長さ(拍)
    bars: int = 8                # 生成長（bar単位）
    dials: Dials = field(default_factory=Dials)

    def apply_dials(self) -> None:
        """dialsからグリッド・和声・ベロシティ・音価を上書き"""
        # drive → グリッド（8分 or 16分）
        self.grid = 0.25 if self.dials.drive >= 0.45 or self.genre == "rock" else 0.5
        # intensity → ベロシティ・音価（強く・短く）
        self.base_velocity = int(70 + 50 * max(0.0, min(1.0, self.dials.intensity)))
        self.note_beats = max(0.25, 0.6 - 0.25 * self.dials.intensity)
        # warmth → 和声バイアス
        if self.harmony is None:
            if self.genre == "ballad":
                # warmth高め → add9/sus2 多め
                self.harmony = ["triad", "add9"] if self.dials.warmth >= 0.4 else ["triad", "sus2"]
            else:
                # rock → power中心、warmth高めなら triad を少し混ぜる
                self.harmony = ["power5", "octave"] if self.dials.warmth < 0.6 else ["power5", "octave", "triad"]

def _load_pm_from_any(v: str | Path | bytes | io.BytesIO | pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    if isinstance(v, pretty_midi.PrettyMIDI):
        return v
    if isinstance(v, (bytes, io.BytesIO)):
        b = v if isinstance(v, bytes) else v.getvalue()
        buf = io.BytesIO(b)
        return pretty_midi.PrettyMIDI(buf)
    return pretty_midi.PrettyMIDI(str(v))

def _vocal_active_mask(vocal_pm: pretty_midi.PrettyMIDI, total_beats: float, sec_per_beat: float, grid: Grid) -> List[bool]:
    steps = int(round(total_beats / grid))
    mask = [False] * steps
    mel = None
    for inst in vocal_pm.instruments:
        if not inst.is_drum and inst.notes:
            mel = inst; break
    if mel is None:
        return mask
    for n in mel.notes:
        s_b = n.start / sec_per_beat
        e_b = n.end / sec_per_beat
        s_idx = max(0, int(s_b / grid))
        e_idx = min(steps, int(math.ceil(e_b / grid)))
        for i in range(s_idx, e_idx):
            mask[i] = True
    return mask

def _chord_root_midi(symbol: str, instrument: str = "guitar") -> int:
    if m21pitch is not None:
        try:
            root = m21chord.ChordSymbol(symbol).root()
            return int(root.midi)  # type: ignore[attr-defined]
        except Exception:
            pass
    pcs = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
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

def _is_minor(symbol: str) -> bool:
    cs = symbol.lower().replace(" ", "")
    return ("m" in cs and not cs.startswith("maj")) or "min" in cs

def _clamp_reg(midi: int, lo: int, hi: int) -> int:
    while midi < lo: midi += 12
    while midi > hi: midi -= 12
    return midi

def _harmony_to_pitches(root_midi: int, tags: List[str], minor_like: bool) -> List[int]:
    out: List[int] = []
    for tag in tags:
        if tag == "power5": out += [root_midi, root_midi+7]
        elif tag == "octave": out += [root_midi, root_midi+12]
        elif tag == "triad":
            third = 3 if minor_like else 4
            out += [root_midi, root_midi+third, root_midi+7]
        elif tag == "add9":
            third = 3 if minor_like else 4
            out += [root_midi, root_midi+third, root_midi+7, root_midi+14]
        elif tag == "sus2": out += [root_midi, root_midi+2, root_midi+7]
        elif tag == "root": out.append(root_midi)
        elif tag == "fifth": out.append(root_midi+7)
        else: out.append(root_midi)
    # order-preserving dedup
    seen=set(); uniq=[]
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def generate_riff_from_vocal(
    vocal: str | Path | bytes | io.BytesIO | pretty_midi.PrettyMIDI,
    chord_seq: ChordSeq,
    tempo: float,
    *,
    genre: str = "ballad",
    bars: int = 8,
    dials: Dict[str, float] | None = None,
    humanize: bool | Dict | None = True,
    humanize_profile: Optional[str] = None,
    quantize: Optional[Dict] = None,
    groove: Optional[Dict] = None,
    late_humanize_ms: int = 0,
) -> pretty_midi.PrettyMIDI:
    """
    既存ボーカル（MIDI）から、ボーカルの“隙間”を優先してリフを生成。
    dials（intensity/drive/warmth）を反映して、グリッド・和声・ベロ・音価を調整。
    """
    dial_obj = Dials(**dials) if dials else Dials()
    cfg = RiffFromVocalConfig(genre=genre, bars=bars, dials=dial_obj)
    cfg.apply_dials()

    pm_v = _load_pm_from_any(vocal)
    sec_per_beat = 60.0 / float(tempo or 120.0)
    total_beats = (chord_seq[-1][0] + 4.0) if chord_seq else cfg.bars * 4.0
    steps = int(round(total_beats / cfg.grid))

    mask = _vocal_active_mask(pm_v, total_beats, sec_per_beat, cfg.grid) if cfg.avoid_overlap else [False]*steps

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo or 120.0))
    inst = pretty_midi.Instrument(program=29, name="RiffFromVocal:guitar")  # Overdriven
    lo, hi = cfg.register

    # barごとにコードを解決
    chord_by_bar = []
    for b in range(int(math.ceil(total_beats / 4.0))):
        chord_by_bar.append(chord_seq[b % len(chord_seq)][1] if chord_seq else "Am")

    for i in range(steps):
        if mask[i]:
            continue
        beat = i * cfg.grid
        bar_idx = int(beat // 4.0)
        # バラードは拍頭寄りに制限して情報量を抑える
        if genre == "ballad" and (beat % 1.0) != 0.0:
            continue
        symbol = chord_by_bar[min(bar_idx, len(chord_by_bar)-1)] if chord_by_bar else "Am"
        root = _chord_root_midi(symbol, instrument="guitar")
        minor_like = _is_minor(symbol)
        for p in _harmony_to_pitches(root, cfg.harmony or ["power5"], minor_like):
            p = _clamp_reg(p, lo, hi)
            start = beat * sec_per_beat
            end = (beat + cfg.note_beats) * sec_per_beat
            inst.notes.append(pretty_midi.Note(velocity=cfg.base_velocity, pitch=p, start=start, end=end))

    if humanize:
        from utilities.humanize_bridge import apply_humanize_to_instrument

        profile = humanize_profile or ("rock_tight" if genre == "rock" else "ballad_subtle")
        overrides = (humanize if isinstance(humanize, dict) else None) or {}
        quant = quantize or {
            "grid": 0.25,
            "swing": 0.0 if genre == "rock" else 0.10,
        }
        groove_spec = groove or {
            "name": "rock_16_loose" if genre == "rock" else "ballad_8_swing"
        }
        apply_humanize_to_instrument(
            inst,
            tempo,
            profile=profile,
            overrides=overrides,
            quantize=quant,
            groove=groove_spec,
            late_humanize_ms=late_humanize_ms,
        )

    pm.instruments.append(inst)
    return pm
