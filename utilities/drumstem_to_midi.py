#!/usr/bin/env python3
"""
drumstem_to_midi.py — Convert a drum audio stem (WAV/AIFF/MP3) into MIDI with timing + velocity.
- 依存: numpy, scipy.signal, pydub, pretty_midi（composer2 既存セット前提）
- 方針:
  1) オンセット検出: マルチフレームRMS + スペクトルフラックス（簡易）
  2) ヒットごとの帯域エネルギー比で KICK / SNARE / HIHAT を簡易分類
  3) 音量・局所RMS・ピーク比から velocity を 1..127 にマッピング
  4) ヒューマナイズは最小限（±数msのジッタ&弱いランダム化）。本番は別工程でGrooveSampler等に接続

CLI:
    python -m utilities.drumstem_to_midi IN.wav OUT.mid \
        --bpm 120 --sr 44100 --tight 0.5 --gate -36 \
        --kick C1 --snare D1 --hihat F#1 \
        --min-sep-ms 30 --humanize-ms 3

Note:
- 分類とvelocity推定はヒューリスティック。高精度化はML分類器に差し替え可能（同一I/FでOK）。
- ステレオはモノラル合成（Mid= (L+R)/2）で処理。
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.signal import find_peaks, get_window
from pydub import AudioSegment
import pretty_midi

# ---------------------------
# Utils
# ---------------------------

NOTE_NAME_TO_MIDI = {
    # C1=36, GMドラム準拠の典型域（柔軟に差し替え可）
    "C1": 36,
    "C#1": 37,
    "Db1": 37,
    "D1": 38,
    "D#1": 39,
    "Eb1": 39,
    "E1": 40,
    "F1": 41,
    "F#1": 42,
    "Gb1": 42,
    "G1": 43,
    "G#1": 44,
    "Ab1": 44,
    "A1": 45,
    "A#1": 46,
    "Bb1": 46,
    "B1": 47,
    "C2": 48,
    "C#2": 49,
    "D2": 50,
    "D#2": 51,
    "E2": 52,
    "F#2": 54,
    "G#2": 56,
    "A#2": 58,
}


def note_to_midi(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        if s in NOTE_NAME_TO_MIDI:
            return NOTE_NAME_TO_MIDI[s]
        raise ValueError(f"Unknown note: {s}")


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(x, eps))


@dataclass
class BandConfig:
    low: float
    high: float
    weight: float


# Kick/Snare/HiHat 簡易帯域（Hz）
BANDS = {
    "kick": [BandConfig(20, 80, 1.0), BandConfig(80, 200, 0.7)],
    "snare": [BandConfig(150, 400, 1.0), BandConfig(1_500, 4_000, 0.8)],
    "hihat": [BandConfig(5_000, 12_000, 1.0), BandConfig(10_000, 16_000, 0.8)],
}


def stft_mag(
    signal: np.ndarray, sr: int, win_ms: float = 20.0, hop_ms: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    win_len = int(sr * win_ms * 1e-3)
    hop = int(sr * hop_ms * 1e-3)
    win = get_window("hann", win_len, fftbins=True)
    n_frames = 1 + (len(signal) - win_len) // hop if len(signal) >= win_len else 0
    mags = []
    for i in range(n_frames):
        s = signal[i * hop : i * hop + win_len] * win
        spec = np.fft.rfft(s)
        mags.append(np.abs(spec))
    mags = np.array(mags)  # [frames, bins]
    freqs = np.fft.rfftfreq(win_len, 1.0 / sr)
    return mags, freqs


def spectral_flux(mags: np.ndarray) -> np.ndarray:
    # 正の差分のみ加算
    flux = np.maximum(mags[1:] - mags[:-1], 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    return flux


def frame_to_time(idx: int, hop_ms: float) -> float:
    return (idx * hop_ms) * 1e-3


def audio_to_mono_np(seg: AudioSegment) -> Tuple[np.ndarray, int]:
    if seg.channels == 2:
        # Mid合成（L+R)/2
        samples = (
            np.array(seg.split_to_mono()[0].get_array_of_samples(), dtype=np.float32)
            + np.array(seg.split_to_mono()[1].get_array_of_samples(), dtype=np.float32)
        ) * 0.5
    else:
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    # 16bit想定正規化
    samples /= 1 << (8 * seg.sample_width - 1)
    return samples, seg.frame_rate


def band_energy(spec_mag: np.ndarray, freqs: np.ndarray, bands: List[BandConfig]) -> float:
    total = 0.0
    for b in bands:
        mask = (freqs >= b.low) & (freqs < b.high)
        if mask.any():
            total += b.weight * float(spec_mag[mask].mean())
    return total


def choose_class(spec_mag: np.ndarray, freqs: np.ndarray) -> str:
    ek = band_energy(spec_mag, freqs, BANDS["kick"])
    es = band_energy(spec_mag, freqs, BANDS["snare"])
    eh = band_energy(spec_mag, freqs, BANDS["hihat"])
    # 距離を単純最大
    if ek >= es and ek >= eh:
        return "kick"
    if es >= ek and es >= eh:
        return "snare"
    return "hihat"


def map_velocity(
    rms_local: float,
    peak_local: float,
    rms_global: float,
    peak_global: float,
    floor_db: float = -36.0,
) -> int:
    # グローバル対比＋ローカルピークを混合：音量差をvelocityにマップ
    # 1) 正規化（対数域の感じに近づける）
    rel_rms = rms_local / (rms_global + 1e-12)
    rel_peak = peak_local / (peak_global + 1e-12)
    raw = 0.6 * rel_rms + 0.4 * rel_peak
    # 2) ダイナミックレンジ下限（ゲート）: floor_db以下は弱打→クリップ
    floor = db_to_lin(floor_db)  # ex: -36dB ≈ 0.0158
    raw = max(raw, floor)
    # 3) 対数圧縮→MIDI 1..127
    logv = math.log(raw / floor) / max(math.log(1.0 / floor), 1e-6)
    v = int(round(1 + 126 * min(max(logv, 0.0), 1.0)))
    return v


# ---------------------------
# Main conversion
# ---------------------------


@dataclass
class Options:
    bpm: float
    sr: int
    gate_db: float
    tight: float
    min_sep_ms: float
    humanize_ms: float
    kick_note: int
    snare_note: int
    hihat_note: int
    win_ms: float
    hop_ms: float


def detect_hits(
    x: np.ndarray, sr: int, win_ms: float, hop_ms: float, gate_db: float, min_sep_ms: float
) -> List[int]:
    mags, _ = stft_mag(x, sr, win_ms=win_ms, hop_ms=hop_ms)
    if mags.shape[0] < 3:
        return []
    flux = spectral_flux(mags)
    # 平滑 & 正規化
    flux = flux / (np.max(flux) + 1e-12)
    # ゲート
    thresh = db_to_lin(gate_db + 60.0)  # 粗い経験則の“フロア上げ”
    cand, _ = find_peaks(flux, height=thresh, distance=max(1, int(min_sep_ms / hop_ms)))
    return cand.tolist()


# --- NEW: per-song monotone rank mapping (no-ML) -----------------
def build_rank_velocity_mapper(values, vmin=1, vmax=127):
    """
    values: np.ndarray of positive scalars (e.g., local RMS or a mix score)
    returns: function f(value)->int (1..127), monotone non-decreasing

    単調性保証のため、順位に応じて線形に割り当て。
    """
    if len(values) == 0:
        return lambda v: 64
    vals = np.asarray(values, dtype=np.float64)
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, num=len(vals), endpoint=True)

    # ルックアップ（最近傍）
    def f(v):
        # rank相当を最近傍探索
        idx = np.searchsorted(vals[order], v, side="left")
        idx = np.clip(idx, 0, len(vals) - 1)
        r = ranks[idx]
        return int(round(vmin + r * (vmax - vmin)))

    return f


def convert(audio_path: str, out_path: str, opt: Options) -> None:
    seg = AudioSegment.from_file(audio_path)
    x, sr = audio_to_mono_np(seg)
    if opt.sr and opt.sr != sr:
        # pydub resample（簡易・品質は十分）：
        seg = seg.set_frame_rate(opt.sr)
        x, sr = audio_to_mono_np(seg)

    # 全体RMS/PEAK
    rms_global = float(np.sqrt(np.mean(x * x)))
    peak_global = float(np.max(np.abs(x)) + 1e-12)

    # オンセットフレーム
    hits = detect_hits(x, sr, opt.win_ms, opt.hop_ms, opt.gate_db, opt.min_sep_ms)
    if not hits:
        pm = pretty_midi.PrettyMIDI()
        pm.write(out_path)
        return

    mags, freqs = stft_mag(x, sr, win_ms=opt.win_ms, hop_ms=opt.hop_ms)

    # MIDI 準備（チャネル10=drum慣習だが pretty_midi は note.program=0 でもOK）
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    # 各ヒット
    for hi in hits:
        # 近傍フレームでローカルRMS/PEAK推定
        t = frame_to_time(hi, opt.hop_ms)
        f0 = max(0, hi - 1)
        f1 = min(mags.shape[0] - 1, hi + 1)
        spec_local = mags[hi]
        cls = choose_class(spec_local, freqs)

        # ローカル波形窓（±20ms）
        w = int(sr * 0.02)
        c = int(t * sr)
        l = max(0, c - w)
        r = min(len(x), c + w)
        seg_x = x[l:r] if r > l else np.array([0.0], dtype=np.float32)

        rms_local = float(np.sqrt(np.mean(seg_x * seg_x)))
        peak_local = float(np.max(np.abs(seg_x)) + 1e-12)
        vel = map_velocity(rms_local, peak_local, rms_global, peak_global, floor_db=opt.gate_db)

        # タイミング微調整（tight<1.0でスライトヒューマナイズ）
        jitter = (np.random.rand() - 0.5) * 2.0 * (opt.humanize_ms * 1e-3) * (1.0 - opt.tight)
        t_hit = max(0.0, t + jitter)
        # ドラムの短い長さ（20〜60ms）
        dur = 0.02 + 0.04 * np.random.rand()

        note_num = (
            opt.hihat_note
            if cls == "hihat"
            else opt.snare_note if cls == "snare" else opt.kick_note
        )
        n = pretty_midi.Note(velocity=vel, pitch=note_num, start=t_hit, end=t_hit + dur)
        inst.notes.append(n)

    pm.instruments.append(inst)
    pm.write(out_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert drum stem audio to MIDI with velocity & class."
    )
    p.add_argument("audio", type=str)
    p.add_argument("out", type=str)
    p.add_argument("--bpm", type=float, default=120.0)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument(
        "--gate", type=float, default=-36.0, dest="gate_db", help="dBFS floor (e.g., -36)"
    )
    p.add_argument("--tight", type=float, default=0.5, help="0..1（1=機械的/0=やや人間味）")
    p.add_argument("--min-sep-ms", type=float, default=30.0, help="最小オンセット間隔")
    p.add_argument("--humanize-ms", type=float, default=3.0, help="タイミング微揺らぎ最大値")
    p.add_argument("--kick", type=str, default="C1")
    p.add_argument("--snare", type=str, default="D1")
    p.add_argument("--hihat", type=str, default="F#1")
    p.add_argument("--win-ms", type=float, default=20.0)
    p.add_argument("--hop-ms", type=float, default=5.0)
    return p


def main():
    ap = build_argparser()
    a = ap.parse_args()
    opt = Options(
        bpm=a.bpm,
        sr=a.sr,
        gate_db=a.gate_db,
        tight=a.tight,
        min_sep_ms=a.min_sep_ms,
        humanize_ms=a.humanize_ms,
        kick_note=note_to_midi(a.kick),
        snare_note=note_to_midi(a.snare),
        hihat_note=note_to_midi(a.hihat),
        win_ms=a.win_ms,
        hop_ms=a.hop_ms,
    )
    convert(a.audio, a.out, opt)


if __name__ == "__main__":
    main()
