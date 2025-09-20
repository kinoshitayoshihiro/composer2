#!/usr/bin/env python3
"""
tempo_from_mix.py — 統合ミックスからテンポカーブ/拍(downbeats含む)を推定し、
任意で複数ステムのオンセット情報で微修正（±数十ms）する。

出力:
- beats: [t0, t1, ...] 秒
- downbeats: [d0, d1, ...] 秒 (4/4仮定、位相は最適化)
- tempo_bpm: 区間平均BPM（参考）
"""

from __future__ import annotations
import argparse, math, json
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import medfilt, find_peaks, get_window
from pydub import AudioSegment


@dataclass
class Opt:
    sr: int = 44100
    win_ms: float = 20.0
    hop_ms: float = 5.0
    min_bpm: float = 60.0
    max_bpm: float = 200.0
    tighten_ms: float = 20.0  # ステムでの微修正許容
    meter: int = 4


def load_mono(path: str, sr: int) -> Tuple[np.ndarray, int]:
    seg = AudioSegment.from_file(path)
    if sr and seg.frame_rate != sr:
        seg = seg.set_frame_rate(sr)
    if seg.channels == 2:
        L, R = seg.split_to_mono()
        x = (
            np.array(L.get_array_of_samples(), dtype=np.float32)
            + np.array(R.get_array_of_samples(), dtype=np.float32)
        ) * 0.5
    else:
        x = np.array(seg.get_array_of_samples(), dtype=np.float32)
    x /= 1 << (8 * seg.sample_width - 1)
    return x, seg.frame_rate


def stft_mag(x: np.ndarray, sr: int, win_ms: float, hop_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    win_len = int(sr * win_ms * 1e-3)
    hop = int(sr * hop_ms * 1e-3)
    n_frames = 1 + max(0, (len(x) - win_len) // hop)
    win = get_window("hann", win_len, fftbins=True)
    mags = []
    for i in range(n_frames):
        s = x[i * hop : i * hop + win_len] * win
        spec = np.fft.rfft(s)
        mags.append(np.abs(spec))
    mags = np.array(mags)  # [T, F]
    return mags, np.fft.rfftfreq(win_len, 1.0 / sr)


def onset_envelope(
    x: np.ndarray, sr: int, win_ms: float, hop_ms: float
) -> Tuple[np.ndarray, float]:
    mags, _ = stft_mag(x, sr, win_ms, hop_ms)
    if len(mags) < 2:
        return np.zeros(0, dtype=np.float32), hop_ms * 1e-3
    # 簡易HPSS的な“パーカッシブ強調”：時間方向に微分→半波整流
    flux = np.maximum(mags[1:] - mags[:-1], 0.0).sum(axis=1)
    # スムージング＆ロバスト化
    flux = flux / (np.max(flux) + 1e-12)
    flux = medfilt(flux, kernel_size=5)
    return flux.astype(np.float32), hop_ms * 1e-3


def estimate_bpm(envelope: np.ndarray, hop_s: float, bpm_min: float, bpm_max: float) -> float:
    if len(envelope) < 4:
        return 120.0
    # 自己相関
    env = envelope - envelope.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1 :]
    # ラグをBPMに対応付け
    lags = np.arange(1, len(ac))
    bpms = 60.0 / (lags * hop_s + 1e-9)
    mask = (bpms >= bpm_min) & (bpms <= bpm_max)
    if not np.any(mask):
        return 120.0
    i = np.argmax(ac[1:][mask])
    bpm = bpms[mask][i]
    return float(np.clip(bpm, bpm_min, bpm_max))


def track_beats(envelope: np.ndarray, hop_s: float, bpm: float) -> List[float]:
    if len(envelope) == 0:
        return []
    # 最強オンセット近傍から開始し、ビート間隔で前後に伸ばす
    peak_idx = int(np.argmax(envelope))
    period = 60.0 / bpm
    beats = [peak_idx * hop_s]
    # forward
    t = beats[0]
    while (t + period) < len(envelope) * hop_s:
        t_next = t + period
        # 近傍±30msで最大オンセットへスナップ
        rad = 0.03
        i0 = max(0, int((t_next - rad) / hop_s))
        i1 = min(len(envelope) - 1, int((t_next + rad) / hop_s))
        j = np.argmax(envelope[i0 : i1 + 1]) + i0
        beats.append(j * hop_s)
        t = beats[-1]
    # backward
    t = beats[0]
    tmp = []
    while (t - period) > 0:
        t_prev = t - period
        rad = 0.03
        i0 = max(0, int((t_prev - rad) / hop_s))
        i1 = min(len(envelope) - 1, int((t_prev + rad) / hop_s))
        j = np.argmax(envelope[i0 : i1 + 1]) + i0
        tmp.append(j * hop_s)
        t = tmp[-1]
    beats = list(reversed(tmp)) + beats
    # 重複除去＆昇順
    out = []
    last = -1.0
    for b in beats:
        if last < 0 or (b - last) > 1e-3:
            out.append(b)
            last = b
    return out


def refine_with_stems(
    beats: List[float], stems_env: List[Tuple[np.ndarray, float]], tighten_ms: float
) -> List[float]:
    if not stems_env or not beats:
        return beats
    max_shift = tighten_ms * 1e-3
    refined = []
    for b in beats:
        # 各ステムの±窓で最大オンセットに寄せる（加重平均ではなく“投票”に近い形）
        votes = []
        for env, hop_s in stems_env:
            i = int(b / hop_s)
            rad = int(max_shift / hop_s)
            i0 = max(0, i - rad)
            i1 = min(len(env) - 1, i + rad)
            if i1 > i0:
                j = np.argmax(env[i0 : i1 + 1]) + i0
                votes.append(j * hop_s)
        if votes:
            # 票の中央値へ（外れ値に強い）
            refined.append(float(np.median(votes)))
        else:
            refined.append(b)
    # 単調増加を保証
    refined = sorted(refined)
    return refined


def choose_downbeat_offset(
    beats: List[float], envelope: np.ndarray, hop_s: float, meter: int
) -> List[float]:
    if not beats:
        return []
    # 4/4仮定：位相0..meter-1を試し、onset強度が最大の位相をdownbeatに採用
    scores = []
    for off in range(meter):
        idxs = list(range(off, len(beats), meter))
        s = 0.0
        for k in idxs:
            t = beats[k]
            i = min(len(envelope) - 1, max(0, int(t / hop_s)))
            s += envelope[i]
        scores.append(s)
    best_off = int(np.argmax(scores))
    return [beats[i] for i in range(best_off, len(beats), meter)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mix", type=str, help="統合ミックスのオーディオファイル")
    ap.add_argument("--stems", type=str, default="", help="カンマ区切りのステムwav（任意）")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--min-bpm", type=float, default=60.0)
    ap.add_argument("--max-bpm", type=float, default=200.0)
    ap.add_argument("--meter", type=int, default=4)
    ap.add_argument("--tighten-ms", type=float, default=20.0)
    ap.add_argument("--out", type=str, default="tempo_map.json")
    args = ap.parse_args()

    opt = Opt(
        sr=args.sr,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        meter=args.meter,
        tighten_ms=args.tighten_ms,
    )

    x, sr = load_mono(args.mix, opt.sr)
    env_mix, hop_s = onset_envelope(x, sr, opt.win_ms, opt.hop_ms)
    bpm = estimate_bpm(env_mix, hop_s, opt.min_bpm, opt.max_bpm)
    beats = track_beats(env_mix, hop_s, bpm)

    stems_env = []
    if args.stems:
        for p in [s.strip() for s in args.stems.split(",") if s.strip()]:
            xs, srs = load_mono(p, opt.sr)
            env, hs = onset_envelope(xs, srs, opt.win_ms, opt.hop_ms)
            stems_env.append((env, hs))
        beats = refine_with_stems(beats, stems_env, opt.tighten_ms)

    downbeats = choose_downbeat_offset(beats, env_mix, hop_s, opt.meter)

    out = {
        "tempo_bpm": round(bpm, 3),
        "beats": beats,
        "downbeats": downbeats,
        "sr": sr,
        "hop_s": hop_s,
        "meter": opt.meter,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
