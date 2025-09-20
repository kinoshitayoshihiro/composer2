from __future__ import annotations
import argparse
import json
import os
import random
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import io

try:
    import yaml
except Exception:  # pragma: no cover - dependency guard
    print("PyYAML is required. Please `pip install pyyaml`.")
    raise SystemExit(0)
import numpy as np
import pretty_midi as pm
import mido
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(x, **kwargs):
        return x

DRUM_CRASH = {49, 57}
DRUM_TOMS = {41, 43, 45, 47}
SNARE = {38, 40}

# デフォルト閾値（必要に応じて JSON で上書き）
DEFAULT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "drums": {
        "detect": {
            "channel10_weight": 1.0,
            "name_weight": 1.0,
            "gm_hits_weight": 2.0,
            "min_hits": 0,
        },
        "section": {
            "fill": 0.15,
            "chorus_density": 80,
            "chorus_crash": 0.2,
            "intro_density": 20,
            "intro_crash": 0.05,
            "bridge_sync": 0.22,
            "bridge_crash": 0.1,
            "outro_density": 18,
            "outro_vel": 60,
        },
        "mood": {
            "energetic_bpm": 130,
            "energetic_density": 90,
            "energetic_vel": 90,
            "relaxed_swing": 0.35,
            "relaxed_bpm_low": 90,
            "relaxed_bpm_high": 120,
            "relaxed_dens_low": 30,
            "relaxed_dens_high": 80,
            "aggressive_sync": 0.3,
            "aggressive_vstd": 25,
            "melancholic_bpm": 85,
            "melancholic_density": 30,
        },
    },
    "poly": {
        "section": {
            "chorus_density": 50,
            "chorus_range": 24,
            "intro_density": 15,
            "bridge_change": 0.3,
            "outro_density": 10,
        },
        "mood": {
            "energetic_bpm": 130,
            "relaxed_sustain": 0.5,
            "aggressive_change": 0.4,
            "melancholic_bpm": 85,
            "melancholic_density": 20,
        },
    },
}


@dataclass
class TagResult:
    """結果保持用データクラス"""
    path: str
    section: str
    mood: str
    intensity: str
    conf_section: float
    conf_mood: float
    section_rule: str
    section_score: float
    mood_rule: str
    mood_score: float
    features: Dict[str, Any]


def _load_thresholds(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    with open(path, "r") as f:
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def write_yaml_sharded(data: Dict[str, Any], out_path: Path, shard_size: Optional[int]) -> List[Tuple[Path, int]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shards: List[Tuple[Path, int]] = []
    if shard_size:
        items = list(data.items())
        for i in range(0, len(items), shard_size):
            shard = i // shard_size
            shard_data = dict(items[i:i + shard_size])
            sp = out_path.with_name(f"{out_path.stem}_{shard:03d}{out_path.suffix}")
            sp.write_text(yaml.safe_dump(shard_data, allow_unicode=True, sort_keys=True))
            shards.append((sp, len(shard_data)))
    else:
        out_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=True))
        shards.append((out_path, len(data)))
    return shards

# ------------------------------
# MIDI 解析ユーティリティ
# ------------------------------

def load_pretty_midi(data: bytes) -> pm.PrettyMIDI | None:
    try:
        return pm.PrettyMIDI(io.BytesIO(data))
    except Exception:
        try:
            _ = mido.MidiFile(file=io.BytesIO(data))
            return pm.PrettyMIDI(io.BytesIO(data))
        except Exception:
            return None


def estimate_bpm(p: pm.PrettyMIDI, fallback: float = 120.0) -> Tuple[float, bool]:
    """BPM を推定。get_beats が不安定な場合はテンポイベントから計算。"""
    try:
        beats = p.get_beats()
        if len(beats) >= 2:
            spb = np.median(np.diff(beats))
            if spb > 1e-6:
                return float(60.0 / spb), False
    except Exception:
        pass
    try:
        _times, tempi = p.get_tempo_changes()
        if len(tempi) > 1:
            return float(np.median(tempi)), False
        if len(tempi) == 1:
            return float(tempi[0]), True
    except Exception:
        pass
    return float(fallback), True


def tempo_mad(p: pm.PrettyMIDI) -> float:
    beats = p.get_beats()
    if len(beats) >= 3:
        spb = np.diff(beats)
        bpm = 60.0 / np.maximum(spb, 1e-9)
        return float(np.median(np.abs(bpm - np.median(bpm))))
    return 0.0


def is_drum_midi(p: pm.PrettyMIDI, path: Path, th: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
    """ドラムMIDI判定 with tunable weights. Returns (is_drum, detail_scores)."""
    score = 0.0
    details = {"detect_channel10": 0.0, "detect_name": 0.0, "detect_gm_ratio": 0.0, "detect_score": 0.0}
    try:
        mf = mido.MidiFile(str(path))
        has_ch10 = any(getattr(msg, "channel", None) == 9 for tr in mf.tracks for msg in tr)
        has_name = any(
            msg.type == "track_name" and any(k in msg.name.lower() for k in ("drum", "kit", "perc"))
            for tr in mf.tracks for msg in tr
        )
        if has_ch10:
            score += th.get("channel10_weight", 1.0)
            details["detect_channel10"] = 1.0
        if has_name:
            score += th.get("name_weight", 1.0)
            details["detect_name"] = 1.0
    except Exception:
        pass

    drum_hits = 0
    total_hits = 0
    for inst in p.instruments:
        for n in inst.notes:
            total_hits += 1
            if inst.is_drum or 35 <= n.pitch <= 81:
                drum_hits += 1
    if total_hits == 0 or total_hits < th.get("min_hits", 0):
        return False, details
    details["detect_gm_ratio"] = drum_hits / total_hits if total_hits else 0.0
    score += details["detect_gm_ratio"] * th.get("gm_hits_weight", 2.0)
    details["detect_score"] = score
    return score >= 1.0, details


def get_time_signature(p: pm.PrettyMIDI) -> Tuple[str, bool]:
    ts = p.time_signature_changes
    if ts:
        first = ts[0]
        if len(ts) == 1 and first.time == 0 and first.numerator == 4 and first.denominator == 4:
            return "4/4", True
        return f"{first.numerator}/{first.denominator}", False
    return "4/4", True


def bar_count(p: pm.PrettyMIDI, bpm: float) -> float:
    """time_signature_changes を考慮した小節数推定"""
    spb = 60.0 / bpm
    tss = p.time_signature_changes
    if not tss:
        sec = p.get_end_time()
        bar_sec = spb * 4.0
        return max(1.0, sec / bar_sec)
    bars = 0.0
    end_time = p.get_end_time()
    for ts, next_ts in zip(tss, list(tss[1:]) + [None]):
        n, d = ts.numerator, ts.denominator
        bar_sec = spb * n * 4.0 / d
        next_time = next_ts.time if next_ts else end_time
        dur = max(0.0, next_time - ts.time)
        bars += dur / bar_sec
    return max(1.0, bars)


def collect_hits(p: pm.PrettyMIDI) -> List[Tuple[int, float, int]]:
    """(pitch, onset_sec, velocity) のリスト。ドラム優先で全トラック横断。"""
    hits: List[Tuple[int, float, int]] = []
    for inst in p.instruments:
        for n in inst.notes:
            hits.append((int(n.pitch), float(n.start), int(n.velocity)))
    hits.sort(key=lambda x: x[1])
    return hits


def quantize_q16(onsets: np.ndarray, bpm: float) -> np.ndarray:
    # 16分音符間隔（秒）
    step = (60.0 / bpm) / 4.0
    return np.rint(onsets / step).astype(int)


def bar_positions(p: pm.PrettyMIDI, bpm: float, onsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """各ノートの小節内位置と小節長(秒)を返す。"""
    downbeats = p.get_downbeats()
    base = downbeats[0] if len(downbeats) else 0.0
    on = onsets - base
    spb = 60.0 / bpm
    tss = p.time_signature_changes
    if not tss:
        bar_len = spb * 4.0
        return on % bar_len, np.full_like(on, bar_len)
    ts_times = np.array([ts.time for ts in tss] + [p.get_end_time()]) - base
    nums = np.array([ts.numerator for ts in tss])
    dens = np.array([ts.denominator for ts in tss])
    bar_lens = spb * nums * 4.0 / dens
    idx = np.searchsorted(ts_times, on, side="right") - 1
    idx = np.clip(idx, 0, len(bar_lens) - 1)
    rel = on - ts_times[idx]
    bl = bar_lens[idx]
    pos = np.mod(rel, bl)
    return pos, bl


def drum_features(p: pm.PrettyMIDI, bpm: float, bars: float, tempo_fb: bool) -> Dict[str, float]:
    hits = collect_hits(p)
    ts, ts_fb = get_time_signature(p)
    if not hits:
        return {
            "bpm": bpm, "bars": bars, "density": 0.0, "vel_mean": 0.0,
            "vel_std": 0.0, "swing": 0.0, "sync_score": 0.0, "crash_rate": 0.0,
            "fill_ratio": 0.0, "backbeat": 0.0, "ghost_snare_ratio": 0.0,
            "bpm_mad": tempo_mad(p), "note_count": 0, "time_signature": ts,
            "tempo_fallback": tempo_fb, "ts_fallback": ts_fb,
        }
    pitches = np.array([h[0] for h in hits], dtype=int)
    onsets = np.array([h[1] for h in hits], dtype=float)
    vels = np.array([h[2] for h in hits], dtype=float)

    density = (len(hits) / max(1e-6, bars))
    vel_mean = float(np.mean(vels))
    vel_std = float(np.std(vels))

    q16 = quantize_q16(onsets, bpm)
    off8 = (q16 % 2 == 1).mean()
    step = (60.0 / bpm) / 4.0
    residual = np.abs(onsets - (q16 * step)) / max(1e-9, step)
    sync = float(np.mean(residual))

    spb = 60.0 / bpm
    pos_in_bar, bar_len = bar_positions(p, bpm, onsets)
    near_bar_head = (pos_in_bar < (0.07 * bar_len))
    crash_ratio = float(
        ((np.isin(pitches, list(DRUM_CRASH)) & near_bar_head).sum()) / max(1, near_bar_head.sum())
    )

    near_bar_tail = ((bar_len - pos_in_bar) < spb)
    tail_hits = near_bar_tail.sum()
    tom_snare = (np.isin(pitches, list(DRUM_TOMS | SNARE)) & near_bar_tail).sum()
    tail_density_ratio = tail_hits * 4.0 / max(1, len(hits))
    tom_snare_ratio = tom_snare / max(1, tail_hits)
    fill_ratio = float(tail_density_ratio * tom_snare_ratio)

    q8 = np.rint(onsets / (spb / 2)).astype(int)
    is_backbeat_slot = ((q8 % 4 == 1) | (q8 % 4 == 3))
    backbeat = float(((np.isin(pitches, list(SNARE))) & is_backbeat_slot).mean())

    snare_hits = np.isin(pitches, list(SNARE))
    ghost_snare_ratio = float(((snare_hits) & (vels <= 35)).sum() / max(1, snare_hits.sum()))

    return {
        "bpm": float(bpm), "bars": float(bars), "density": float(density),
        "vel_mean": vel_mean, "vel_std": vel_std,
        "swing": float(off8), "sync_score": sync,
        "crash_rate": crash_ratio, "fill_ratio": fill_ratio, "backbeat": backbeat,
        "ghost_snare_ratio": ghost_snare_ratio, "bpm_mad": tempo_mad(p),
        "note_count": len(hits), "time_signature": ts,
        "tempo_fallback": tempo_fb, "ts_fallback": ts_fb,
    }


def poly_features(p: pm.PrettyMIDI, bpm: float, bars: float, tempo_fb: bool) -> Dict[str, float]:
    hits = collect_hits(p)
    ts, ts_fb = get_time_signature(p)
    if not hits:
        return {
            "bpm": bpm, "bars": bars, "density": 0.0, "range_semitones": 0.0,
            "poly_sustain_ratio": 0.0, "poly_mean": 0.0, "change_rate": 0.0,
            "pitch_centroid_shift": 0.0, "chord_change_rate": 0.0,
            "bpm_mad": tempo_mad(p), "note_count": 0, "time_signature": ts,
            "tempo_fallback": tempo_fb, "ts_fallback": ts_fb,
        }
    pitches = np.array([h[0] for h in hits], dtype=int)
    onsets = np.array([h[1] for h in hits], dtype=float)
    total_time = p.get_end_time()

    events = []
    for inst in p.instruments:
        for n in inst.notes:
            events.append((n.start, 1))
            events.append((n.end, -1))
    events.sort()
    active = 0
    poly_sum = 0.0
    active_time = 0.0
    last_t = 0.0
    changes = 0
    step = (60.0 / bpm) / 4.0
    grid_notes: Dict[int, List[int]] = {}
    for inst in p.instruments:
        for n in inst.notes:
            q = int(round(n.start / step))
            grid_notes.setdefault(q, []).append(n.pitch)
    for t, v in events:
        if t > last_t:
            dt = t - last_t
            poly_sum += active * dt
            if active > 0:
                active_time += dt
            last_t = t
        prev = active
        active += v
        if active != prev:
            changes += 1
    sustain_ratio = float(active_time / max(1e-9, total_time))
    poly_mean = float(poly_sum / max(1e-9, total_time))
    pitch_range = float(np.percentile(pitches, 90) - np.percentile(pitches, 10))

    keys = sorted(grid_notes)
    prev_centroid = None
    prev_chord = None
    centroid_shift = 0.0
    chord_changes = 0
    for k in keys:
        notes = grid_notes[k]
        centroid = sum(notes) / len(notes)
        chord = {n % 12 for n in notes}
        if prev_centroid is not None:
            centroid_shift += abs(centroid - prev_centroid)
        if prev_chord is not None and chord != prev_chord:
            chord_changes += 1
        prev_centroid = centroid
        prev_chord = chord
    pitch_centroid_shift = centroid_shift / max(1.0, bars)
    chord_change_rate = chord_changes / max(1.0, len(keys) - 1 or 1)
    change_rate = pitch_centroid_shift + chord_change_rate
    density = len(hits) / max(1e-6, bars)

    return {
        "bpm": float(bpm), "bars": float(bars), "density": float(density),
        "range_semitones": pitch_range, "poly_sustain_ratio": sustain_ratio,
        "poly_mean": poly_mean, "change_rate": change_rate,
        "pitch_centroid_shift": pitch_centroid_shift, "chord_change_rate": chord_change_rate,
        "bpm_mad": tempo_mad(p), "note_count": len(hits), "time_signature": ts,
        "tempo_fallback": tempo_fb, "ts_fallback": ts_fb,
    }

# ------------------------------
# ルールベース分類
# ------------------------------

def _pick_label(scores: Dict[str, float]) -> Tuple[str, float, float]:
    total = sum(scores.values())
    if total <= 0:
        return "unknown", 0.0, 0.0
    label = max(scores, key=scores.get)
    score = scores[label]
    conf = score / total if total > 0 else 0.0
    return label, conf, score


def decide_section_drums(feat: Dict[str, float], th: Dict[str, float]) -> Tuple[str, float, float]:
    d = feat
    scores = {
        "fill": d["fill_ratio"],
        "chorus": d["density"] / th["chorus_density"] + d["crash_rate"],
        "intro": max(0.0, (th["intro_density"] - d["density"]) / th["intro_density"]) + max(0.0, (th["intro_crash"] - d["crash_rate"]) / th["intro_crash"]),
        "bridge": d["sync_score"] / th["bridge_sync"] + d.get("bpm_mad", 0.0),
        "outro": max(0.0, (th["outro_density"] - d["density"]) / th["outro_density"]) + max(0.0, (th["outro_vel"] - d["vel_mean"]) / th["outro_vel"]),
        "verse": 0.1,
    }
    label, conf, score = _pick_label(scores)
    return label, conf, score


def decide_mood_drums(feat: Dict[str, float], th: Dict[str, float]) -> Tuple[str, str, float, float]:
    bpm = feat["bpm"]; dens = feat["density"]; vstd = feat["vel_std"]; swing = feat["swing"]
    ghost = feat.get("ghost_snare_ratio", 0.0)
    bpm_var = feat.get("bpm_mad", 0.0)
    scores = {
        "energetic": max(bpm / th["energetic_bpm"], (dens / th["energetic_density"] + feat["vel_mean"] / th["energetic_vel"]) / 2),
        "relaxed": swing if th["relaxed_bpm_low"] <= bpm <= th["relaxed_bpm_high"] and th["relaxed_dens_low"] <= dens <= th["relaxed_dens_high"] else 0.0,
        "aggressive": (feat["sync_score"] / th["aggressive_sync"]) * (vstd / th["aggressive_vstd"]),
        "melancholic": max(0.0, (th["melancholic_bpm"] - bpm) / th["melancholic_bpm"]) * max(0.0, (th["melancholic_density"] - dens) / th["melancholic_density"]),
        "neutral": 0.1,
    }
    scores["relaxed"] = scores["relaxed"] * (1.0 / (1.0 + bpm_var)) + ghost
    scores["aggressive"] *= max(0.1, 1.0 - ghost)
    mood, conf, score = _pick_label(scores)
    intensity = {
        "energetic": "high",
        "relaxed": "medium",
        "aggressive": "high",
        "melancholic": "low",
        "neutral": "medium",
    }[mood]
    return mood, intensity, conf, score


def decide_section_poly(feat: Dict[str, float], th: Dict[str, float]) -> Tuple[str, float, float]:
    d = feat
    scores = {
        "chorus": d["density"] / th["chorus_density"] + d["range_semitones"] / th["chorus_range"],
        "intro": max(0.0, (th["intro_density"] - d["density"]) / th["intro_density"]),
        "bridge": d["change_rate"] / th["bridge_change"] + d.get("bpm_mad", 0.0),
        "outro": max(0.0, (th["outro_density"] - d["density"]) / th["outro_density"]),
        "verse": 0.1,
    }
    label, conf, score = _pick_label(scores)
    return label, conf, score


def decide_mood_poly(feat: Dict[str, float], th: Dict[str, float]) -> Tuple[str, str, float, float]:
    bpm = feat["bpm"]; dens = feat["density"]
    scores = {
        "energetic": bpm / th["energetic_bpm"],
        "relaxed": feat["poly_sustain_ratio"] / th["relaxed_sustain"],
        "aggressive": feat["change_rate"] / th["aggressive_change"],
        "melancholic": max(0.0, (th["melancholic_bpm"] - bpm) / th["melancholic_bpm"]) * max(0.0, (th["melancholic_density"] - dens) / th["melancholic_density"]),
        "neutral": 0.1,
    }
    mood, conf, score = _pick_label(scores)
    intensity = {
        "energetic": "high",
        "relaxed": "medium",
        "aggressive": "high",
        "melancholic": "low",
        "neutral": "medium",
    }[mood]
    return mood, intensity, conf, score


def classify(feat: Dict[str, Any], mode: str, th: Dict[str, Dict[str, Any]]) -> Tuple[str, float, float, str, str, float, float]:
    if mode == "drums":
        section, cs, ss = decide_section_drums(feat, th["drums"]["section"])
        mood, intensity, cm, ms = decide_mood_drums(feat, th["drums"]["mood"])
    else:
        section, cs, ss = decide_section_poly(feat, th["poly"]["section"])
        mood, intensity, cm, ms = decide_mood_poly(feat, th["poly"]["mood"])
    return section, cs, ss, mood, intensity, cm, ms


def process_file(
    path: Path,
    root: Path,
    rel_base: Path,
    mode: str,
    mode_overrides: List[Tuple[str, str]],
    th: Dict[str, Dict[str, Any]],
    cache_dir: Optional[Path],
    min_notes: int,
    min_bars: float,
    min_conf_section: float,
    min_conf_mood: float,
) -> Tuple[Optional[TagResult], Optional[Dict[str, str]]]:
    try:
        data = path.read_bytes()
        p = load_pretty_midi(data)
        if p is None:
            raise ValueError("load_failed")
        rel_for_override = path.relative_to(root).as_posix()
        actual_mode = mode
        for pat, m in mode_overrides:
            if fnmatch(rel_for_override, pat):
                actual_mode = m
                break
        detect_detail = {"detect_channel10": 0.0, "detect_name": 0.0, "detect_gm_ratio": 0.0, "detect_score": 0.0}
        if actual_mode == "auto":
            is_drum, detect_detail = is_drum_midi(p, path, th["drums"]["detect"])
            actual_mode = "drums" if is_drum else "poly"
        bpm, tempo_fb = estimate_bpm(p)
        bars = bar_count(p, bpm)
        if bars < min_bars:
            raise ValueError("short_bars")

        cache_file = None
        if cache_dir is not None:
            rel_cache = path.relative_to(root).as_posix().replace("/", "__") + ".json"
            cache_file = cache_dir / rel_cache
            if cache_file.exists() and cache_file.stat().st_mtime >= path.stat().st_mtime:
                feat = json.loads(cache_file.read_text())
            else:
                feat = drum_features(p, bpm, bars, tempo_fb) if actual_mode == "drums" else poly_features(p, bpm, bars, tempo_fb)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(feat))
        else:
            feat = drum_features(p, bpm, bars, tempo_fb) if actual_mode == "drums" else poly_features(p, bpm, bars, tempo_fb)

        feat["mode"] = actual_mode
        feat.update(detect_detail)
        if feat.get("note_count", 0) < min_notes:
            raise ValueError("few_notes")

        section, cs, ss, mood, intensity, cm, ms = classify(feat, actual_mode, th)
        if cs < min_conf_section:
            section = "unknown"
        if cm < min_conf_mood:
            mood = "unknown"
        rel = path.relative_to(rel_base).as_posix()
        return TagResult(rel, section, mood, intensity, cs, cm, section, ss, mood, ms, feat), None
    except Exception as e:
        return None, {"path": str(path), "error": str(e)}

# ------------------------------
# メイン
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ind", required=True)
    ap.add_argument("--out-sections", default=None, help="YAML for sections (requires --out-mood unless --out-combined is used)")
    ap.add_argument("--out-mood", default=None, help="YAML for mood (requires --out-sections unless --out-combined is used)")
    ap.add_argument("--out-combined", default=None, help="Single YAML with section+mood (mutually exclusive with --out-sections/--out-mood)")
    ap.add_argument("--report", default=None, help="CSV report with features and heuristic scores")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--thresholds", default=None, help="JSON/YAML thresholds file (e.g., drums: {section:{fill:0.2}})")
    ap.add_argument("--mode", choices=["auto", "drums", "poly"], default="auto")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--relative-from", default=None)
    ap.add_argument("--shard-size", type=int, default=None)
    ap.add_argument("--glob", default="*", help="Comma-separated patterns to include")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-notes", type=int, default=0)
    ap.add_argument("--min-bars", type=float, default=0.0)
    ap.add_argument("--min-conf-section", type=float, default=0.0)
    ap.add_argument("--min-conf-mood", type=float, default=0.0)
    ap.add_argument("--errors", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--executor", choices=["auto", "thread", "process"], default="auto")
    ap.add_argument("--mode-override", action="append", default=[], help="pattern:mode to override detection")
    ap.add_argument("--compat", choices=["full", "simple"], default="full")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--dedupe-strategy", choices=["error", "skip", "suffix"], default="error")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--split-output", action="store_true", help="When using --out-combined also write sections/mood files")
    args = ap.parse_args()

    if args.out_combined:
        if args.split_output and not (args.out_sections and args.out_mood):
            ap.error("--split-output requires --out-sections and --out-mood")
    else:
        if not (args.out_sections and args.out_mood):
            ap.error("--out-sections and --out-mood are required unless --out-combined is used")

    root = Path(args.ind).resolve()
    assert root.exists(), f"not found: {root}"

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_update(dst[k], v)
            else:
                dst[k] = v

    thresholds = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    user_th = None
    if args.thresholds:
        user_th = _load_thresholds(Path(args.thresholds))
        deep_update(thresholds, user_th)

    mode_overrides: List[Tuple[str, str]] = []
    for mo in args.mode_override:
        try:
            pat, m = mo.split(":", 1)
            mode_overrides.append((pat, m))
        except ValueError:
            ap.error("Invalid --mode-override format")

    cache_dir = None
    if args.cache_dir and not args.no_cache:
        cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    rel_root = Path(args.relative_from).resolve() if args.relative_from else root

    files: List[Path] = []
    for ext in ("*.mid", "*.midi", "*.MID"):
        files.extend(root.rglob(ext))
    files.sort()

    glob_patterns = [g.strip() for g in args.glob.split(",") if g.strip()]
    selected: List[Path] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        if not any(fnmatch(rel, g) for g in glob_patterns):
            continue
        if any(fnmatch(rel, ex) for ex in args.exclude):
            continue
        selected.append(p)
        if args.limit and len(selected) >= args.limit:
            break

    results: List[TagResult] = []
    errors: List[Dict[str, str]] = []

    Executor = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    with Executor(max_workers=args.num_workers) as ex:
        futures = [
            ex.submit(
                process_file,
                p,
                root,
                rel_root,
                args.mode,
                mode_overrides,
                thresholds,
                cache_dir,
                args.min_notes,
                args.min_bars,
                args.min_conf_section,
                args.min_conf_mood,
            )
            for p in selected
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            res, err = fut.result()
            if res:
                results.append(res)
            if err:
                errors.append(err)

    def add_with_strategy(dst: Dict[str, Any], key: str, val: Any) -> None:
        if key in dst:
            if args.dedupe_strategy == "skip":
                return
            if args.dedupe_strategy == "suffix":
                i = 1
                nk = f"{key}_{i}"
                while nk in dst:
                    i += 1
                    nk = f"{key}_{i}"
                key = nk
            else:
                raise ValueError(f"duplicate key: {key}")
        dst[key] = val

    sec_map: Dict[str, Dict[str, str]] = {}
    mood_map: Dict[str, Dict[str, str]] = {}
    combined_map: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, str]] = []
    for r in results:
        meta = {
            "mode": r.features["mode"],
            "section": r.section,
            "mood": r.mood,
            "intensity": r.intensity,
            "confidence_section": round(r.conf_section, 3),
            "confidence_mood": round(r.conf_mood, 3),
            "bpm": round(float(r.features.get("bpm", 0.0)), 3),
            "time_signature": r.features.get("time_signature"),
            "bars": round(float(r.features.get("bars", 0.0)), 3),
            "tempo_fallback": bool(r.features.get("tempo_fallback")),
            "ts_fallback": bool(r.features.get("ts_fallback")),
        }
        if args.compat == "simple":
            add_with_strategy(sec_map, r.path, {"section": r.section})
            add_with_strategy(mood_map, r.path, {"mood": r.mood})
        else:
            add_with_strategy(sec_map, r.path, {**meta, "confidence": f"{r.conf_section:.3f}"})
            add_with_strategy(mood_map, r.path, {**meta, "confidence": f"{r.conf_mood:.3f}"})
        combined_map[r.path] = meta
        row = {
            "path": r.path,
            **{k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in r.features.items()},
            "section": r.section,
            "mood": r.mood,
            "intensity": r.intensity,
            "conf_section": f"{r.conf_section:.3f}",
            "conf_mood": f"{r.conf_mood:.3f}",
            "section_decision_rule": r.section_rule,
            "section_rule_score": f"{r.section_score:.4f}",
            "mood_decision_rule": r.mood_rule,
            "mood_rule_score": f"{r.mood_score:.4f}",
        }
        rows.append(row)

    if not args.dry_run:
        def merge_existing(path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
            if path.exists():
                if args.update:
                    existing = yaml.safe_load(path.read_text()) or {}
                    for k, v in data.items():
                        add_with_strategy(existing, k, v)
                    return existing
                if not args.overwrite:
                    raise SystemExit("YAML already exists. Use --overwrite or --update")
            return data

        if args.out_combined:
            out_comb = Path(args.out_combined)
            combined_write: Dict[str, Any] = {}
            if user_th:
                combined_write["meta"] = {"threshold_overrides": user_th}
            combined_write.update(combined_map)
            combined_final = merge_existing(out_comb, combined_write)
            shards = write_yaml_sharded(combined_final, out_comb, args.shard_size)
            if args.shard_size:
                manifest = {
                    "total": len(combined_final) - (1 if "meta" in combined_final else 0),
                    "shards": [{"path": s[0].name, "count": s[1]} for s in shards],
                }
                (out_comb.parent / "manifest.json").write_text(json.dumps(manifest))
            if args.split_output:
                out_sec = Path(args.out_sections)
                out_mood = Path(args.out_mood)
                sec_final = merge_existing(out_sec, sec_map)
                mood_final = merge_existing(out_mood, mood_map)
                write_yaml_sharded(sec_final, out_sec, args.shard_size)
                write_yaml_sharded(mood_final, out_mood, args.shard_size)
        else:
            out_sec = Path(args.out_sections)
            out_mood = Path(args.out_mood)
            sec_final = merge_existing(out_sec, sec_map)
            mood_final = merge_existing(out_mood, mood_map)
            write_yaml_sharded(sec_final, out_sec, args.shard_size)
            write_yaml_sharded(mood_final, out_mood, args.shard_size)

        if args.report and rows:
            import csv
            rp = Path(args.report); rp.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = sorted({k for row in rows for k in row.keys()})
            with rp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader(); w.writerows(rows)

        if args.errors and errors:
            import csv
            ep = Path(args.errors); ep.parent.mkdir(parents=True, exist_ok=True)
            with ep.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["path", "error"])
                w.writeheader(); w.writerows(errors)

    if args.summary:
        summary = {
            "files": len(selected),
            "tagged": len(results),
            "sections": {k: sum(1 for r in results if r.section == k) for k in {r.section for r in results}},
            "moods": {k: sum(1 for r in results if r.mood == k) for k in {r.mood for r in results}},
            "time_signatures": {k: sum(1 for r in results if r.features.get("time_signature") == k) for k in {r.features.get("time_signature") for r in results}},
            "errors": len(errors),
        }
        Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    logging.info(f"tagged {len(results)} files (errors: {len(errors)})")

if __name__ == "__main__":
    main()
