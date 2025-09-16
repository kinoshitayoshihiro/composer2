from __future__ import annotations
import random
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import os

DENSITY_PRESETS = {
    "low": {"stride": 2, "len": 1.25, "accent": 0.8},
    "med": {"stride": 1, "len": 1.0, "accent": 1.0},
    "high": {"stride": 1, "len": 0.75, "accent": 1.2},
}

# Set SPARKLE_DETERMINISTIC=1 to force deterministic RNG defaults for tests.
_SPARKLE_DETERMINISTIC = os.getenv("SPARKLE_DETERMINISTIC") == "1"


def select_phrase_by_harmony(
    chord_quality: Optional[str], pool: List[Optional[int]], rng: random.Random
) -> Optional[int]:
    """Pick a phrase note deterministically based on chord quality."""
    if not pool:
        return None
    if chord_quality == "min":
        return pool[-1]
    if chord_quality == "maj":
        return pool[0]
    return _weighted_pick(pool, [1.0] * len(pool), rng)


def schedule_phrase_keys(
    num_bars: int,
    cycle_phrase_notes: Optional[List[Optional[int]]],
    sections: Optional[List[Dict]],
    fill_note: Optional[int],
    *,
    cycle_start_bar: int = 0,
    cycle_stride: int = 1,
    lfo: Optional["SectionLFO"] = None,
    style_inject: Optional[Dict] = None,
    fill_policy: str = "section",
    pulse_subdiv: float = 1.0,
    markov: Optional[Dict] = None,
    bar_qualities: Optional[List[Optional[str]]] = None,
    section_pool_weights: Optional[Dict[str, Dict[int, float]]] = None,
    rng: Optional[random.Random] = None,
    stats: Optional[Dict] = None,
) -> Tuple[List[Optional[int]], Dict[int, Tuple[int, float, float]], Dict[int, str]]:
    if rng is None:
        rng = random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random()
    plan: List[Optional[int]] = []
    if markov:
        states = markov.get("states") or []
        T = markov.get("T") or []
        if states and T:
            state = 0
            for i in range(num_bars):
                plan.append(states[state])
                if stats is not None:
                    stats.setdefault("bar_reason", {})[i] = {
                        "source": "markov",
                        "note": states[state],
                    }
                state = markov_pick(T, state, rng)
    if not plan:
        if cycle_phrase_notes:
            L = len(cycle_phrase_notes)
            start = ((cycle_start_bar % L) + L) % L
            stride = max(1, cycle_stride)
            for i in range(num_bars):
                idx = ((i + start) // stride) % L
                plan.append(cycle_phrase_notes[idx])
        else:
            plan = [None] * num_bars
    prev: Optional[int] = None
    for i, pn in enumerate(plan):
        if pn is None:
            prev = None
            continue
        if pn == prev:
            plan[i] = None
        else:
            prev = pn
    density_map: Dict[int, str] = {}
    if stats is not None:
        stats.setdefault("bar_density", density_map)
    if sections:
        for sec in sections:
            mcfg = sec.get("markov")
            start = int(sec.get("start_bar", 0))
            end = int(sec.get("end_bar", num_bars))
            dens = sec.get("density")
            if dens in DENSITY_PRESETS:
                for b in range(max(0, start), min(num_bars, end)):
                    density_map[b] = dens
            if mcfg:
                states = mcfg.get("states") or []
                T = mcfg.get("T") or []
                if states and T:
                    state = 0
                    for b in range(max(0, start), min(num_bars, end)):
                        plan[b] = states[state]
                        if stats is not None:
                            stats.setdefault("bar_reason", {})[b] = {
                                "source": "section",
                                "note": states[state],
                            }
                        state = markov_pick(T, state, rng)
                continue
            pool = sec.get("phrase_pool") or sec.get("pool")
            pbq = sec.get("pool_by_quality")
            weights = sec.get("pool_weights") if pool else None
            tag = sec.get("tag")
            spw = section_pool_weights.get(tag) if section_pool_weights and tag else None
            for b in range(max(0, start), min(num_bars, end)):
                chosen = pool
                use_w = weights
                if pbq and bar_qualities and b < len(bar_qualities):
                    q = bar_qualities[b]
                    if q and q in pbq:
                        chosen = pbq[q]
                        use_w = None
                if chosen:
                    if use_w:
                        note = _weighted_pick(chosen, use_w, rng)
                    elif spw:
                        wlist = [spw.get(n, 1.0) for n in chosen]
                        note = _weighted_pick(chosen, wlist, rng)
                    else:
                        note = select_phrase_by_harmony(
                            bar_qualities[b] if bar_qualities and b < len(bar_qualities) else None,
                            chosen,
                            rng,
                        )
                    plan[b] = note
                    if stats is not None:
                        stats.setdefault("bar_reason", {})[b] = {"source": "section", "note": note}
    fills: Dict[int, Tuple[int, float, float]] = {}
    sources: Dict[int, str] = {}
    fn = None
    candidates = [fill_note, 34, 35]
    for c in candidates:
        if c is not None:
            fn = int(c)
            break
    if fn is not None:
        if sections:
            for sec in sections:
                start_bar = int(sec.get("start_bar", 0))
                end_bar = sec.get("end_bar")
                fc = sec.get("fill_cadence")
                if fc:
                    end = int(end_bar if isinstance(end_bar, int) else num_bars)
                    for b in range(max(0, start_bar), min(num_bars, end)):
                        if (b - start_bar + 1) % fc == 0:
                            if _fill_take(
                                sources.get(b), "section", fill_policy, bar=b, stats=stats
                            ):
                                fills[b] = (fn, pulse_subdiv, 1.0)
                                sources[b] = "section"
                if isinstance(end_bar, int) and 0 < end_bar < num_bars:
                    if _fill_take(
                        sources.get(end_bar - 1),
                        "section",
                        fill_policy,
                        bar=end_bar - 1,
                        stats=stats,
                    ):
                        fills[end_bar - 1] = (fn, pulse_subdiv, 1.0)
                        sources[end_bar - 1] = "section"
        if lfo:
            for i in range(num_bars):
                if lfo.fill_scale(i) >= 0.99 and _fill_take(
                    sources.get(i), "lfo", fill_policy, bar=i, stats=stats
                ):
                    fills[i] = (fn, pulse_subdiv, 1.0)
                    sources[i] = "lfo"
    if style_inject:
        period = int(style_inject.get("period", 0))
        note = style_inject.get("note")
        dur = float(style_inject.get("duration_beats", pulse_subdiv))
        vscale = float(style_inject.get("vel_scale", 1.0))
        min_gap = float(style_inject.get("min_gap_beats", 0.0))
        avoid = {int(n) for n in style_inject.get("avoid_pitches", [])}
        last = -1e9
        if note is not None and period > 0 and dur > 0:
            note = int(note)
            for b in range(0, num_bars, period):
                if avoid and note in avoid:
                    continue
                if (b - last) * pulse_subdiv < min_gap:
                    continue
                if _fill_take(sources.get(b), "style", fill_policy, bar=b, stats=stats):
                    fills[b] = (note, dur, vscale)
                    sources[b] = "style"
                    last = b
    return plan, fills, sources


def _weighted_pick(
    pool: List[Optional[int]], weights: List[float], rng: random.Random
) -> Optional[int]:
    total = sum(weights)
    if total <= 0:
        idx = rng_range(len(pool), rng)
        return pool[idx]
    r = rng.random() * total
    acc = 0.0
    for note, w in zip(pool, weights):
        acc += w
        if r <= acc:
            return note
    return pool[-1]


def _fill_take(
    existing: Optional[str],
    new: str,
    policy: str,
    *,
    bar: Optional[int] = None,
    stats: Optional[Dict] = None,
) -> bool:
    if existing is None:
        return True
    if policy == "first":
        take = False
    elif policy == "last":
        take = True
    elif policy == new:
        take = True
    elif policy == existing:
        take = False
    else:
        take = False
    if stats is not None and not take and bar is not None:
        stats.setdefault("fill_conflicts", []).append(
            {"bar": bar, "existing": existing, "new": new, "policy": policy}
        )
    return take


@dataclass
class ChordSpan:
    start: float
    end: float
    root_pc: int
    quality: str


@dataclass
class SectionLFO:
    bars_period: int
    phase: float = 0.0
    vel_range: Tuple[float, float] = (1.0, 1.0)
    fill_range: Tuple[float, float] = (0.0, 0.0)
    shape: str = "linear"

    def _pos(self, bar: int) -> float:
        if self.bars_period <= 0:
            return 0.0
        base = (bar % self.bars_period) / self.bars_period
        x = (base + self.phase) % 1.0
        if self.shape == "sine":
            return 0.5 - 0.5 * math.cos(2 * math.pi * x)
        if self.shape == "triangle":
            return 2 * x if x < 0.5 else 2 - 2 * x
        return x

    def vel_scale(self, bar: int) -> float:
        a, b = self.vel_range
        return a + (b - a) * self._pos(bar)

    def fill_scale(self, bar: int) -> float:
        a, b = self.fill_range
        return a + (b - a) * self._pos(bar)


@dataclass
class StableChordGuard:
    min_hold_beats: int
    strategy: str = "skip"

    def __post_init__(self) -> None:
        self.current = None
        self.hold = 0.0
        self.toggle = False

    def step(self, chord_key: Tuple[int, str], beat_inc: float) -> None:
        if self.current == chord_key:
            self.hold += beat_inc
        else:
            self.current = chord_key
            self.hold = beat_inc
            self.toggle = False

    def filter(self, note: Optional[int]) -> Optional[int]:
        if note is None:
            return None
        if self.hold < self.min_hold_beats:
            return note
        if self.strategy == "alternate":
            self.toggle = not self.toggle
            return note if self.toggle else None
        return None


@dataclass
class VocalAdaptive:
    dense_onset: int
    dense_phrase: Optional[int]
    sparse_phrase: Optional[int]
    onsets: List[int]
    ratios: Optional[List[float]] = None
    dense_ratio: Optional[float] = None
    smooth_bars: int = 0

    def phrase_for_bar(self, bar_idx: int) -> Optional[int]:
        cnt = self.onsets[bar_idx] if bar_idx < len(self.onsets) else 0
        ratio = self.ratios[bar_idx] if self.ratios and bar_idx < len(self.ratios) else 0.0
        if cnt >= self.dense_onset or (self.dense_ratio is not None and ratio >= self.dense_ratio):
            return self.dense_phrase
        return self.sparse_phrase

    def __post_init__(self) -> None:
        if self.smooth_bars and self.smooth_bars > 1:
            k = self.smooth_bars
            self.onsets = _moving_average(self.onsets, k)
            if self.ratios:
                self.ratios = _moving_average(self.ratios, k)


def _moving_average(seq: List[float], k: int) -> List[float]:
    out: List[float] = []
    for i in range(len(seq)):
        s = 0.0
        c = 0
        for j in range(i - k + 1, i + k):
            if 0 <= j < len(seq):
                s += seq[j]
                c += 1
        out.append(s / c if c else 0.0)
    return out


def markov_pick(T: List[List[float]], state: int, rng: random.Random) -> int:
    row = T[state]
    s = sum(row)
    if s <= 0:
        return rng_range(len(T), rng)
    r = rng.random() * s
    acc = 0.0
    for i, p in enumerate(row):
        acc += p
        if r <= acc:
            return i
    return len(row) - 1


def rng_range(n: int, rng: random.Random) -> int:
    return int(rng.random() * n)
