# utilities/groove_sampler.py
"""Simple n-gram groove sampler for drum patterns."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List
import random

import pretty_midi


def _tokenize_bar(pm: pretty_midi.PrettyMIDI, start_beat: float, n_subdiv: int = 16) -> List[str]:
    """Return list of hit tokens for a single bar."""
    sec_per_beat = 60.0 / (pm.get_tempo_changes()[1][0] if pm.get_tempo_changes()[1].size else 120.0)
    bar_tokens = ["" for _ in range(n_subdiv)]
    for inst in pm.instruments:
        for note in inst.notes:
            beat_pos = note.start / sec_per_beat
            if start_beat <= beat_pos < start_beat + 4.0:
                step = int(round((beat_pos - start_beat) * (n_subdiv / 4)))
                if 0 <= step < n_subdiv:
                    token = str(note.pitch)
                    if bar_tokens[step]:
                        bar_tokens[step] += "+" + token
                    else:
                        bar_tokens[step] = token
    return bar_tokens


def load_grooves(midi_dir: Path, n: int = 3) -> Dict:
    """Load MIDI files in ``midi_dir`` and build an n-gram model."""
    model: Dict[Tuple[str, ...], Dict[str, int]] = {}
    midi_paths = [p for p in midi_dir.glob("*.mid") if p.is_file()]
    for mp in midi_paths:
        pm = pretty_midi.PrettyMIDI(str(mp))
        end_beat = pm.get_end_time() / (60.0 / (pm.get_tempo_changes()[1][0] if pm.get_tempo_changes()[1].size else 120.0))
        bars = int(end_beat // 4)
        for b in range(bars):
            seq = _tokenize_bar(pm, b * 4.0)
            for i in range(len(seq) - n + 1):
                prev = tuple(seq[i : i + n - 1])
                nxt = seq[i + n - 1]
                if prev not in model:
                    model[prev] = {nxt: 1}
                else:
                    model[prev][nxt] = model[prev].get(nxt, 0) + 1
    model['n'] = n
    return model


def sample_next(prev_hits: Tuple[str, ...], model: Dict, rng: random.Random) -> str:
    """Sample the next hit given ``prev_hits`` using ``model``."""
    trans = model.get(prev_hits)
    if not trans:
        keys = [k for k in model.keys() if isinstance(k, tuple) and k]
        if not keys:
            return ""
        prev_hits = rng.choice(keys)
        trans = model.get(prev_hits, {})
    total = sum(trans.values())
    r = rng.uniform(0, total)
    cum = 0
    for hit, cnt in trans.items():
        cum += cnt
        if r <= cum:
            return hit
    return ""


def generate_bar(prev_hits: List[str], model: Dict, rng: random.Random, length: int = 16) -> List[Dict[str, float]]:
    """Generate one bar of events from the model."""
    n = int(model.get('n', 3))
    context = tuple(prev_hits[-(n - 1) :]) if n > 1 else tuple()
    events: List[Dict[str, float]] = []
    for i in range(length):
        hit = sample_next(context, model, rng)
        if hit:
            events.append({
                'instrument': hit,
                'offset': i * 0.25,
                'duration': 0.25,
            })
        if n > 1:
            context = (*context[1:], hit) if context else (hit,)
    return events
