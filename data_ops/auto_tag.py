from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np
import pretty_midi
from sklearn.cluster import KMeans

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - optional dependency
    GaussianHMM = None


def _extract_features(pm: pretty_midi.PrettyMIDI) -> tuple[list[int], list[int]]:
    tempo = pm.get_tempo_changes()[1]
    bpm = float(tempo[0]) if getattr(tempo, "size", 0) else 120.0
    beat = 60.0 / bpm
    bar_len = beat * 4
    n_bars = max(1, int(round(pm.get_end_time() / bar_len)))
    density: list[int] = []
    velocity: list[int] = []
    for i in range(n_bars):
        start = i * bar_len
        end = start + bar_len
        notes = [n for inst in pm.instruments for n in inst.notes if start <= n.start < end]
        density.append(len(notes))
        if notes:
            velocity.append(int(median(n.velocity for n in notes)))
        else:
            velocity.append(0)
    return density, velocity


def auto_tag(loop_dir: Path) -> dict[str, dict[str, str]]:
    """Return metadata inferred from loops in ``loop_dir``."""
    densities: list[int] = []
    velocities: list[int] = []
    per_file: list[tuple[str, list[int], list[int]]] = []
    for p in sorted(loop_dir.glob("*.mid")):
        pm = pretty_midi.PrettyMIDI(str(p))
        dens, vels = _extract_features(pm)
        densities.extend(dens)
        velocities.extend(vels)
        per_file.append((p.name, dens, vels))
    if not densities:
        return {}
    if len(velocities) < 3:
        return {name: {"intensity": "mid", "section": "verse"} for name, _, _ in per_file}
    kmeans = KMeans(n_clusters=3, random_state=0)
    vel_lbl = kmeans.fit_predict(np.array(velocities).reshape(-1, 1))
    order = sorted((c, idx) for idx, c in enumerate(kmeans.cluster_centers_.flatten()))
    lab_map = {idx: label for label, (_, idx) in zip(["low", "mid", "high"], order)}
    if GaussianHMM is not None:
        hmm = GaussianHMM(n_components=4, random_state=0, n_iter=10)
        hmm.fit(np.array(densities).reshape(-1, 1))
        sec_idx = hmm.predict(np.array(densities).reshape(-1, 1))
    else:
        sec_idx = [i % 4 for i in range(len(densities))]
    secs = ["verse", "pre-chorus", "chorus", "bridge"]
    meta: dict[str, dict[str, str]] = {}
    j = 0
    for name, dens, vels in per_file:
        length = len(dens)
        sub_idx = sec_idx[j : j + length]
        j += length
        intensity = lab_map[vel_lbl.pop(0)] if vel_lbl.size else "mid"
        meta[name] = {"intensity": intensity, "section": secs[sub_idx[0] % 4]}
    return meta


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry
    import argparse

    ap = argparse.ArgumentParser(prog="modcompose tag")
    ap.add_argument("loop_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("meta.json"))
    ns = ap.parse_args(argv)
    meta = auto_tag(ns.loop_dir)
    ns.out.write_text(json.dumps(meta))
    print(f"wrote {ns.out}")


if __name__ == "__main__":  # pragma: no cover - manual
    main()
