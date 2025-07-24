from __future__ import annotations

import csv
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


def _extract_features(pm: pretty_midi.PrettyMIDI) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tempo = pm.get_tempo_changes()[0]
    bpm = float(tempo[0]) if getattr(tempo, "size", 0) else 120.0
    if bpm <= 0:
        bpm = 120.0
    beat = 60.0 / bpm
    bar_len = beat * 4
    n_bars = max(1, int(round(pm.get_end_time() / bar_len)))
    density: list[int] = []
    velocity: list[int] = []
    energy: list[int] = []
    for i in range(n_bars):
        start = i * bar_len
        end = start + bar_len
        notes = [n for inst in pm.instruments for n in inst.notes if start <= n.start < end]
        density.append(len(notes))
        if notes:
            velocity.append(int(median(n.velocity for n in notes)))
            energy.append(int(sum(n.velocity for n in notes)))
        else:
            velocity.append(0)
            energy.append(0)
    return np.array(density), np.array(velocity), np.array(energy)


def auto_tag(
    loop_dir: Path,
    *,
    k_intensity: int = 3,
    csv_path: Path | None = None,
) -> dict[str, dict[int, dict[str, str]]]:
    """Return per-bar section and intensity labels for ``loop_dir``."""

    densities: list[int] = []
    velocities: list[int] = []
    energies: list[int] = []
    per_file: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for p in sorted(loop_dir.glob("*.mid")):
        pm = pretty_midi.PrettyMIDI(str(p))
        dens, vels, eng = _extract_features(pm)
        densities.extend(dens)
        velocities.extend(vels)
        energies.extend(eng)
        per_file.append((p.name, dens, vels, eng))
    if not densities:
        return {}

    feats = np.column_stack([densities, velocities, energies])
    if len(feats) < k_intensity:
        k_intensity = max(1, len(feats))
    kmeans = KMeans(n_clusters=k_intensity, random_state=0)
    labels = kmeans.fit_predict(feats)
    order = sorted((c.sum(), idx) for idx, c in enumerate(kmeans.cluster_centers_))
    int_map: dict[int, str] = {}
    names = ["low", "mid", "high"]
    for label, (_, idx) in zip(names, order):
        int_map[idx] = label

    if GaussianHMM is not None and len(densities) >= 4:
        try:
            hmm = GaussianHMM(n_components=4, random_state=0, n_iter=10)
            hmm.fit(np.array(densities).reshape(-1, 1))
            sec_idx = hmm.predict(np.array(densities).reshape(-1, 1))
        except Exception:
            sec_idx = [i % 4 for i in range(len(densities))]
    else:
        sec_idx = [i % 4 for i in range(len(densities))]

    secs = ["intro", "verse", "chorus", "bridge"]
    meta: dict[str, dict[int, dict[str, str]]] = {}
    i = 0
    j = 0
    for name, dens, vels, eng in per_file:
        bar_map: dict[int, dict[str, str]] = {}
        length = len(dens)
        sec_sub = sec_idx[i : i + length]
        int_sub = labels[j : j + length]
        for b in range(length):
            bar_map[b] = {
                "section": secs[sec_sub[b] % 4],
                "intensity": int_map.get(int_sub[b], "mid"),
            }
        meta[name] = bar_map
        i += length
        j += length

    if csv_path is not None:
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "bar", "section", "intensity"])
            for fn, bars in meta.items():
                for b, d in bars.items():
                    writer.writerow([fn, b, d["section"], d["intensity"]])

    return meta


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry
    import argparse

    ap = argparse.ArgumentParser(prog="modcompose tag")
    ap.add_argument("loop_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("meta.json"))
    ap.add_argument("--k-intensity", type=int, default=3)
    ap.add_argument("--csv", type=Path, default=None)
    ns = ap.parse_args(argv)
    meta = auto_tag(ns.loop_dir, k_intensity=ns.k_intensity, csv_path=ns.csv)
    ns.out.write_text(json.dumps(meta))
    print(f"wrote {ns.out}")


if __name__ == "__main__":  # pragma: no cover - manual
    main()
