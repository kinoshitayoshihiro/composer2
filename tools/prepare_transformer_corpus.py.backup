from __future__ import annotations

__doc__ = """Prepare a small Transformer corpus from MIDI files.

This script builds a dataset suitable for training sequence models on personal
MIDI collections. It chops each MIDI file into fixed-size bar segments, applies
simple tokenisation and optionally merges metadata such as tags or lyric text.
The resulting dataset is written as JSONL files with deterministic train/valid/
test splits.

Example
-------
python -m tools.prepare_transformer_corpus \
  --in data/midi_personal \
  --out data/corpus/personal_v1 \
  --bars-per-sample 4 --quant 480 --min-notes 8 \
  --duv on --dur-bins 16 --vel-bins 8 --duv-max 1000 \
  --embed-offline embeds.json \
  --tags sections.yaml mood.yaml \
  --split 0.9 0.05 0.05 --seed 42
"""

import argparse
import json
import logging
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
)

import concurrent.futures

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Tokenised sample with metadata."""

    tokens: List[str]
    meta: Dict[str, object]


def build_beat_map(pm: "pretty_midi.PrettyMIDI") -> Tuple[Callable[[float], float], float]:
    """Return a function mapping seconds to beats and an estimated tempo."""

    beat_times = pm.get_beats()
    if len(beat_times) >= 2:
        beat_idx = np.arange(len(beat_times), dtype=float)
        tempo_est = 60.0 / np.diff(beat_times).mean()

        def sec_to_beats(t: float) -> float:
            if t <= beat_times[-1]:
                return float(np.interp(t, beat_times, beat_idx))
            extra = (t - beat_times[-1]) * tempo_est / 60.0
            return float(beat_idx[-1] + extra)

    else:  # fallback to estimate_tempo
        tempo_est = float(pm.estimate_tempo() or 120.0)

        def sec_to_beats(t: float) -> float:
            return t * tempo_est / 60.0

    return sec_to_beats, tempo_est


def get_time_signature(mid: "mido.MidiFile") -> Tuple[float, str]:
    """Extract first time signature; default to 4/4."""

    for track in mid.tracks:
        for msg in track:
            if msg.type == "time_signature":
                beats_per_bar = msg.numerator * 4.0 / msg.denominator
                return beats_per_bar, f"{msg.numerator}/{msg.denominator}"
    return 4.0, "4/4"


def quantise_beats(value: float, quant: int) -> float:
    """Quantise ``value`` (in beats) to ``1/quant`` resolution."""

    return round(value * quant) / quant


def bin_duration(duration_beats: float, bins: int) -> int:
    """Map a duration in beats to a discrete bin index."""

    return int(np.clip(round(duration_beats * bins), 0, bins - 1))


def bin_velocity(velocity: int, bins: int) -> int:
    """Map velocity 0-127 to a discrete bin index."""

    return int(np.clip(velocity * bins // 128, 0, bins - 1))


def tokenize_notes(
    notes: Iterable["pretty_midi.Note"],
    *,
    duv: bool,
    dur_bins: int,
    vel_bins: int,
    quant: int,
) -> List[str]:
    """Tokenise notes into simple NOTE/DUR/VEL or NOTE/DUV tokens.

    Notes are sorted for determinism and start/end are expressed in beats.
    """

    tokens: List[str] = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch, n.end, n.velocity)):
        start = quantise_beats(n.start, quant)
        end = quantise_beats(n.end, quant)
        dur_beats = max(0.0, end - start)
        d_bin = bin_duration(dur_beats, dur_bins)
        v_bin = bin_velocity(n.velocity, vel_bins)
        tokens.append(f"NOTE_{n.pitch}")
        if duv:
            tokens.append(f"DUV_{d_bin}_{v_bin}")
        else:
            tokens.append(f"D_{d_bin}")
            tokens.append(f"V_{v_bin}")
    return tokens


def load_tag_maps(tag_files: Sequence[Path]) -> Dict[str, Dict[str, str]]:
    """Load per-file metadata from YAML files."""

    tag_map: Dict[str, Dict[str, str]] = {}
    for path in tag_files:
        if not path.is_file():
            logger.warning("tag file %s missing", path)
            continue
        data = yaml.safe_load(path.read_text()) or {}
        for fp, tags in data.items():
            tag_map.setdefault(fp, {}).update(tags or {})
    return tag_map


def gather_midi_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.mid") if p.is_file())


def normalize_key(path: Path, base: Path) -> str:
    """Return a base-relative, lowercased POSIX-style key for *path*."""

    try:
        rel = path.resolve().relative_to(base.resolve())
    except Exception:
        rel = Path(path.name)
    return rel.as_posix().lower()


def load_embed_map(path: Path) -> Dict[str, List[float]]:
    """Load pre-computed text embeddings from JSON or NPZ."""

    import numpy as np

    if path.suffix == ".npz":
        data = np.load(path)
        embed_map = {k: data[k].tolist() for k in data.files}
    else:
        embed_map = {
            k: list(v)
            for k, v in json.loads(path.read_text()).items()
        }
    dims = {len(v) for v in embed_map.values()}
    if len(dims) != 1:
        raise ValueError("inconsistent embedding vector lengths")
    return embed_map


def split_samples(
    pm: "pretty_midi.PrettyMIDI",
    *,
    bars_per_sample: int,
    min_notes: int,
    beats_per_bar: float,
    sec_to_beats: Callable[[float], float],
    include_programs: set[int] | None,
    drums_only: bool,
    exclude_drums: bool,
    max_segments: int | None = None,
) -> Iterator[List["pretty_midi.Note"]]:
    """Yield note groups for each slice of ``bars_per_sample`` bars."""

    import pretty_midi

    segment_beats = bars_per_sample * beats_per_bar
    total_beats = sec_to_beats(pm.get_end_time())
    n_segments = int(total_beats // segment_beats)
    for i in range(n_segments):
        start = i * segment_beats
        end = start + segment_beats
        seg: List[pretty_midi.Note] = []
        for inst in pm.instruments:
            if drums_only and not inst.is_drum:
                continue
            if exclude_drums and inst.is_drum:
                continue
            if include_programs and inst.program not in include_programs:
                continue
            for n in inst.notes:
                n_start = sec_to_beats(n.start)
                if start <= n_start < end:
                    n_end = sec_to_beats(n.end)
                    seg.append(
                        pretty_midi.Note(
                            velocity=n.velocity,
                            pitch=n.pitch,
                            start=n_start - start,
                            end=min(n_end, end) - start,
                        )
                    )
        if len(seg) >= min_notes:
            yield seg
        if max_segments is not None and i + 1 >= max_segments:
            break


def build_corpus(args: argparse.Namespace) -> Dict[str, List[Sample]]:
    """Process input tree and return split samples."""

    try:
        import numpy as np  # noqa: F401
        import pretty_midi  # noqa: F401
        import mido  # noqa: F401
        import yaml  # noqa: F401
        from utilities.pretty_midi_safe import pm_to_mido
    except ImportError as exc:  # pragma: no cover - checked in tests
        if exc.name == "numpy":
            raise SystemExit("numpy is required; pip install numpy") from exc
        raise SystemExit("Missing MIDI dependencies; pip install pretty_midi mido PyYAML") from exc

    tag_map = load_tag_maps([Path(p) for p in args.tags]) if args.tags else {}
    lyric_json = getattr(args, "lyric_json", None)
    if lyric_json:
        base = Path(args.in_dir).resolve()
        raw_map = json.loads(Path(lyric_json).read_text())
        lyric_map = {normalize_key(Path(k), base): v for k, v in raw_map.items()}
    else:
        lyric_map = {}
    embed_map: Dict[str, List[float]] = {}
    if getattr(args, "embed_offline", None):
        embed_map = load_embed_map(Path(args.embed_offline))
        dim = len(next(iter(embed_map.values()))) if embed_map else 0
        logger.info("loaded %d offline embeddings (dim=%d)", len(embed_map), dim)
    embed_model = None
    if lyric_map and not embed_map:
        try:  # pragma: no cover - optional dependency
            from sentence_transformers import SentenceTransformer

            embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:  # pragma: no cover - handled in tests
            logger.warning("sentence-transformers unavailable; storing raw text")

    base = Path(args.in_dir)
    files = gather_midi_files(base)
    if args.max_files:
        files = files[: args.max_files]

    iterator: Iterable[Path] = files
    if args.progress:
        try:  # pragma: no cover - optional
            from tqdm import tqdm

            iterator = tqdm(iterator)
        except Exception:  # pragma: no cover
            pass

    lyric_matches = sum(1 for f in files if normalize_key(f, base) in lyric_map)
    logger.info("lyrics matched: %d/%d", lyric_matches, len(files))

    def process_path(midi_path: Path) -> List[Sample]:
        pm = pretty_midi.PrettyMIDI(midi_path.as_posix())
        mid = pm_to_mido(pm)
        beats_per_bar, ts_str = get_time_signature(mid)
        sec_to_beats, tempo_est = build_beat_map(pm)
        rel = normalize_key(midi_path, base)
        segments: List[Sample] = []
        include_programs = set(args.include_programs) if args.include_programs else None
        for idx, seg in enumerate(
            split_samples(
                pm,
                bars_per_sample=args.bars_per_sample,
                min_notes=args.min_notes,
                beats_per_bar=beats_per_bar,
                sec_to_beats=sec_to_beats,
                include_programs=include_programs,
                drums_only=args.drums_only,
                exclude_drums=args.exclude_drums,
                max_segments=args.max_samples_per_file,
            )
        ):
            tokens = tokenize_notes(
                seg,
                duv=args.duv == "on",
                dur_bins=args.dur_bins,
                vel_bins=args.vel_bins,
                quant=args.quant,
            )
            tags = tag_map.get(rel, {})
            if args.section_tokens and tags.get("section"):
                tokens.insert(0, f"<SECTION={tags['section']}>")
            if args.mood_tokens and tags.get("mood"):
                tokens.insert(0, f"<MOOD={tags['mood']}>")
            meta: Dict[str, object] = {
                **tags,
                "source_path": rel,
                "segment_index": idx,
                "tempo_est": tempo_est,
                "beats_per_bar": beats_per_bar,
                "time_signature": ts_str,
            }
            if rel in embed_map:
                meta["text_emb"] = embed_map[rel]
            elif rel in lyric_map:
                text = lyric_map[rel]
                if embed_model is not None:
                    meta["text_emb"] = embed_model.encode(text).tolist()
                else:
                    meta["text"] = text
            segments.append(Sample(tokens=tokens, meta=meta))
        return segments

    samples: List[Sample] = []
    use_mp = args.num_workers > 1 and embed_model is None
    if use_mp:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            for res in ex.map(process_path, iterator):
                samples.extend(res)
    else:
        if args.num_workers > 1 and embed_model is not None:
            logger.warning("lyric embeddings disable multiprocessing; using single worker")
        for p in iterator:
            samples.extend(process_path(p))

    if not samples:
        logger.warning("no samples found under %s", args.in_dir)
        return {"train": [], "valid": [], "test": []}, lyric_matches, len(files)

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_total = len(samples)
    n_train = int(args.split[0] * n_total)
    n_valid = int(args.split[1] * n_total)
    n_test = n_total - n_train - n_valid
    logger.info(
        "split counts train=%d valid=%d test=%d total=%d", n_train, n_valid, n_test, n_total
    )
    splits = {
        "train": samples[:n_train],
        "valid": samples[n_train : n_train + n_valid],
        "test": samples[n_train + n_valid : n_train + n_valid + n_test],
    }
    return splits, lyric_matches, len(files)


def save_jsonl(path: Path, samples: Sequence[Sample], *, compress: str = "none") -> None:
    """Write ``samples`` to ``path`` as JSONL, optionally gzip-compressed."""

    actual_path = path
    if compress == "gz":
        import gzip

        actual_path = path.with_suffix(path.suffix + ".gz")
        fh = gzip.open(actual_path, "wt", encoding="utf-8")
    else:
        fh = actual_path.open("w", encoding="utf-8")
    with fh:
        for s in samples:
            fh.write(json.dumps({"tokens": s.tokens, "meta": s.meta}) + "\n")


def build_vocab(samples: Iterable[Sample]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for s in samples:
        for tok in s.tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def build_tag_vocab(samples: Iterable[Sample]) -> Dict[str, Dict[str, int]]:
    vocab: Dict[str, Dict[str, int]] = {}
    for s in samples:
        for k, v in s.meta.items():
            if not isinstance(v, str):
                continue
            vocab.setdefault(k, {})
            if v not in vocab[k]:
                vocab[k][v] = len(vocab[k])
    return vocab


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", type=str, required=True, help="input MIDI folder")
    ap.add_argument("--out", dest="out_dir", type=str, required=True, help="output corpus root")
    ap.add_argument("--bars-per-sample", type=int, default=4)
    ap.add_argument(
        "--quant",
        type=int,
        default=480,
        help="per-beat resolution (e.g., 96/240/480)",
    )
    ap.add_argument("--min-notes", type=int, default=1)
    ap.add_argument("--duv", type=str, choices=["on", "off"], default="off")
    ap.add_argument("--dur-bins", type=int, default=16)
    ap.add_argument("--vel-bins", type=int, default=8)
    ap.add_argument("--tags", nargs="*", default=[], help="YAML metadata files")
    ap.add_argument("--lyric-json", type=str, default=None, help="JSON file mapping paths to lyrics")
    ap.add_argument(
        "--embed-offline",
        type=str,
        default=None,
        help="JSON or NPZ mapping paths to precomputed text embeddings",
    )
    ap.add_argument("--split", nargs=3, type=float, default=(0.9, 0.05, 0.05))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--section-tokens", action="store_true")
    ap.add_argument("--mood-tokens", action="store_true")
    ap.add_argument("--include-programs", nargs="*", type=int, default=None)
    ap.add_argument("--drums-only", action="store_true")
    ap.add_argument("--exclude-drums", action="store_true")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-samples-per-file", type=int, default=None)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "worker processes (disabled when embedding text at runtime; "
            "use --embed-offline to re-enable)"
        ),
    )
    ap.add_argument(
        "--duv-max",
        type=int,
        default=None,
        help="max distinct DUV tokens; rare pairs collapse to DUV_OOV",
    )
    ap.add_argument("--compress", choices=["none", "gz"], default="none", help="compress output JSONL")
    ns = ap.parse_args(argv)

    if ns.dur_bins not in {4, 8, 16} or ns.vel_bins not in {4, 8, 16}:
        ap.error("--dur-bins and --vel-bins must be one of 4,8,16")
    if ns.duv_max is not None and not (1 <= ns.duv_max <= 255):
        ap.error("--duv-max must be between 1 and 255")
    if (
        len(ns.split) != 3
        or any((s < 0 or s > 1) for s in ns.split)
        or not math.isclose(sum(ns.split), 1.0, abs_tol=1e-6)
    ):
        ap.error("--split must be three fractions summing to 1.0")

    logging.basicConfig(level=logging.INFO)
    splits, lyric_matches, files_scanned = build_corpus(ns)

    duv_freq = Counter()
    for samp in splits.values():
        for s in samp:
            for tok in s.tokens:
                if tok.startswith("DUV_"):
                    duv_freq[tok] += 1
    if ns.duv_max is not None and duv_freq:
        total_duv = len(duv_freq)
        keep = {t for t, _ in duv_freq.most_common(ns.duv_max)}
        collapsed = total_duv - len(keep)
        if collapsed > 0:
            for samp in splits.values():
                for s in samp:
                    s.tokens = [
                        t if not t.startswith("DUV_") or t in keep else "DUV_OOV"
                        for t in s.tokens
                    ]
            duv_freq = Counter()
            for samp in splits.values():
                for s in samp:
                    for tok in s.tokens:
                        if tok.startswith("DUV_"):
                            duv_freq[tok] += 1
        logger.info("DUV kept: %d collapsed: %d", len(keep), collapsed)

    out_root = Path(ns.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for name, samp in splits.items():
        save_jsonl(out_root / f"{name}.jsonl", samp, compress=ns.compress)

    vocab = build_vocab(s for s in splits.values() for s in s)
    tag_vocab = build_tag_vocab(s for s in splits.values() for s in s)
    stats = {k: len(v) for k, v in splits.items()}
    meta = {
        "vocab": vocab,
        "tag_vocab": tag_vocab,
        "stats": stats,
        "duv_freq": dict(duv_freq),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    total_samples = sum(stats.values())
    summary = {
        "files_scanned": files_scanned,
        "samples": total_samples,
        "train": stats.get("train", 0),
        "valid": stats.get("valid", 0),
        "test": stats.get("test", 0),
        "lyrics_matched": lyric_matches,
    }
    logger.info("%s", json.dumps(summary))


if __name__ == "__main__":  # pragma: no cover
    main()
