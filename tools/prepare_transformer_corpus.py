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
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable, Sequence
from collections import Counter
import concurrent.futures

import pretty_midi
import numpy as np

from utilities.pretty_midi_safe import pm_to_mido

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Tokenised sample with metadata."""

    tokens: List[str]
    meta: Dict[str, object]


def _total_notes(pm) -> int:
    """Return total note count in a PrettyMIDI object; safe on weird inputs."""
    try:
        return sum(len(inst.notes) for inst in getattr(pm, "instruments", []))
    except Exception:
        return 0


def _make_const_sec_to_beats(tempo_bpm: float):
    """Simple seconds->beats mapping under constant tempo assumption."""
    spb = 60.0 / max(float(tempo_bpm), 1e-6)
    return lambda t: t / spb


def build_beat_map(
    pm: "pretty_midi.PrettyMIDI", *, path: Path | None = None
) -> tuple[Callable[[float], float], float, bool]:
    """Return seconds->beats map, tempo estimate and fallback flag."""

    tempo_est: Optional[float] = None
    try:
        t = float(pm.estimate_tempo())
        if math.isfinite(t) and t > 0:
            tempo_est = t
    except Exception as e:  # pragma: no cover
        logger.debug("estimate_tempo failed for %s: %s", path, e)
    fallback = False
    if tempo_est is None:
        tempi, _times = pm.get_tempo_changes()
        if len(tempi) and math.isfinite(float(tempi[0])) and float(tempi[0]) > 0:
            tempo_est = float(tempi[0])
        else:
            tempo_est = 120.0
        fallback = True
        logger.debug("tempo fallback used for %s: %.1f", path, tempo_est)

    sec_to_beats = _make_const_sec_to_beats(tempo_est)
    return sec_to_beats, tempo_est, fallback


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
        embed_map = {k: list(v) for k, v in json.loads(path.read_text()).items()}
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
        else:
            _inc_skip("too_few_notes")
        if max_segments is not None and i + 1 >= max_segments:
            break


def build_corpus(args: argparse.Namespace, files: Sequence[Path]) -> Dict[str, List[Sample]]:
    """Process *files* and return split samples."""

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

    global _ARGS, _BASE, _TAG_MAP, _LYRIC_MAP, _EMBED_MAP, _EMBED_MODEL
    _ARGS = args
    base = Path(args.in_dir)
    _BASE = base
    _TAG_MAP = tag_map
    _LYRIC_MAP = lyric_map
    _EMBED_MAP = embed_map
    _EMBED_MODEL = embed_model

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

    samples: List[Sample] = []
    use_mp = args.num_workers > 1 and embed_model is None
    paths = list(iterator)
    cfg = FastCfg()
    cfg.drums_only = args.drums_only
    cfg.min_file_notes = args.min_file_notes
    cfg.min_file_seconds = args.min_file_seconds
    cfg.max_file_seconds = args.max_file_seconds
    cfg.silence_threshold_db = args.silence_threshold_db
    cfg.silent_fraction = args.silent_fraction
    cfg.skip_lyrics = args.skip_lyrics
    if use_mp:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                for res in ex.map(_worker, ((p, cfg) for p in paths)):
                    samples.extend(res)
        except Exception:
            logger.warning("ProcessPoolExecutor failed; falling back to ThreadPoolExecutor")
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                for res in ex.map(_worker, ((p, cfg) for p in paths)):
                    samples.extend(res)
    else:
        if args.num_workers > 1 and embed_model is not None:
            logger.warning("lyric embeddings disable multiprocessing; using single worker")
        for p in iterator:
            samples.extend(process_path(p, cfg))

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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prepare_transformer_corpus")
    p.add_argument("--in", dest="in_dir", type=str, required=True, help="input MIDI folder")
    p.add_argument("--out", dest="out_dir", type=str, required=True, help="output corpus root")
    p.add_argument("--bars-per-sample", type=int, default=4)
    p.add_argument(
        "--quant",
        type=int,
        default=480,
        help="per-beat resolution (e.g., 96/240/480)",
    )
    p.add_argument("--min-notes", type=int, default=1)
    p.add_argument("--duv", type=str, choices=["on", "off"], default="off")
    p.add_argument("--dur-bins", type=int, default=16)
    p.add_argument("--vel-bins", type=int, default=8)
    p.add_argument("--tags", nargs="*", default=[], help="YAML metadata files")
    p.add_argument("--lyric-json", type=str, default=None, help="JSON file mapping paths to lyrics")
    p.add_argument(
        "--embed-offline",
        type=str,
        default=None,
        help="JSON or NPZ mapping paths to precomputed text embeddings",
    )
    p.add_argument("--split", nargs=3, type=float, default=(0.9, 0.05, 0.05))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--section-tokens", action="store_true")
    p.add_argument("--mood-tokens", action="store_true")
    p.add_argument("--include-programs", nargs="*", type=int, default=None)
    p.add_argument("--drums-only", action="store_true")
    p.add_argument("--exclude-drums", action="store_true")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-samples-per-file", type=int, default=None)
    p.add_argument("--progress", action="store_true")
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "worker processes (disabled when embedding text at runtime; "
            "use --embed-offline to re-enable)"
        ),
    )
    p.add_argument(
        "--duv-max",
        type=int,
        default=None,
        help="max distinct DUV tokens; rare pairs collapse to DUV_OOV",
    )
    p.add_argument(
        "--compress", choices=["none", "gz"], default="none", help="compress output JSONL"
    )
    # ---- Silence / envelope early exit
    p.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-60.0,
        help="RMS dB threshold to treat as silent (default: -60dB)",
    )
    p.add_argument(
        "--silent-fraction",
        type=float,
        default=0.95,
        help="fraction of silent frames to skip (default: 0.95)",
    )
    # ---- Dry run: collect stats only (no writes)
    p.add_argument(
        "--dry-run", action="store_true", help="collect stats only; skip writing outputs to --out"
    )

    # ---- Fast-path filters (file-level early skip) ----
    p.add_argument(
        "--min-file-notes",
        type=int,
        default=0,
        help="Skip files whose total note count is below this threshold (0=disabled)",
    )
    p.add_argument(
        "--min-file-seconds",
        type=float,
        default=0.0,
        help="Skip files shorter than this duration in seconds (0=disabled)",
    )
    p.add_argument(
        "--max-file-seconds",
        type=float,
        default=0.0,
        help="Skip files longer than this duration in seconds (0=disabled)",
    )
    # ---- Lyrics pipeline toggle ----
    p.add_argument(
        "--skip-lyrics",
        action="store_true",
        help="Disable lyrics matching (speeds up corpus build)",
    )
    return p


# silence pretty_midi/pkg_resources deprecation noise
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated.*",
    module="pretty_midi.*",
)


# ---- tiny, picklable config for process workers ----
class FastCfg:
    drums_only: bool = False
    min_file_notes: int = 0
    min_file_seconds: float = 0.0
    max_file_seconds: float = 0.0
    # envelope-based fast filter
    silence_threshold_db: float = -60.0
    silent_fraction: float = 0.95


# --- skip metrics (already added previously) --------------------------
_SKIP_COUNTS: Counter[str] = Counter()
_TEMPO_FALLBACKS = 0

_ARGS: argparse.Namespace | None = None
_BASE: Path | None = None
_TAG_MAP: Dict[str, Dict[str, str]] = {}
_LYRIC_MAP: Dict[str, str] = {}
_EMBED_MAP: Dict[str, List[float]] = {}
_EMBED_MODEL = None


def _inc_skip(reason: str) -> None:
    _SKIP_COUNTS[reason] += 1


# A tiny helper for safe getattr with default
_g = lambda ns, k, d=None: getattr(ns, k, d)


def _worker(args: tuple[Path, FastCfg]) -> List[Sample]:
    """Process worker for multiprocessing; unwraps args."""
    return process_path(*args)


def process_path(midi_path: Path, ns: FastCfg) -> List[Sample]:
    global _TEMPO_FALLBACKS
    # Load file
    try:
        pm = pretty_midi.PrettyMIDI(midi_path.as_posix())
    except Exception:
        _inc_skip("invalid_midi")
        return []
    mid = pm_to_mido(pm)
    beats_per_bar, ts_str = get_time_signature(mid)

    notes_total = 0
    for inst in pm.instruments:
        if ns and getattr(ns, "drums_only", False) and not inst.is_drum:
            continue
        notes_total += len(inst.notes)
    dur_sec = float(pm.get_end_time() or 0.0)

    if ns is not None:
        if getattr(ns, "min_file_notes", 0) and notes_total < ns.min_file_notes:
            _inc_skip("too_few_notes")
            return []
        if getattr(ns, "min_file_seconds", 0.0) and dur_sec > 0.0 and dur_sec < ns.min_file_seconds:
            _inc_skip("too_short")
            return []
        if (
            getattr(ns, "max_file_seconds", 0.0)
            and ns.max_file_seconds > 0.0
            and dur_sec > ns.max_file_seconds
        ):
            _inc_skip("too_long")
            return []

    sec_to_beats, tempo_est, fb = build_beat_map(pm, path=midi_path)
    if fb:
        _TEMPO_FALLBACKS += 1
    rel = normalize_key(midi_path, _BASE or Path("."))
    segments: List[Sample] = []
    include_programs = set(_ARGS.include_programs) if _ARGS.include_programs else None
    for idx, seg in enumerate(
        split_samples(
            pm,
            bars_per_sample=_ARGS.bars_per_sample,
            min_notes=_ARGS.min_notes,
            beats_per_bar=beats_per_bar,
            sec_to_beats=sec_to_beats,
            include_programs=include_programs,
            drums_only=_ARGS.drums_only,
            exclude_drums=_ARGS.exclude_drums,
            max_segments=_ARGS.max_samples_per_file,
        )
    ):
        tokens = tokenize_notes(
            seg,
            duv=_ARGS.duv == "on",
            dur_bins=_ARGS.dur_bins,
            vel_bins=_ARGS.vel_bins,
            quant=_ARGS.quant,
        )
        tags = _TAG_MAP.get(rel, {})
        if _ARGS.section_tokens and tags.get("section"):
            tokens.insert(0, f"<SECTION={tags['section']}>")
        if _ARGS.mood_tokens and tags.get("mood"):
            tokens.insert(0, f"<MOOD={tags['mood']}>")
        meta: Dict[str, object] = {
            **tags,
            "source_path": rel,
            "segment_index": idx,
            "tempo_est": tempo_est,
            "beats_per_bar": beats_per_bar,
            "time_signature": ts_str,
        }
        if rel in _EMBED_MAP:
            meta["text_emb"] = _EMBED_MAP[rel]
        elif (not (ns and getattr(ns, "skip_lyrics", False))) and rel in _LYRIC_MAP:
            text = _LYRIC_MAP[rel]
            if _EMBED_MODEL is not None:
                meta["text_emb"] = _EMBED_MODEL.encode(text).tolist()
            else:
                meta["text"] = text
        segments.append(Sample(tokens=tokens, meta=meta))
    return segments


# -------------- メイン処理 -------------------
def main(argv: list[str] | None = None) -> None:
    # 引数パース
    args = build_argparser().parse_args(argv)

    # ロガー設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("command: %s", " ".join(argv or []))
    logger.info("Python %s", sys.version.split()[0])
    logger.info("prepare_transformer_corpus starting up")

    # 入出力パス
    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    os.makedirs(out_dir, exist_ok=True)

    # 簡易ヘルプ
    if args.tags and not all(Path(p).is_file() for p in args.tags):
        logger.error("一部のタグファイルが見つかりません: %s", args.tags)
        return
    if args.lyric_json and not Path(args.lyric_json).is_file():
        logger.error("歌詞 JSON ファイルが見つかりません: %s", args.lyric_json)
        return

    # MIDI ファイル収集
    files = gather_midi_files(in_dir)

    # I/O バッチング：num-workers > 1 のときシャッフル（ホットスポット回避）
    try:
        if getattr(args, "num_workers", 0) and int(args.num_workers) > 1:
            rnd = random.Random(getattr(args, "seed", None) or 0)
            rnd.shuffle(files)
    except Exception:
        pass

    # dry-run の明示ログ
    if _g(args, "dry_run", False):
        logger.info(
            "dry_run is ON: no dataset files will be written (out=%s)",
            _g(args, "out_dir", None),
        )
        # もし書き出し関数内で環境変数を見て分岐できるなら、フラグも立てておく
        os.environ["PREP_CORPUS_DRY_RUN"] = "1"

    max_files = getattr(args, "max_files", 0) or 0
    if max_files:
        files = files[:max_files]

    # コーパス構築
    splits, lyric_matches, midi_file_count = build_corpus(args, files)
    logger.info("tempo fallback used: %d", _TEMPO_FALLBACKS)

    # ボキャブラリ構築
    vocab = build_vocab(splits["train"])
    tag_vocab = build_tag_vocab(splits["train"])

    # メタデータ収集
    meta = {
        "midi_file_count": midi_file_count,
        "lyric_matches": lyric_matches,
        "vocab_size": len(vocab),
        "tag_vocab_size": len(tag_vocab),
        "split": args.split,
        "duv_max": args.duv_max,
        "bars_per_sample": args.bars_per_sample,
        "quant": args.quant,
        "min_notes": args.min_notes,
        "dur_bins": args.dur_bins,
        "vel_bins": args.vel_bins,
        "embed_offline": bool(args.embed_offline),
        "tags": args.tags,
        "min_file_notes": args.min_file_notes,
        "min_file_seconds": args.min_file_seconds,
        "max_file_seconds": args.max_file_seconds,
        "skipped_too_few_notes": _SKIP_COUNTS.get("too_few_notes", 0),
        "skipped_too_short": _SKIP_COUNTS.get("too_short", 0),
        "skipped_invalid_midi": _SKIP_COUNTS.get("invalid_midi", 0),
        "skipped_too_long": _SKIP_COUNTS.get("too_long", 0),
        "tempo_fallback_used": _TEMPO_FALLBACKS,
    }
    logger.info("meta: %s", json.dumps(meta, ensure_ascii=False, indent=2))

    # サンプル書き出し
    logger.info("writing samples...")
    for split, samples in splits.items():
        out_path = out_dir / f"{split}.jsonl"
        if args.compress == "gz":
            out_path = out_path.with_suffix(out_path.suffix + ".gz")
        save_jsonl(out_path, samples, compress=args.compress)
        logger.info("  %s: %d samples", split, len(samples))

    # ボキャブラリ書き出し
    if args.compress == "gz":
        vocab_path = out_dir / "vocab.json.gz"
    else:
        vocab_path = out_dir / "vocab.json"
    logger.info("writing vocab: %s", vocab_path)
    with (
        gzip.open(vocab_path, "wt", encoding="utf-8")
        if args.compress == "gz"
        else open(vocab_path, "w", encoding="utf-8")
    ) as fh:
        json.dump(vocab, fh, ensure_ascii=False)

    # タグボキャブラリ書き出し
    if args.compress == "gz":
        tag_vocab_path = out_dir / "tag_vocab.json.gz"
    else:
        tag_vocab_path = out_dir / "tag_vocab.json"
    logger.info("writing tag vocab: %s", tag_vocab_path)
    with (
        gzip.open(tag_vocab_path, "wt", encoding="utf-8")
        if args.compress == "gz"
        else open(tag_vocab_path, "w", encoding="utf-8")
    ) as fh:
        json.dump(tag_vocab, fh, ensure_ascii=False)

    logger.info("done.")


if __name__ == "__main__":
    import sys

    main()
