#!/usr/bin/env python3
"""Aggregate statistics from an enriched manifest JSONL file."""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping

LOGGER = logging.getLogger(__name__)

_NUMERIC_FIELDS: tuple[str, ...] = (
    "tempo",
    "tempo_count",
    "beats",
    "notes",
    "duration_sec",
    "avg_note_duration",
    "time_signature_count",
)


def read_manifest(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            yield json.loads(line)


def aggregate(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    totals = 0
    failure_count = 0
    values: Dict[str, List[float]] = {field: [] for field in _NUMERIC_FIELDS}
    time_sigs: Counter[str] = Counter()

    for record in records:
        totals += 1
        meta = record.get("meta", {})  # type: ignore[assignment]
        if not isinstance(meta, MutableMapping):
            meta = {}
        error = meta.get("error")
        if error:
            failure_count += 1
            continue

        for field in _NUMERIC_FIELDS:
            value = meta.get(field)
            if isinstance(value, (int, float)):
                values[field].append(float(value))
        time_sig = meta.get("time_signature")
        if isinstance(time_sig, str) and time_sig:
            time_sigs[time_sig] += 1

    processed = totals - failure_count

    summary: Dict[str, Any] = {
        "total_records": totals,
        "with_stats": processed,
        "error_records": failure_count,
    }

    numeric_summary: Dict[str, Any] = {}
    for field, samples in values.items():
        if not samples:
            continue
        numeric_summary[field] = {
            "count": len(samples),
            "min": min(samples),
            "max": max(samples),
            "mean": statistics.fmean(samples),
            "median": statistics.median(samples),
        }
    summary["numeric"] = numeric_summary

    summary["time_signatures"] = dict(time_sigs.most_common())
    summary["time_signature_unique"] = len(time_sigs)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise an enriched JSONL manifest")
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to enriched JSONL file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write JSON summary (defaults to stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def dump_summary(summary: Mapping[str, Any], out_path: Path | None, pretty: bool) -> None:
    if out_path is None:
        text = json.dumps(
            summary,
            ensure_ascii=False,
            indent=2 if pretty else None,
        )
        print(text)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as writer:
        json.dump(
            summary,
            writer,
            ensure_ascii=False,
            indent=2 if pretty else None,
        )
        writer.write("\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not args.manifest.exists():
        raise SystemExit(f"Manifest file not found: {args.manifest}")

    summary = aggregate(read_manifest(args.manifest))
    LOGGER.info(
        "Records=%d with_stats=%d errors=%d",
        summary.get("total_records", 0),
        summary.get("with_stats", 0),
        summary.get("error_records", 0),
    )
    dump_summary(summary, args.out, args.pretty)


if __name__ == "__main__":
    main()
