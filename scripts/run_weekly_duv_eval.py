from __future__ import annotations

"""Batch runner for the weekly DUV evaluation suites.

This command stitches together curated subsets defined in a JSON config,
converts them to the note-wise CSV format expected by :mod:`scripts.eval_duv`,
and records metrics for tracking model drift over time.
"""

import argparse
import datetime as _dt
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd

from . import eval_duv


@dataclass(slots=True)
class SubsetSpec:
    """Configuration for a single evaluation subset."""

    name: str
    csv: Path
    song_ids: list[str]
    instrument: str | None
    rows_per_song: int | None
    limit: int | None

    @classmethod
    def from_dict(cls, root: Path, payload: dict[str, Any]) -> "SubsetSpec":
        name = str(payload.get("name"))
        if not name:
            raise ValueError("subset missing 'name'")
        csv_path = Path(payload.get("csv", ""))
        if not csv_path:
            raise ValueError(f"subset '{name}' missing 'csv'")
        if not csv_path.is_absolute():
            csv_path = (root / csv_path).resolve()
        song_ids_raw = payload.get("song_ids")
        if song_ids_raw is None:
            raise ValueError(f"subset '{name}' missing 'song_ids'")
        if isinstance(song_ids_raw, str):
            song_ids = [song_ids_raw]
        elif isinstance(song_ids_raw, Iterable):
            song_ids = [str(x) for x in song_ids_raw]
        else:
            raise TypeError(f"subset '{name}' expected list for 'song_ids'")
        instrument = payload.get("instrument")
        instrument_str = str(instrument) if instrument is not None else None
        rows_per_song = payload.get("rows_per_song")
        rows_val = int(rows_per_song) if rows_per_song is not None else None
        limit = payload.get("limit")
        limit_val = int(limit) if limit is not None else None
        return cls(
            name=name,
            csv=csv_path,
            song_ids=song_ids,
            instrument=instrument_str,
            rows_per_song=rows_val,
            limit=limit_val,
        )


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "position" not in df.columns:
        if "pos" in df.columns:
            df = df.rename(columns={"pos": "position"})
        else:
            raise ValueError("input CSV missing 'position' or 'pos' column")
    if "track_id" not in df.columns:
        key = df.get("song_id")
        if key is None:
            key = df.index.astype(str)
        else:
            key = key.fillna("<unknown>")
        df = df.copy()
        df["track_id"] = pd.factorize(key)[0].astype("int32")
    if "program" not in df.columns:
        df = df.copy()
        df["program"] = -1
    return df


def _filter_subset(df: pd.DataFrame, spec: SubsetSpec) -> pd.DataFrame:
    mask = df["song_id"].isin(spec.song_ids)
    subset = df.loc[mask].copy()
    if spec.instrument:
        subset = subset[subset["instrument"] == spec.instrument].copy()
    if subset.empty:
        raise ValueError(
            "subset '%s' resolved to 0 rows (song_ids=%s, instrument=%s)"
            % (spec.name, spec.song_ids, spec.instrument)
        )
    if spec.rows_per_song:
        subset = (
            subset.groupby("song_id", group_keys=False)
            .head(spec.rows_per_song)
            .reset_index(drop=True)
        )
    if spec.limit:
        subset = subset.head(spec.limit).reset_index(drop=True)
    return subset


def _write_subset_csv(subset: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False)


def _run_eval(
    subset_csv: Path,
    ckpt: Path,
    stats_json: Path | None,
    batch: int,
    device: str,
    num_workers: int,
    verbose: bool,
) -> dict[str, Any]:
    args = argparse.Namespace(
        csv=subset_csv,
        ckpt=ckpt,
        stats_json=stats_json,
        batch=batch,
        device=device,
        dur_quant=None,
        num_workers=num_workers,
        verbose=verbose,
        filter_program=None,
        limit=0,
        out_json=None,
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = eval_duv.run(args)
    if exit_code != 0:
        raise RuntimeError(f"eval_duv failed for {subset_csv} with exit code {exit_code}")
    lines = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"no metrics produced for {subset_csv}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"failed to parse metrics JSON for {subset_csv}: {lines[-1]}") from exc


def _build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Weekly DUV Evaluation Report")
    lines.append("")
    meta = report.get("meta", {})
    lines.append(f"- Generated at: {meta.get('generated_at', 'unknown')}")
    lines.append(f"- Checkpoint: {meta.get('ckpt', 'unknown')}")
    stats_path = meta.get("stats_json")
    lines.append(f"- Stats JSON: {stats_path if stats_path else 'auto'}")
    lines.append("")
    headers = [
        "subset",
        "rows",
        "velocity_mae",
        "velocity_rmse",
        "duration_mae",
        "duration_rmse",
        "velocity_count",
        "duration_count",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + " --- |" * len(headers))
    metrics: list[dict[str, Any]] = report.get("metrics", [])
    for item in metrics:
        row: list[str] = [
            item.get("name", ""),
            str(item.get("rows", 0)),
            _fmt_float(item.get("velocity_mae")),
            _fmt_float(item.get("velocity_rmse")),
            _fmt_float(item.get("duration_mae")),
            _fmt_float(item.get("duration_rmse")),
            str(item.get("velocity_count", "-")),
            str(item.get("duration_count", "-")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _fmt_float(value: Any, precision: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{val:.{precision}f}"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _resolve_stats_path(ckpt: Path, user_override: Path | None) -> Path | None:
    if user_override:
        return user_override
    candidate = ckpt.with_suffix(ckpt.suffix + ".stats.json")
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the weekly DUV evaluation harness")
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the DUV checkpoint (.ckpt or TorchScript)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="Optional path to feature stats JSON",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/duv_weekly_eval.json"),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override passed to eval_duv (default: auto)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="Batch size for eval_duv forward passes",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for eval_duv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Global cap on rows per subset after subset limits",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward verbose flag to eval_duv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only materialise subset CSVs without invoking eval_duv",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    config = _load_config(config_path)

    output_dir_cfg = config.get("output_dir", "outputs/duv_weekly")
    output_root = Path(output_dir_cfg)
    if not output_root.is_absolute():
        output_root = config_path.parent.parent / output_root
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (output_root / timestamp).resolve()
    csv_dir = run_dir / "csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    subsets_cfg_raw = config.get("subsets")
    if not isinstance(subsets_cfg_raw, list) or not subsets_cfg_raw:
        raise SystemExit("config missing non-empty 'subsets' array")

    repo_root = config_path.parent.parent
    specs: list[SubsetSpec] = []
    for entry_raw in subsets_cfg_raw:
        if not isinstance(entry_raw, dict):
            raise SystemExit("subset entries must be objects")
        entry = cast(dict[str, Any], entry_raw)
        specs.append(SubsetSpec.from_dict(repo_root, entry))

    results: list[dict[str, Any]] = []
    stats_path = _resolve_stats_path(args.ckpt.resolve(), args.stats)

    for spec in specs:
        print(f"[weekly-duv] Preparing subset '{spec.name}' from {spec.csv}")
        df = pd.read_csv(str(spec.csv))  # type: ignore[arg-type]
        df = _prepare_dataframe(df)
        subset_df = _filter_subset(df, spec)
        if args.limit:
            subset_df = subset_df.head(args.limit).reset_index(drop=True)
        rows = len(subset_df)
        target_csv = csv_dir / f"{spec.name}.csv"
        _write_subset_csv(subset_df, target_csv)
        metrics: dict[str, Any]
        if args.dry_run:
            print("[weekly-duv] Dry run: skipping eval_duv for '%s' (%d rows)" % (spec.name, rows))
            metrics = {}
        else:
            print(f"[weekly-duv] Evaluating '{spec.name}' ({rows} rows) ...")
            metrics = _run_eval(
                target_csv,
                ckpt=args.ckpt.resolve(),
                stats_json=stats_path,
                batch=args.batch,
                device=args.device,
                num_workers=args.num_workers,
                verbose=args.verbose,
            )
        record: dict[str, Any] = {
            "name": spec.name,
            "rows": rows,
            "csv": str(
                target_csv.relative_to(run_dir)
                if target_csv.is_relative_to(run_dir)
                else target_csv
            ),
            "source_csv": str(spec.csv),
            "song_ids": spec.song_ids,
            "instrument": spec.instrument,
        }
        record.update(metrics)
        results.append(record)

    report: dict[str, Any] = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "ckpt": str(args.ckpt.resolve()),
            "stats_json": str(stats_path) if stats_path else None,
            "config": str(config_path),
            "dry_run": bool(args.dry_run),
            "device": args.device,
            "batch": int(args.batch),
            "num_workers": int(args.num_workers),
            "global_limit": int(args.limit) if args.limit else None,
        },
        "metrics": results,
    }

    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = _build_markdown(report)
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")

    print(f"[weekly-duv] Report written to {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
