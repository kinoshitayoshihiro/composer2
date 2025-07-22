import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from utilities.breath_edit import process_breath
from utilities.breath_mask import infer_breath_mask

logger = logging.getLogger("breath")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Apply breath processing to WAV")
    ap.add_argument("wav", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/render.yaml"))
    ap.add_argument(
        "--breath-mode",
        choices=["keep", "attenuate", "remove"],
        default=None,
    )
    ap.add_argument("--hop-ms", type=float, default=None)
    ap.add_argument("--thr-off", type=float, default=None)
    ap.add_argument("--atten-gain", type=float, default=None)
    ap.add_argument("--percentile", type=float, default=None)
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-level", default=None)
    ns = ap.parse_args(argv)

    cfg: dict[str, Any] = {}
    if ns.config.is_file():
        cfg = yaml.safe_load(ns.config.read_text()) or {}
    rcfg = cfg.get("render", {})
    mode = ns.breath_mode or rcfg.get("breath_mode", "keep")
    gain = float(ns.atten_gain or rcfg.get("attenuate_gain_db", -15))
    xfade = int(rcfg.get("crossfade_ms", 50))
    hop_ms = float(ns.hop_ms or rcfg.get("hop_ms", 10))
    thr_key = (
        "thr_offset_db" if "thr_offset_db" in rcfg else "breath_threshold_offset_db"
    )
    if thr_key == "breath_threshold_offset_db" and "breath_threshold_offset_db" in rcfg:
        import warnings

        warnings.warn(
            "breath_threshold_offset_db is deprecated; use thr_offset_db",
            DeprecationWarning,
        )
    thr_off = float(ns.thr_off or rcfg.get(thr_key, -30))
    percentile = float(ns.percentile or rcfg.get("energy_percentile", 95))
    onnx_path = ns.onnx
    log_level = (ns.log_level or rcfg.get("log_level", "WARN")).upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.WARN))

    mask = infer_breath_mask(
        ns.wav,
        hop_ms=hop_ms,
        thr_offset_db=thr_off,
        percentile=percentile,
        onnx_path=onnx_path,
    )

    out_path = ns.out
    if ns.dry_run:
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        out_path = Path(tmp_name)

    process_breath(ns.wav, out_path, mask, mode, gain, xfade, hop_ms=hop_ms)

    if ns.dry_run:
        from pydub import AudioSegment

        before = AudioSegment.from_file(ns.wav)
        after = AudioSegment.from_file(out_path)
        stats = {
            "mode": mode,
            "in_ms": len(before),
            "out_ms": len(after),
            "rms_before": before.rms,
            "rms_after": after.rms,
        }
        print(json.dumps(stats))
        Path(out_path).unlink(missing_ok=True)
    else:
        logger.info("wrote %s", ns.out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("error during rendering")
        sys.exit(1)
