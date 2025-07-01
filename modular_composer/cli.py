from __future__ import annotations

import argparse
import glob
import importlib
import importlib.metadata as _md
import json
import random
import tempfile
from pathlib import Path
from types import ModuleType
from typing import cast

import click
import pretty_midi

import utilities.loop_ingest as loop_ingest
from utilities import (
    groove_sampler_ngram,
    groove_sampler_rnn,
    live_player,
    streaming_sampler,
    synth,
)

groove_rnn_v2: ModuleType | None
try:
    groove_rnn_v2 = importlib.import_module("utilities.groove_rnn_v2")
except Exception:
    groove_rnn_v2 = None
from utilities.golden import compare_midi, update_golden  # noqa: E402
from utilities.groove_sampler_ngram import Event, State  # noqa: E402
from utilities.groove_sampler_v2 import generate_events, load, save, train  # noqa: F401,E402
from utilities.peak_synchroniser import PeakSynchroniser  # noqa: E402
from utilities.realtime_engine import RealtimeEngine  # noqa: E402
from utilities.tempo_utils import beat_to_seconds  # noqa: E402
from utilities.tempo_utils import load_tempo_curve as load_tempo_curve_simple  # noqa: E402


def _lazy_import_groove_rnn() -> ModuleType | None:
    try:
        import importlib
        return importlib.import_module("utilities.groove_rnn_v2")
    except Exception:
        return None


@click.group()
def cli() -> None:
    """Command group for modular-composer."""


@cli.group()
def groove() -> None:
    """Groove sampler commands."""


groove.add_command(groove_sampler_ngram.train_cmd, name="train")
groove.add_command(groove_sampler_ngram.sample_cmd, name="sample")
groove.add_command(groove_sampler_ngram.info_cmd, name="info")


@cli.group()
def rnn() -> None:
    """RNN groove sampler commands."""
    pass


@rnn.command("train", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def rnn_train(args: tuple[str, ...]) -> None:
    mod = _lazy_import_groove_rnn()
    if mod is None:
        click.echo("RNN extras not installed")
        raise SystemExit(1)
    mod.train_cmd.main(list(args), standalone_mode=False)

if groove_rnn_v2 is not None:
    rnn.add_command(groove_rnn_v2.train_cmd, name="train")
    rnn.add_command(groove_rnn_v2.sample_cmd, name="sample")
else:
    @rnn.command("train")
    def _rnn_missing_train() -> None:
        click.echo("RNN extras not installed")
        raise SystemExit(1)

@rnn.command("sample", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def rnn_sample(args: tuple[str, ...]) -> None:
    mod = _lazy_import_groove_rnn()
    if mod is None:
        click.echo("RNN extras not installed")
        raise SystemExit(1)
    mod.sample_cmd.main(list(args), standalone_mode=False)


@cli.group()
def loops() -> None:
    """Loop ingestion utilities."""


loops.add_command(loop_ingest.scan)
loops.add_command(loop_ingest.info)


@cli.group()
def eval() -> None:
    """Evaluation commands."""


@eval.command("metrics")
@click.argument("midi", type=Path)
@click.option("--ref", "ref_midi", type=Path, default=None)
def eval_metrics(midi: Path, ref_midi: Path | None) -> None:
    """Print basic metrics for ``midi``.

    When ``--ref`` is supplied, also compute ``blec`` between the files.
    """
    import pretty_midi
    
    from eval import metrics

    pm = pretty_midi.PrettyMIDI(str(midi))
    tempo = pm.get_tempo_changes()[1]
    bpm = float(tempo[0]) if getattr(tempo, "size", 0) else 120.0
    beat = 60.0 / bpm
    events = [
        {"offset": n.start / beat, "velocity": n.velocity}
        for inst in pm.instruments
        for n in inst.notes
    ]
    swing = metrics.swing_score(events)
    density = metrics.note_density(events)
    var = metrics.velocity_var(events)
    res = {
        "swing_score": round(swing, 4),
        "note_density": round(density, 4),
        "velocity_var": round(var, 4),
    }
    if ref_midi:
        pm_ref = pretty_midi.PrettyMIDI(str(ref_midi))
        events_ref = [
            {"offset": n.start / beat, "velocity": n.velocity}
            for inst in pm_ref.instruments
            for n in inst.notes
        ]
        res["blec"] = round(metrics.blec_score(events_ref, events), 4)
    click.echo(json.dumps(res))


@eval.command("latency")
@click.argument("model", type=Path)
@click.option("--backend", default="ngram")
def eval_latency(model: Path, backend: str) -> None:
    from eval import latency

    res = latency.evaluate_model(str(model), backend=backend)
    click.echo(str(res))


@eval.command("abx")
@click.argument("human", type=Path)
@click.argument("ai", type=Path)
@click.option("--trials", type=int, default=12, show_default=True)
def eval_abx(human: Path, ai: Path, trials: int) -> None:
    from eval import abx_gui

    abx_gui.run_gui(human, ai, trials=trials)


@cli.group()
def plugin() -> None:
    """Plugin utilities."""


@plugin.command("build")
@click.option("--format", type=click.Choice(["vst3", "clap"]), default="vst3")
@click.option("--out", type=Path, default=Path("build"))
def plugin_build(format: str, out: Path) -> None:
    """Build the JUCE plugin."""
    import subprocess

    out.mkdir(exist_ok=True)
    cfg = ["cmake", "-B", str(out), "-DMODC_BUILD_PLUGIN=ON"]
    if format == "clap":
        cfg.append("-DMODC_CLAP=ON")
    subprocess.check_call(cfg)
    subprocess.check_call(["cmake", "--build", str(out), "--config", "Release"])
try:
    __version__ = _md.version("modular_composer")
except _md.PackageNotFoundError:
    __version__ = "0.0.0"


@cli.command("live")
@click.argument("model", type=Path)
@click.option("--backend", type=click.Choice(["ngram", "rnn"]), default=None)
@click.option("--sync", type=click.Choice(["internal", "external"]), default="internal")
@click.option("--bpm", type=float, default=120.0, show_default=True)
@click.option("--buffer", type=int, default=1, show_default=True)
def live_cmd(model: Path, backend: str | None, sync: str, bpm: float, buffer: int) -> None:
    """Stream a trained groove model live."""
    if backend is None:
        backend = "rnn" if model.suffix == ".pt" else "ngram"
    engine = RealtimeEngine(
        str(model), backend=backend, bpm=bpm, sync=sync, buffer_bars=buffer
    )
    live_player.play_live(engine, bpm=bpm)  # type: ignore[arg-type]


def _cmd_demo(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose demo")
    ap.add_argument("-o", "--out", type=Path, default=Path("demo.mid"))
    ap.add_argument("--tempo-curve", type=Path)
    ns = ap.parse_args(args)

    curve = []
    if ns.tempo_curve:
        curve = load_tempo_curve_simple(ns.tempo_curve)

    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=curve[0]["bpm"] if curve else 120)
    inst = pretty_midi.Instrument(program=0)
    for i in range(16):
        start = beat_to_seconds(float(i), curve)
        end = beat_to_seconds(float(i + 1), curve)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=start, end=end))
    pm.instruments.append(inst)
    pm.write(str(ns.out))
    print(f"[demo] wrote {ns.out}")


def _cmd_sample(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose sample")
    ap.add_argument("model", type=Path)
    ap.add_argument("-l", "--length", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond-velocity", choices=["soft", "hard"], default=None)
    ap.add_argument(
        "--cond-kick", choices=["four_on_floor", "sparse"], default=None
    )
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--peaks", type=Path)
    ap.add_argument("--lag", type=float, default=10.0)
    ap.add_argument("--tempo-curve", type=Path)
    ns = ap.parse_args(args)
    model = load(ns.model)
    ev = cast(
        list[Event],
        generate_events(
            model,
            bars=ns.length,
            temperature=ns.temperature,
            seed=ns.seed,
            cond_velocity=ns.cond_velocity,
            cond_kick=ns.cond_kick,
        ),
    )
    if ns.peaks:
        import json

        with ns.peaks.open() as fh:
            peaks = json.load(fh)
        ev = PeakSynchroniser.sync_events(
            peaks,
            ev,
            tempo_bpm=120.0,
            lag_ms=ns.lag,
        )
    import json
    import sys

    if ns.out is None:
        json.dump(ev, sys.stdout)
    else:
        with ns.out.open("w") as fh:
            json.dump(ev, fh)


def _cmd_peaks(args: list[str]) -> None:
    """Wrapper around :func:`utilities.consonant_extract.main`."""
    from utilities import consonant_extract

    consonant_extract.main(args)


def _cmd_render(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose render")
    ap.add_argument("spec", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out.mid"))
    ap.add_argument("--soundfont", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ns = ap.parse_args(args)

    if ns.spec.suffix.lower() in {".yml", ".yaml"}:
        import yaml

        with ns.spec.open("r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh) or {}
    else:
        with ns.spec.open("r", encoding="utf-8") as fh:
            spec = json.load(fh)

    tempo_curve = spec.get("tempo_curve", [])
    events = spec.get("drum_pattern", [])
    peaks = spec.get("peaks", [])
    if peaks:
        events = PeakSynchroniser.sync_events(
            peaks,
            cast(list[Event], events),
            tempo_bpm=spec.get("tempo_bpm", 120),
            lag_ms=10.0,
        )

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_curve[0]["bpm"] if tempo_curve else 120)
    inst = pretty_midi.Instrument(program=0, name="drums")
    pitch_map = {"kick": 36, "snare": 38, "hh_pedal": 44, "ohh": 46}
    for ev in events:
        start = beat_to_seconds(float(ev.get("offset", 0.0)), tempo_curve)
        dur = float(ev.get("duration", 0.25))
        end = start + beat_to_seconds(dur, tempo_curve) - beat_to_seconds(0, tempo_curve)
        pitch = pitch_map.get(ev.get("instrument", "kick"), 60)
        vel = int(ev.get("velocity", 100))
        inst.notes.append(pretty_midi.Note(start=start, end=end, pitch=pitch, velocity=vel))
    pm.instruments.append(inst)
    pm.write(str(ns.out))
    print(f"Wrote {ns.out}")
    if ns.soundfont:
        wav = ns.out.with_suffix(".wav")
        synth.render_midi(ns.out, wav, ns.soundfont)
        print(f"Rendered {wav}")


def _cmd_realtime(args: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="modcompose realtime",
        description=(
            "Stream a trained groove model in real time. "
            "Example: modcompose realtime rnn.pt --bpm 100 --duration 16"
        ),
    )
    ap.add_argument("model", type=Path)
    ap.add_argument("--bpm", type=float, default=100.0)
    ap.add_argument("--duration", type=int, default=16)
    ns = ap.parse_args(args)

    if ns.model.suffix == ".pt":
        model, meta = groove_sampler_rnn.load(ns.model)
        class _WrapR:
            def __init__(self) -> None:
                self.history: list[State] = []
            def feed_history(self, events: list[State]) -> None:
                self.history.extend(events)
            def next_step(self, *, cond: dict[str, object] | None, rng: random.Random) -> Event:
                return groove_sampler_rnn.sample(model, meta, bars=1, temperature=1.0, rng=rng)[0]
        sampler: streaming_sampler.BaseSampler = _WrapR()
    else:
        m = groove_sampler_ngram.load(ns.model)
        class _WrapN:
            def __init__(self) -> None:
                self.hist: list[State] = []
                self.buf: list[Event] = []
            def feed_history(self, events: list[State]) -> None:
                self.hist.extend(events)
            def next_step(self, *, cond: dict[str, object] | None, rng: random.Random) -> Event:
                if not self.buf:
                    self.buf, self.hist = groove_sampler_ngram._generate_bar(
                        self.hist, m, rng=rng
                    )
                return self.buf.pop(0)
        sampler = _WrapN()

    player = streaming_sampler.RealtimePlayer(sampler, bpm=ns.bpm)
    player.play(bars=ns.duration // 4)


def _cmd_gm_test(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose gm-test")
    ap.add_argument("midi", nargs="+")
    ap.add_argument("--update", action="store_true")
    ns = ap.parse_args(args)

    mismatched: list[str] = []
    for pattern in ns.midi:
        for path_str in glob.glob(pattern):
            path = Path(path_str)
            if path.stat().st_size == 0:
                print(f"[gm-test] skipping empty file {path}")
                continue
            pm = pretty_midi.PrettyMIDI(str(path))
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / path.name
                pm.write(str(tmp))
                if not compare_midi(path, tmp):
                    if ns.update:
                        update_golden(tmp, path)
                    else:
                        failed_dir = path.parent / "failed"
                        failed_dir.mkdir(exist_ok=True)
                        new_path = failed_dir / f"{path.stem}_new.mid"
                        update_golden(tmp, new_path)
                        mismatched.append(str(path))
    if mismatched and not ns.update:
        for m in mismatched:
            print(f"Mismatch: {m}")
        raise SystemExit(1)
    print("All golden MIDI match.")


def _cmd_gui(args: list[str]) -> None:
    """Launch the Streamlit GUI."""
    import subprocess
    script = Path(__file__).resolve().parent.parent / "streamlit_app" / "gui.py"
    subprocess.run(["streamlit", "run", str(script), *args], check=True)


def _cmd_tag(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose tag")
    ap.add_argument("loops", type=Path)
    ap.add_argument("--out", type=Path, default=Path("meta.json"))
    ap.add_argument("--k-intensity", type=int, default=3)
    ap.add_argument("--csv", type=Path, default=None)
    ns = ap.parse_args(args)
    from data_ops.auto_tag import auto_tag

    meta = auto_tag(ns.loops, k_intensity=ns.k_intensity, csv_path=ns.csv)
    ns.out.write_text(json.dumps(meta))
    print(f"wrote {ns.out}")


def _cmd_augment(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose augment")
    ap.add_argument("midi", type=Path)
    ap.add_argument("--swing", type=float, default=0.0)
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--shuffle", type=float, default=0.0)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ns = ap.parse_args(args)
    from data_ops import augment

    pm = pretty_midi.PrettyMIDI(str(ns.midi))
    pm = augment.apply_pipeline(
        pm,
        swing_ratio=ns.swing if ns.swing else None,
        shuffle_prob=ns.shuffle if ns.shuffle else None,
        transpose_amt=ns.transpose,
    )
    pm.write(str(ns.out))
    print(f"wrote {ns.out}")


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        cli.main(args=[], standalone_mode=False)
        return
    cmd = argv[0]
    if cmd == "demo":
        _cmd_demo(argv[1:])
    elif cmd == "sample":
        _cmd_sample(argv[1:])
    elif cmd == "peaks":
        _cmd_peaks(argv[1:])
    elif cmd == "render":
        _cmd_render(argv[1:])
    elif cmd == "realtime":
        _cmd_realtime(argv[1:])
    elif cmd == "gm-test":
        _cmd_gm_test(argv[1:])
    elif cmd == "gui":
        _cmd_gui(argv[1:])
    elif cmd == "tag":
        _cmd_tag(argv[1:])
    elif cmd == "augment":
        _cmd_augment(argv[1:])
    else:
        cli.main(args=argv, standalone_mode=False)


if __name__ == "__main__":
    main()
