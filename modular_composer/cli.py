from __future__ import annotations

import argparse
import asyncio
import glob
import importlib
import importlib.metadata as _md
import json
import pickle
import random
import tempfile
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import click
import pretty_midi
import yaml  # type: ignore
from music21 import stream as m21stream

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


def _schema_to_swing(schema: str | None) -> float | None:
    """Map rhythm schema token to swing ratio."""
    if not schema:
        return None
    table = {
        "<straight8>": 0.5,
        "<swing16>": 0.66,
        "<shuffle>": 0.75,
    }
    return table.get(schema)


from utilities.tempo_utils import beat_to_seconds  # noqa: E402
from utilities.tempo_utils import load_tempo_curve as load_tempo_curve_simple  # noqa: E402


def _lazy_import_groove_rnn() -> ModuleType | None:
    try:
        import importlib

        mod = importlib.import_module("utilities.groove_rnn_v2")
    except Exception:
        return None
    if getattr(mod, "pl", None) is None or getattr(mod, "torch", None) is None:
        return None
    return mod


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
        raise click.ClickException("Install extras: rnn")
    mod.train_cmd.main(list(args), standalone_mode=False)


@rnn.command("sample", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def rnn_sample(args: tuple[str, ...]) -> None:
    mod = _lazy_import_groove_rnn()
    if mod is None:
        raise click.ClickException("Install extras: rnn")
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


@cli.group()
def preset() -> None:
    """Preset management commands."""


@preset.command("list")
def preset_list() -> None:
    from utilities import preset_manager

    for name in preset_manager.list_presets():
        click.echo(name)


@preset.command("export")
@click.argument("name")
@click.option("--out", type=Path, required=True)
def preset_export(name: str, out: Path) -> None:
    from utilities import preset_manager

    data = preset_manager.load_preset(name)
    if out.suffix.lower() in {".yml", ".yaml"}:
        yaml.safe_dump(data, out.open("w", encoding="utf-8"))
    else:
        json.dump(data, out.open("w", encoding="utf-8"))


@preset.command("import")
@click.argument("file", type=Path)
@click.option("--name", type=str, default=None)
def preset_import(file: Path, name: str | None) -> None:
    from utilities import preset_manager

    with file.open("r", encoding="utf-8") as fh:
        if file.suffix.lower() in {".yml", ".yaml"}:
            cfg = yaml.safe_load(fh) or {}
        else:
            cfg = json.load(fh)
    preset_manager.save_preset(name or file.stem, cfg)


@cli.group()
def fx() -> None:
    """Effects and rendering commands."""


@fx.command("render")
@click.argument("midi", type=Path)
@click.option("--preset", required=True, type=str)
@click.option("-o", "--out", required=True, type=Path)
@click.option("--soundfont", type=Path, default=None)
def fx_render(
    midi: Path,
    preset: str,
    out: Path,
    soundfont: Path | None,
) -> None:
    """Render ``midi`` to ``out`` applying the preset's IR."""
    from utilities import synth
    from utilities.tone_shaper import ToneShaper

    ts = ToneShaper.from_yaml(Path("data/amp_presets.yml"))
    ir = ts.ir_map.get(preset)
    synth.export_audio(midi, out, soundfont=soundfont, ir_file=ir)
    click.echo(str(out))


@fx.command("list-presets")
def fx_list_presets() -> None:
    """List available amp presets."""
    from utilities.tone_shaper import ToneShaper

    ts = ToneShaper.from_yaml(Path("data/amp_presets.yml"))
    for name in ts.preset_map:
        click.echo(name)


@fx.command("cc")
def fx_cc() -> None:
    """Output preset CC mapping as JSON."""
    from utilities.tone_shaper import ToneShaper

    ts = ToneShaper.from_yaml(Path("data/amp_presets.yml"))
    click.echo(json.dumps(ts.preset_map))


@cli.command("live")
@click.argument("model", type=Path)
@click.option(
    "--backend",
    type=click.Choice(["ngram", "rnn", "realtime", "piano_ml"]),
    default="ngram",
)
@click.option("--stream", type=click.Choice(["rtmidi"]), default="rtmidi", show_default=True)
@click.option(
    "--ai-backend",
    type=click.Choice(["ngram", "rnn", "transformer"]),
    default="ngram",
    show_default=True,
)
@click.option("--model-name", type=str, default="gpt2-music")
@click.option("--use-history", is_flag=True, default=False)
@click.option("--sync", type=click.Choice(["internal", "external"]), default="internal")
@click.option("--bpm", type=float, default=120.0, show_default=True)
@click.option("--buffer", type=int, default=1, show_default=True)
@click.option("--buffer-ahead", type=int, default=0, show_default=True)
@click.option("--parallel-bars", type=int, default=1, show_default=True)
@click.option("--port", type=str, default=None)
@click.option("--latency-buffer", type=float, default=5.0, show_default=True)
@click.option("--measure-latency", is_flag=True, default=False)
@click.option("--late-humanize", is_flag=True, default=False)
@click.option("--lufs-hud", is_flag=True, default=False)
@click.option("--kick-leak-jitter", type=int, default=0, show_default=True)
@click.option("--expr-curve", type=str, default="cubic-in", show_default=True)
@click.option("--rhythm-schema", type=str, default=None)
def live_cmd(
    model: Path,
    backend: str,
    stream: str,
    ai_backend: str,
    model_name: str,
    use_history: bool,
    sync: str,
    bpm: float,
    buffer: int,
    buffer_ahead: int,
    parallel_bars: int,
    port: str | None,
    latency_buffer: float,
    measure_latency: bool,
    late_humanize: bool,
    lufs_hud: bool,
    kick_leak_jitter: int,
    expr_curve: str,
    rhythm_schema: str | None,
) -> None:
    """Stream a trained groove model live."""
    if ai_backend == "transformer":
        from utilities.ai_sampler import TransformerBassGenerator
        from utilities.user_history import load_history, record_generate

        gen = TransformerBassGenerator(model_name, rhythm_schema=rhythm_schema)
        prompt_events: list[dict[str, Any]] = []
        if use_history:
            hist = load_history()
            all_ev = [ev for h in hist for ev in h.get("events", [])]
            prompt_events = all_ev[-128:]
        events = gen.generate(prompt_events, 4)
        for ev in events:
            click.echo(json.dumps(ev))
        if use_history:
            record_generate({"model_name": model_name}, events)
        return
    if backend == "realtime":
        from music21 import converter

        from utilities.rt_midi_streamer import RtMidiStreamer

        ports = RtMidiStreamer.list_ports()
        if port is None:
            if not ports:
                raise click.ClickException("No MIDI output ports")
            click.echo("Available MIDI ports:")
            for idx, name in enumerate(ports):
                click.echo(f"{idx}: {name}")
            return
        streamer = RtMidiStreamer(
            port,
            bpm=bpm,
            buffer_ms=latency_buffer,
            measure_latency=measure_latency,
        )
        parsed = converter.parse(str(model))
        part_stream = parsed.parts[0] if hasattr(parsed, "parts") else parsed
        part = cast(m21stream.Part, part_stream)
        if buffer_ahead > 0:

            async def _run() -> None:
                await streamer.play_live(
                    lambda _i: part,
                    buffer_ahead=buffer_ahead,
                    parallel_bars=parallel_bars,
                    late_humanize=late_humanize,
                )

            asyncio.run(_run())
        else:
            asyncio.run(streamer.play_stream(part, late_humanize=late_humanize))
        if measure_latency:
            stats = streamer.latency_stats() or {}
            click.echo(
                f"Latency mean {stats.get('mean_ms', 0.0):.1f} ms, "
                f"jitter {stats.get('stdev_ms', 0.0):.1f} ms"
            )
        return
    elif backend == "piano_ml" and stream == "rtmidi":
        from generator.piano_ml_generator import PianoMLGenerator
        from realtime import rtmidi_streamer as rtms

        ports = rtms.RtMidiStreamer.list_ports()
        if port is None:
            if not ports:
                raise click.ClickException("No MIDI output ports")
            click.echo("Available MIDI ports:")
            for idx, name in enumerate(ports):
                click.echo(f"{idx}: {name}")
            return
        gen = PianoMLGenerator(str(model))
        streamer = rtms.RtMidiStreamer(port, gen)
        streamer.start(bpm=bpm, buffer_bars=buffer, callback=None)
        try:
            while True:
                streamer.on_tick()
        except KeyboardInterrupt:
            pass
        streamer.stop()
        return

    if RealtimeEngine is None:
        raise click.ClickException("Realtime engine unavailable")
    if backend == "rnn" and _lazy_import_groove_rnn() is None:
        raise click.ClickException("Install extras: rnn")
    try:
        engine = RealtimeEngine(
            str(model),
            backend=backend,
            bpm=bpm,
            sync=sync,
            buffer_bars=buffer,
            swing_ratio=_schema_to_swing(rhythm_schema),
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    meter = None
    hud_thread = None
    if lufs_hud:
        from utilities.loudness_meter import RealtimeLoudnessMeter

        meter = RealtimeLoudnessMeter()
        meter.start(None)

        def _hud() -> None:
            import time

            while meter is not None:
                print(f"LUFS: {meter.get_current_lufs():.1f}", end="\r")
                time.sleep(0.5)

        import threading

        hud_thread = threading.Thread(target=_hud, daemon=True)
        hud_thread.start()

    live_player.play_live(engine, bpm=bpm)  # type: ignore[arg-type]
    if meter is not None:
        meter.stop()
    if hud_thread is not None:
        hud_thread.join(timeout=0.2)


def _cmd_demo(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose demo")
    ap.add_argument("-o", "--out", type=Path, default=Path("demo.mid"))
    ap.add_argument("--tempo-curve", type=Path)
    ap.add_argument("--seed", type=int, default=None)
    ns = ap.parse_args(args)
    if ns.seed is not None:
        random.seed(ns.seed)

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
    ap.add_argument(
        "--backend",
        choices=[
            "ngram",
            "rnn",
            "transformer",
            "piano_template",
            "piano_ml",
            "vocal",
        ],
        default="ngram",
    )
    ap.add_argument(
        "--ai-backend",
        dest="ai_backend",
        choices=["ngram", "rnn", "transformer", "piano_template", "piano_ml"],
        help=argparse.SUPPRESS,
        default=None,
    )
    ap.add_argument("--model-name", type=str, default="gpt2-medium")
    ap.add_argument("--use-history", action="store_true", default=False)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cond-velocity", choices=["soft", "hard"], default=None)
    ap.add_argument("--cond-kick", choices=["four_on_floor", "sparse"], default=None)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--peaks", type=Path)
    ap.add_argument("--lag", type=float, default=10.0)
    ap.add_argument("--tempo-curve", type=Path)
    ap.add_argument("--phoneme-dict", type=Path, default=None)
    ap.add_argument("--rhythm-schema", type=str, default=None)
    ap.add_argument("--humanize-profile", type=str, default=None)
    ap.add_argument("--voicing", choices=["shell", "guide", "drop2"], default="shell")
    ap.add_argument(
        "--intensity",
        choices=["low", "medium", "high"],
        default="medium",
        help="Overall intensity level",
    )
    ap.add_argument(
        "--counterline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a simple counter melody",
    )
    ap.add_argument(
        "--tone-preset",
        choices=["grand_clean", "upright_mellow", "ep_phase"],
        default=None,
        help="Select piano/bass tone preset",
    )
    ap.add_argument(
        "--vibrato-depth",
        type=float,
        default=0.5,
        help="Vocal vibrato depth in semitones",
    )
    ap.add_argument(
        "--vibrato-rate",
        type=float,
        default=5.0,
        help="Vocal vibrato rate in cycles per quarter note",
    )
    ap.add_argument(
        "--enable-articulation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gliss/trill tags",
    )
    ns = ap.parse_args(args)
    if ns.tone_preset:
        backends = str(ns.backend).split(",")
        if any("bass" in b for b in backends):
            setattr(ns, "tone_preset", ns.tone_preset)
    if ns.ai_backend:
        warnings.warn("--ai-backend is deprecated; use --backend", DeprecationWarning)
        ns.backend = ns.ai_backend
    if ns.seed is not None:
        random.seed(ns.seed)
    if ns.backend == "transformer":
        from utilities.user_history import load_history, record_generate

        if ns.use_history:
            load_history()
        from generator.bass_generator import sample_transformer_bass

        try:
            events = sample_transformer_bass(
                ns.model_name,
                ns.length,
                temperature=ns.temperature,
                rhythm_schema=ns.rhythm_schema,
            )
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        if ns.use_history:
            record_generate({"model_name": ns.model_name}, events)
    elif ns.backend == "piano_template":
        from music21 import instrument

        from generator.piano_template_generator import PPQ, PianoTemplateGenerator

        gen = PianoTemplateGenerator(
            part_name="piano",
            default_instrument=instrument.Piano(),
            global_tempo=120,
            global_time_signature="4/4",
            global_key_signature_tonic="C",
            global_key_signature_mode="major",
            global_settings={"humanize_profile": ns.humanize_profile},
            enable_articulation=ns.enable_articulation,
            tone_preset=ns.tone_preset,
        )
        section = {
            "q_length": float(ns.length) * 4.0,
            "chord_symbol_for_voicing": "C",
            "groove_kicks": [i * 2.0 for i in range(int(ns.length))],
            "musical_intent": {"intensity": ns.intensity},
            "voicing_mode": ns.voicing,
            "use_pedal": True,
        }
        parts = gen.compose(section_data=section)
        if ns.counterline and isinstance(parts, dict) and "piano_rh" in parts:
            from generator.counter_line import CounterLineGenerator

            counter = CounterLineGenerator().generate(parts["piano_rh"])
            parts["counterline"] = counter

        def _events(p, hand: str):
            ev = []
            for n in p.flatten().notes:
                ev.append(
                    {
                        "pitch": int(n.pitch.midi),
                        "velocity": int(n.volume.velocity or 64),
                        "offset": float(n.offset),
                        "duration": float(n.duration.quarterLength),
                        "hand": hand,
                        "pedal": any(
                            abs(float(n.offset) - cc["time"]) < (3 / PPQ)
                            and cc["val"] > 0
                            for cc in getattr(p, "extra_cc", [])
                            if cc.get("cc") == 64
                        ),
                    }
                )
            return ev

        events = []
        if isinstance(parts, dict):
            for hid, p in parts.items():
                if "rh" in hid:
                    hand = "RH"
                elif "lh" in hid:
                    hand = "LH"
                else:
                    hand = "CL"
                events.extend(_events(p, hand))
        else:
            events.extend(_events(parts, "RH"))
        events.sort(key=lambda e: e["offset"])
    elif ns.backend == "piano_ml":
        from generator.piano_ml_generator import PianoMLGenerator

        gen = PianoMLGenerator(str(ns.model), temperature=ns.temperature)
        events = gen.generate(max_bars=ns.length)
    elif ns.backend == "vocal":
        from generator.vocal_generator import VocalGenerator

        gen = VocalGenerator(
            phoneme_dict_path=ns.phoneme_dict,
            vibrato_depth=ns.vibrato_depth,
            vibrato_rate=ns.vibrato_rate,
            enable_articulation=ns.enable_articulation,
        )
        part = gen.compose(
            [
                {
                    "offset": 0.0,
                    "pitch": "C4",
                    "length": 1.0,
                    "velocity": 80,
                }
            ],
            processed_chord_stream=[],
            humanize_opt=False,
            lyrics_words=["あ"],
        )
        events = gen.extract_phonemes(part)
    else:
        model = load(ns.model)
        events = cast(
            list[dict[str, Any]],
            generate_events(
                model,
                bars=ns.length,
                temperature=ns.temperature,
                seed=ns.seed,
                cond_velocity=ns.cond_velocity,
                cond_kick=ns.cond_kick,
            ),
        )
        ratio = _schema_to_swing(ns.rhythm_schema)
        if ratio is not None:
            from utilities.humanizer import swing_offset

            for ev in events:
                ev["offset"] = swing_offset(float(ev.get("offset", 0.0)), ratio)
    if ns.peaks:
        import json

        with ns.peaks.open() as fh:
            peaks = json.load(fh)
        events = cast(
            list[dict[str, Any]],
            PeakSynchroniser.sync_events(
                peaks,
                cast(list[Event], events),
                tempo_bpm=120.0,
                lag_ms=ns.lag,
            ),
        )
    import json
    import sys

    if ns.out is None:
        json.dump(events, sys.stdout)
    else:
        with ns.out.open("w") as fh:
            json.dump(events, fh)


def _cmd_peaks(args: list[str]) -> None:
    """Wrapper around :func:`utilities.consonant_extract.main`."""
    from utilities import consonant_extract

    consonant_extract.main(args)


def _cmd_render(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose render")
    ap.add_argument("spec", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out.mid"))
    ap.add_argument("--soundfont", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--velocity-hist", type=Path, default=None)
    ap.add_argument("--ema-alpha", type=float, default=0.1)
    ap.add_argument("--humanize-timing", type=float, default=0.0)
    ap.add_argument("--humanize-velocity", type=float, default=0.0)
    ap.add_argument("--expr-curve", type=str, default="cubic-in")
    ap.add_argument("--kick-leak-jitter", type=int, default=0)
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--normalize-lufs", type=float, default=None)
    ns = ap.parse_args(args)

    if ns.spec.suffix.lower() in {".yml", ".yaml"}:
        import yaml

        with ns.spec.open("r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh) or {}
    else:
        with ns.spec.open("r", encoding="utf-8") as fh:
            spec = json.load(fh)

    if ns.preset:
        from utilities import preset_manager

        preset_cfg = preset_manager.load_preset(ns.preset)
        spec.update(preset_cfg)

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

    if ns.velocity_hist:
        with open(ns.velocity_hist, "rb") as fh:
            hist = pickle.load(fh)
        choices = [int(v) for v in hist.keys()]
        weights = [float(w) for w in hist.values()]
        for n in inst.notes:
            target = random.choices(choices, weights)[0]
            n.velocity = int(
                n.velocity * (1 - ns.humanize_velocity) + target * ns.humanize_velocity
            )

    if ns.humanize_timing > 0:
        inst.notes.sort(key=lambda n: n.start)
        if inst.notes:
            ema = inst.notes[0].start
            alpha = float(ns.ema_alpha)
            for n in inst.notes:
                ema += alpha * (n.start - ema)
                shift = (ema - n.start) * ns.humanize_timing
                n.start += shift
                n.end += shift
    pm.instruments.append(inst)
    pm.write(str(ns.out))
    print(f"Wrote {ns.out}")
    if ns.soundfont:
        wav = ns.out.with_suffix(".wav")
        synth.render_midi(ns.out, wav, ns.soundfont)
        if ns.normalize_lufs is not None:
            from utilities.loudness_normalizer import normalize_wav

            normalize_wav(
                wav,
                section="chorus",
                target_lufs_map={"chorus": float(ns.normalize_lufs)},
            )
        print(f"Rendered {wav}")


def _cmd_ir_render(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose ir-render")
    ap.add_argument("midi", type=Path)
    ap.add_argument("ir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out.wav"))
    ap.add_argument("--sf2", type=Path, default=None)
    ap.add_argument("--part", choices=["auto", "guitar", "strings"], default="auto")
    ns = ap.parse_args(args)
    import importlib.util
    import sys
    from utilities.audio_env import has_fluidsynth

    if not has_fluidsynth() or importlib.util.find_spec("soxr") is None:
        print("fluidsynth and soxr are required for IR rendering", file=sys.stderr)
        raise SystemExit(78)

    part_type = ns.part
    if part_type == "auto":
        name = ns.midi.stem.lower()
        if any(k in name for k in ["violin", "cello", "viola", "bass", "strings"]):
            part_type = "strings"
        else:
            part_type = "guitar"

    if part_type == "strings":
        from generator.strings_generator import StringsGenerator
        from music21 import converter

        gen = StringsGenerator(global_settings={}, part_name="strings")
        score = converter.parse(ns.midi)
        parts = {p.id or f"p{i}": p for i, p in enumerate(score.parts)}
        gen._last_parts = parts  # type: ignore[attr-defined]
        gen._last_section = {"section_name": ns.midi.stem}  # type: ignore[attr-defined]
        gen.export_audio(ir_name=str(ns.ir), out_path=ns.out, sf2=str(ns.sf2) if ns.sf2 else None)
    else:
        from utilities.convolver import render_wav

        render_wav(str(ns.midi), str(ns.ir), str(ns.out), sf2=str(ns.sf2) if ns.sf2 else None)
    print(f"wrote {ns.out}")


def _cmd_meter(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose meter")
    ap.add_argument("--device", type=str, default=None)
    ns = ap.parse_args(args)
    from utilities.loudness_meter import RealtimeLoudnessMeter

    meter = RealtimeLoudnessMeter()
    meter.start(ns.device)
    try:
        import time

        while True:
            lufs = meter.get_current_lufs()
            print(f"LUFS: {lufs:.1f}", end="\r")
            time.sleep(0.5)
    except KeyboardInterrupt:
        meter.stop()


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
                    self.buf, self.hist = groove_sampler_ngram._generate_bar(self.hist, m, rng=rng)
                return self.buf.pop(0)

        sampler = _WrapN()

    player = streaming_sampler.RealtimePlayer(sampler, bpm=ns.bpm)
    player.play(bars=ns.duration // 4)


def _cmd_interact(args: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="modcompose interact")
    ap.add_argument(
        "--backend",
        choices=["ngram", "rnn", "transformer"],
        default="ngram",
    )
    ap.add_argument("--model-name", default="gpt2-medium")
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--midi-in", default="")
    ap.add_argument("--midi-out", default="")
    ap.add_argument("--buffer-ms", type=int, default=100)
    ap.add_argument("--lookahead-bars", type=float, default=0.5)
    ns = ap.parse_args(args)
    if ns.backend == "transformer":
        from utilities import interactive_engine
        from utilities.interactive_engine import TransformerInteractiveEngine

        if interactive_engine.mido is None:
            raise click.ClickException("mido not installed")

        try:
            engine = TransformerInteractiveEngine(
                model_name=ns.model_name,
                bpm=ns.bpm,
                buffer_ms=ns.buffer_ms,
                lookahead_bars=ns.lookahead_bars,
            )
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        if RealtimeEngine is None:
            raise RuntimeError("Realtime engine unavailable")
        engine = RealtimeEngine(
            ns.model_name,
            backend=ns.backend,
            bpm=ns.bpm,
            buffer_bars=max(1, int(ns.lookahead_bars)),
            swing_ratio=_schema_to_swing(ns.rhythm_schema),
        )
    engine.add_callback(lambda ev: print(json.dumps(ev)))
    import asyncio

    asyncio.run(engine.start(ns.midi_in, ns.midi_out))


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
    elif cmd == "ir-render":
        _cmd_ir_render(argv[1:])
    elif cmd == "meter":
        _cmd_meter(argv[1:])
    elif cmd == "realtime":
        _cmd_realtime(argv[1:])
    elif cmd == "interact":
        _cmd_interact(argv[1:])
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
