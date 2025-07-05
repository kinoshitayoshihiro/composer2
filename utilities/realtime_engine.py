from __future__ import annotations

import random
import threading
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import click

try:  # optional dependency
    import torch
except (ModuleNotFoundError, OSError):  # lightweight install or missing libs
    torch: Any | None = None  # type: ignore[assignment]
else:
    torch: Any | None
from . import groove_sampler_rnn

try:
    from .groove_rnn_v2 import GrooveRNN, sample_rnn_v2
except ImportError:
    GrooveRNN = None  # type: ignore[assignment]
    sample_rnn_v2 = None  # type: ignore[assignment]
from . import groove_sampler_ngram
from .groove_sampler_ngram import sample as sample_ngram
from .streaming_sampler import BaseSampler
from .live_buffer import LiveBuffer

try:  # optional dependency
    import mido
except Exception as e:  # pragma: no cover - optional dependency
    mido = None  # type: ignore
    _MIDO_ERROR = e
else:
    _MIDO_ERROR = None

Event = dict[str, object]


class _RnnSampler:
    def __init__(self, model: groove_sampler_rnn.GRUModel, meta: dict) -> None:
        self.model = model
        self.meta = meta
        self.history: list[tuple[int, str]] = []

    def feed_history(self, events: list[tuple[int, str]]) -> None:
        self.history.extend(events)

    def next_step(self, *, cond: dict | None, rng: random.Random) -> Event:
        return groove_sampler_rnn.sample(self.model, self.meta, bars=1, temperature=1.0, rng=rng)[0]


class _NgramSampler:
    def __init__(self, model) -> None:
        self.model = model
        self.history: list[tuple[int, str]] = []

    def feed_history(self, events: list[tuple[int, str]]) -> None:
        self.history.extend(events)

    def next_step(self, *, cond: dict | None, rng: random.Random) -> Event:
        return sample_ngram(self.model, bars=1, cond=cond, rng=rng)[0]


class RealtimeEngine:
    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "rnn",
        bpm: float = 120.0,
        sync: str = "internal",
        buffer_bars: int = 1,
        midi_in_ports: list[str] | None = None,
        swing_ratio: float | None = None,
    ) -> None:
        if backend not in {"rnn", "ngram"}:
            raise ValueError("backend must be 'rnn' or 'ngram'")
        if backend == "rnn" and torch is None:
            raise click.ClickException("Install extras: rnn")
        self.model_path = str(model_path)
        self.backend = backend
        if backend == "rnn":
            model, meta = groove_sampler_rnn.load(Path(self.model_path))
            self.sampler: BaseSampler = _RnnSampler(model, meta)
        else:
            model = groove_sampler_ngram.load(Path(self.model_path))
            self.sampler = _NgramSampler(model)
        self.bpm = bpm
        self.sync = sync
        self.buffer_bars = buffer_bars
        self.swing_ratio = swing_ratio
        self.midi_in_ports = midi_in_ports or []
        self._incoming: list[tuple[int, str]] = []
        self._bar = 0
        self._clock_thread: threading.Thread | None = None
        self._stop = threading.Event()
        if sync == "external":
            self._start_midi_clock()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._next: list[Event] = []
        self._load_model()
        self._midi_threads: list[threading.Thread] = []
        if self.midi_in_ports and mido is not None:
            for name in self.midi_in_ports:
                try:
                    port = mido.open_input(name)
                except Exception:
                    continue
                t = threading.Thread(target=self._listen, args=(port,), daemon=True)
                t.start()
                self._midi_threads.append(t)

    def _listen(self, port: Any) -> None:
        for msg in port:
            if self._stop.is_set():
                break
            if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                self._incoming.append((0, str(msg.note)))

    def _start_midi_clock(self) -> None:
        if mido is None:
            warnings.warn("mido required for external sync", RuntimeWarning)
            self.sync = "internal"
            return
        names = mido.get_input_names()
        if not names:
            warnings.warn("No MIDI input ports available", RuntimeWarning)
            self.sync = "internal"
            return

        def _run() -> None:
            try:
                with mido.open_input(names[0]) as port:
                    ticks = 0
                    last = time.time()
                    for msg in port:
                        if self._stop.is_set():
                            break
                        if msg.type == "clock":
                            now = time.time()
                            dt = now - last
                            last = now
                            if dt > 0:
                                self.bpm = 60.0 / (dt * 24)
                            ticks = (ticks + 1) % 96
                            if ticks == 0:
                                self._bar += 1
                        elif msg.type == "songpos":
                            self._bar = msg.pos // 16
                            ticks = 0
            except Exception:
                warnings.warn("Failed to read MIDI clock", RuntimeWarning)
                self.sync = "internal"

        self._clock_thread = threading.Thread(target=_run, daemon=True)
        self._clock_thread.start()

    def _load_model(self) -> None:
        if self.backend == "rnn":
            if GrooveRNN is None:
                raise click.ClickException("Install extras: rnn")
            obj = torch.load(self.model_path, map_location="cpu")
            model = GrooveRNN(len(obj["meta"]["vocab"]))
            model.load_state_dict(obj["state_dict"])
            self.model = (model, obj["meta"])
        else:
            self.model = groove_sampler_ngram.load(self.model_path)

    def _gen_bar(self) -> list[dict]:
        if self.backend == "rnn":
            if sample_rnn_v2 is None:
                raise click.ClickException("Install extras: rnn")
            events = sample_rnn_v2(self.model, bars=1)
        else:
            events = list(sample_ngram(self.model, bars=1))
        if self.swing_ratio:
            from .humanizer import swing_offset

            for ev in events:
                ev["offset"] = swing_offset(float(ev.get("offset", 0.0)), self.swing_ratio)
        return events

    def run(self, bars: int, sink: Callable[[Event], None]) -> None:
        self._next = self._gen_bar()
        for _ in range(bars):
            fut = self._pool.submit(self._gen_bar)
            start = time.time()
            if getattr(self, "_incoming", None):
                self.sampler.feed_history(self._incoming)
                self._incoming.clear()
            for ev in self._next:
                t = start + ev.get("offset", 0.0) * 60.0 / self.bpm
                delay = t - time.time()
                if delay > 0:
                    time.sleep(delay)
                sink(ev)
            self._next = fut.result()


__all__ = ["RealtimeEngine", "ThreadPoolExecutor"]

