from __future__ import annotations

import random
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path

from . import groove_sampler_rnn
from .streaming_sampler import BaseSampler, RealtimePlayer

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


class RealtimeEngine:
    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "rnn",
        bpm: float = 120.0,
        sync: str = "internal",
        buffer_bars: int = 1,
    ) -> None:
        if backend != "rnn":
            raise ValueError("Only rnn backend supported")
        model, meta = groove_sampler_rnn.load(Path(model_path))
        self.sampler: BaseSampler = _RnnSampler(model, meta)
        self.bpm = bpm
        self.sync = sync
        self.buffer_bars = buffer_bars
        self._bar = 0
        self._clock_thread: threading.Thread | None = None
        self._stop = threading.Event()
        if sync == "external":
            self._start_midi_clock()

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

    def run(self, bars: int, sink: Callable[[Event], None]) -> None:
        player = RealtimePlayer(
            self.sampler, bpm=self.bpm, sink=sink, buffer_bars=self.buffer_bars
        )
        for _ in range(bars):
            player.bpm = self.bpm
            player.play(bars=1)
        self._stop.set()
        if self._clock_thread is not None:
            self._clock_thread.join(timeout=0.1)
        self.model_path = model_path
        self.backend = backend
        self.bpm = bpm
        self.sync = sync
        self._mido = None
        if self.sync == "external":
            try:
                import mido  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                logging.warning("mido not installed; falling back to internal sync")
                self.sync = "internal"
            else:
                self._mido = mido
        self.buffer_bars = buffer_bars
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._next = []
        self._load_model()

    def _load_model(self) -> None:
        if self.backend == "rnn":
            obj = torch.load(self.model_path, map_location="cpu")
            model = GrooveRNN(len(obj["meta"]["vocab"]))
            model.load_state_dict(obj["state_dict"])
            self.model = (model, obj["meta"])
        else:
            self.model = load_ngram(self.model_path)

    def _gen_bar(self) -> list[dict]:
        if self.backend == "rnn":
            return sample_rnn_v2(self.model, bars=1)
        return list(sample_ngram(self.model, bars=1))

    def run(self, bars: int, sink: Callable[[dict], None]) -> None:
        self._next = self._gen_bar()
        for _ in range(bars):
            fut = self._pool.submit(self._gen_bar)
            start = time.time()
            for ev in self._next:
                t = start + ev.get("offset", 0.0) * 60.0 / self.bpm
                delay = t - time.time()
                if delay > 0:
                    time.sleep(delay)
                sink(ev)
            self._next = fut.result()

