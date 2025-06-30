from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .groove_rnn_v2 import GrooveRNN, sample_rnn_v2
from .groove_sampler_ngram import load as load_ngram
from .groove_sampler_ngram import sample as sample_ngram


class RealtimeEngine:
    def __init__(
        self,
        model_path: Path,
        backend: str = "rnn",
        bpm: float = 120.0,
        sync: str = "internal",
        buffer_bars: int = 1,
    ) -> None:
        self.model_path = model_path
        self.backend = backend
        self.bpm = bpm
        self.sync = sync
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

