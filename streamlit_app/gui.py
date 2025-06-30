from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from utilities import groove_sampler_ngram
from utilities import groove_sampler_rnn


def _to_midi(events: list[groove_sampler_ngram.Event]) -> bytes:
    pm = groove_sampler_ngram.events_to_midi(events)
    tmp = Path(tempfile.mkstemp(suffix=".mid")[1])
    pm.write(tmp)
    data = tmp.read_bytes()
    tmp.unlink()
    return data


def main() -> None:
    backend = st.radio("Backend", ("ngram", "rnn"))
    bars = st.slider("Bars", 1, 16, 4)
    temp = st.slider("Temperature", 0.0, 1.5, 1.0)
    file = st.file_uploader("Model", type=["pkl", "pt"])
    if st.button("Generate") and file is not None:
        path = Path(file.name)
        path.write_bytes(file.getbuffer())
        if backend == "rnn" and path.suffix == ".pt":
            model, meta = groove_sampler_rnn.load(path)
            events = groove_sampler_rnn.sample(model, meta, bars=bars, temperature=temp)
        else:
            model = groove_sampler_ngram.load(path)
            events = groove_sampler_ngram.sample(model, bars=bars, temperature=temp)
        st.audio(_to_midi(events), format="audio/midi")


if __name__ == "__main__":  # pragma: no cover - UI entry
    main()
