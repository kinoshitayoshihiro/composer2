from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from utilities import groove_sampler_ngram
from utilities.streaming_sampler import RealtimePlayer, sd
import threading

st.sidebar.title("Groove Visualiser")
model_file = st.sidebar.file_uploader("Model", type=["pkl", "pt"])
bars = st.sidebar.slider("Bars", 1, 8, 4)
temp = st.sidebar.slider("Temperature", 0.1, 1.5, 1.0)
bpm = st.sidebar.slider("BPM", 60, 180, 100)

if model_file is not None:
    path = Path(model_file.name)
    path.write_bytes(model_file.getbuffer())
    if path.suffix == ".pt":
        try:
            from utilities import groove_sampler_rnn
        except ModuleNotFoundError:  # pragma: no cover - torch missing
            st.error("PyTorch not installed")
            events = []
            model = meta = None
        else:
            model, meta = groove_sampler_rnn.load(path)
            events = groove_sampler_rnn.sample(model, meta, bars=bars, temperature=temp)
    else:
        model = groove_sampler_ngram.load(path)
        events = groove_sampler_ngram.sample(model, bars=bars, temperature=temp)
    steps = [ev["offset"] for ev in events]
    pitches = [36 if ev["instrument"] == "kick" else 38 for ev in events]
    fig = go.Figure(go.Scatter(x=steps, y=pitches, mode="markers"))
    st.plotly_chart(fig)
    if sd is None:
        st.button("Play", disabled=True)
        st.info("Install extras: pip install -e .[gui]")
    elif st.button("Play"):

        class _Wrap:
            def __init__(self):
                self.events = []

            def feed_history(self, events):
                pass

            def next_step(self, *, cond, rng):
                if not self.events:
                    if path.suffix == ".pt":
                        self.events = groove_sampler_rnn.sample(
                            model, meta, bars=bars, temperature=temp
                        )
                    else:
                        self.events = groove_sampler_ngram.sample(
                            model, bars=bars, temperature=temp
                        )
                return self.events.pop(0)

        player = RealtimePlayer(_Wrap(), bpm=bpm)
        threading.Thread(target=player.play, args=(bars,), daemon=True).start()
