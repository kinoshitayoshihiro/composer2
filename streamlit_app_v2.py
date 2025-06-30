from __future__ import annotations

import json
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from utilities import groove_sampler_ngram, groove_sampler_rnn
from utilities.groove_sampler_ngram import Event
from utilities.streaming_sampler import sd


@st.cache_data  # type: ignore[misc]
def _to_json(events: list[Event]) -> str:
    return json.dumps(events)


def _play_events(events: list[Event]) -> None:
    data = _to_json(events)
    st.components.v1.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/tone@14"></script>
        <button onclick="play()">Play</button>
        <script>
        const events = {data};
        async function play() {{
            await Tone.start();
            const synth = new Tone.MembraneSynth().toDestination();
            const start = Tone.now();
            for (const ev of events) {{
                const dur = ev.duration || 0.25;
                synth.triggerAttackRelease('C2', dur, start + ev.offset, ev.velocity/127);
            }}
        }}
        </script>
        """,
        height=80,
    )


def main() -> None:
    st.set_page_config(layout="wide")
    left, right = st.columns(2)
    with left:
        backend = st.selectbox("Backend", ["ngram", "rnn"], index=0)
        bars = st.slider("Bars", 1, 16, 4)
        temp = st.slider("Temperature", 0.0, 1.5, 1.0)
        section = st.selectbox("Section", ["verse", "pre-chorus", "chorus", "bridge"])
        intensity = st.selectbox("Intensity", ["low", "mid", "high"])
        file = st.file_uploader("Model", type=["pkl", "pt"])
        preset_name = st.text_input("Preset name")
        if st.button("Save Preset"):
            st.session_state.setdefault("presets", {})[preset_name] = {
                "backend": backend,
                "bars": bars,
                "temp": temp,
                "section": section,
                "intensity": intensity,
            }
        load = st.selectbox("Load Preset", [""] + list(st.session_state.get("presets", {}).keys()))
        if load:
            pre = st.session_state["presets"][load]
            backend = pre["backend"]
            bars = pre["bars"]
            temp = pre["temp"]
            section = pre["section"]
            intensity = pre["intensity"]
    if file is None:
        return
    path = Path(file.name)
    path.write_bytes(file.getbuffer())
    events: list[Event]
    if backend == "rnn" and path.suffix == ".pt":
        model_rnn, meta = groove_sampler_rnn.load(path)
        events = groove_sampler_rnn.sample(
            model_rnn, meta, bars=bars, temperature=temp
        )
    else:
        model_ng = groove_sampler_ngram.load(path)
        events = groove_sampler_ngram.sample(model_ng, bars=bars, temperature=temp)
    events = list(events)
    with right:
        xs = [ev["offset"] for ev in events]
        ys = [36 if ev["instrument"] == "kick" else 38 for ev in events]
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers"))
        st.plotly_chart(fig, use_container_width=True)
        if sd is None:
            _play_events(events)
        else:
            if st.button("Play"):
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td) / "tmp.mid"
                    pm = groove_sampler_ngram.events_to_midi(events)
                    pm.write(tmp)
                    st.audio(tmp.read_bytes(), format="audio/midi")


if __name__ == "__main__":  # pragma: no cover - UI entry
    main()
