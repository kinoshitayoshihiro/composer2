from __future__ import annotations

import tempfile
from pathlib import Path
import random

import streamlit as st

from utilities import groove_sampler_ngram
from utilities import groove_sampler_rnn
from utilities.midi_capture import MIDIRecorder
from utilities import preset_manager


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
    human_timing = st.slider("Timing Humanization", 0.0, 1.0, 0.0)
    human_velocity = st.slider("Velocity Humanization", 0.0, 1.0, 0.0)
    seed_input = st.text_input("Random Seed (optional)")
    if seed_input:
        random.seed(int(seed_input))
    file = st.file_uploader("Model", type=["pkl", "pt"])

    if "preset_names" not in st.session_state:
        st.session_state["preset_names"] = preset_manager.list_presets()
    with st.sidebar.expander("Presets", expanded=False):
        if st.button("Refresh Presets"):
            st.session_state["preset_names"] = preset_manager.list_presets()
        selected = st.selectbox("Load", [""] + st.session_state["preset_names"])
        if selected:
            cfg = preset_manager.load_preset(selected)
            bars = cfg.get("bars", bars)
            temp = cfg.get("temp", temp)
        name = st.text_input("Preset Name")
        if st.button("Save Preset"):
            preset_manager.save_preset(name, {"bars": bars, "temp": temp})

    if "recorder" not in st.session_state:
        st.session_state["recorder"] = None

    col1, col2 = st.columns(2)
    if col1.button("Record"):
        st.session_state["recorder"] = MIDIRecorder()
        st.session_state["recorder"].start_recording()
    if col2.button("Stop") and st.session_state["recorder"]:
        part = st.session_state["recorder"].stop_recording()
        tmp = Path(tempfile.mkstemp(suffix=".mid")[1])
        part.write("midi", fp=str(tmp))
        st.audio(tmp.read_bytes(), format="audio/midi")
        tmp.unlink()
        st.session_state["recorder"] = None

    if st.button("Generate") and file is not None:
        path = Path(file.name)
        path.write_bytes(file.getbuffer())
        if backend == "rnn" and path.suffix == ".pt":
            model, meta = groove_sampler_rnn.load(path)
            events = groove_sampler_rnn.sample(model, meta, bars=bars, temperature=temp)
        else:
            model = groove_sampler_ngram.load(path)
            events = groove_sampler_ngram.sample(model, bars=bars, temperature=temp)
        # simple velocity scaling
        if human_velocity > 0:
            for ev in events:
                if "velocity" in ev:
                    ev["velocity"] = int(ev["velocity"] * (1 - human_velocity) + 100 * human_velocity)
        if human_timing > 0:
            offset = 0.0
            for ev in events:
                off = ev.get("offset", 0.0)
                offset += human_timing * (off - offset)
                ev["offset"] = offset
        st.audio(_to_midi(events), format="audio/midi")


if __name__ == "__main__":  # pragma: no cover - UI entry
    main()
