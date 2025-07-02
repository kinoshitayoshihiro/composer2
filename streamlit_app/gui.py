from __future__ import annotations

import random
import tempfile
from collections.abc import Callable
from pathlib import Path

from utilities.interactive_engine import InteractiveEngine

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - optional
    raise RuntimeError(
        "streamlit is required for GUI mode – install with `pip install streamlit`."
    ) from exc

try:
    import mido  # noqa: F401  # optional dependency for interactive mode
except Exception as exc:  # pragma: no cover - optional
    mido = None
    _MIDO_ERROR: Exception | None = exc
else:  # pragma: no cover - optional
    _MIDO_ERROR = None

from utilities import groove_sampler_ngram, groove_sampler_rnn, preset_manager
from utilities.midi_capture import MIDIRecorder


def _to_midi(events: list[groove_sampler_ngram.Event]) -> bytes:
    pm = groove_sampler_ngram.events_to_midi(events)
    tmp = Path(tempfile.mkstemp(suffix=".mid")[1])
    pm.write(tmp)
    data = tmp.read_bytes()
    tmp.unlink()
    return data


def setup_interactive(
    model_name: str,
    bpm: float,
    midi_in: str,
    midi_out: str,
    log_func: Callable[[str], None],
) -> InteractiveEngine | None:
    """Initialise InteractiveEngine and register ``log_func`` callback."""
    if mido is None:
        log_func(f"mido unavailable: {_MIDO_ERROR}")
        return None
    import json

    from utilities.interactive_engine import InteractiveEngine

    engine = InteractiveEngine(model_name=model_name, bpm=bpm)
    engine.add_callback(lambda ev: log_func(json.dumps(ev)))
    return engine


def main() -> None:
    mode = st.radio("Mode", ("Generate", "Interact"))
    backend = st.radio("Backend", ("ngram", "rnn"))
    bars = st.slider("Bars", 1, 16, 4)
    temp = st.slider("Temperature", 0.0, 1.5, 1.0)
    human_timing = st.slider("Timing Humanization", 0.0, 1.0, 0.0)
    human_velocity = st.slider("Velocity Humanization", 0.0, 1.0, 0.0)
    seed_input = st.text_input("Random Seed (optional)")
    if seed_input:
        random.seed(int(seed_input))
    file = st.file_uploader("Model", type=["pkl", "pt"])

    if mode == "Interact":
        if mido is None:
            st.error(f"mido unavailable: {_MIDO_ERROR}")
            return
        model_name = st.text_input("Model name", "gpt2-medium")
        bpm = st.slider("BPM", 60, 200, 120)
        in_names = mido.get_input_names()
        out_names = mido.get_output_names()
        midi_in = st.selectbox("MIDI In", in_names)
        midi_out = st.selectbox("MIDI Out", out_names)
        if "inter_logs" not in st.session_state:
            st.session_state["inter_logs"] = []
        log_box = st.empty()
        chart = st.line_chart([])

        def _log(msg: str) -> None:
            st.session_state["inter_logs"].append(msg)
            log_box.text("\n".join(st.session_state["inter_logs"][-12:]))

        if st.button("Start Interact"):
            engine = setup_interactive(model_name, bpm, midi_in, midi_out, _log)
            if engine is not None:
                import asyncio
                import threading

                def _run() -> None:
                    asyncio.run(engine.start(midi_in, midi_out))

                t = threading.Thread(target=_run, daemon=True)
                t.start()
                st.session_state["inter_thread"] = t
                st.session_state["inter_engine"] = engine
                engine.add_callback(lambda ev: chart.add_rows({"y": [ev.get("offset", 0.0)]}))
        return

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
            model = groove_sampler_ngram.load(path)  # type: ignore[assignment]
            events = groove_sampler_ngram.sample(model, bars=bars, temperature=temp)
        # simple velocity scaling
        if human_velocity > 0:
            for ev in events:
                if "velocity" in ev:
                    vel = int(ev["velocity"] * (1 - human_velocity) + 100 * human_velocity)
                    ev["velocity"] = vel
        if human_timing > 0:
            offset = 0.0
            for ev in events:
                off = ev.get("offset", 0.0)
                offset += human_timing * (off - offset)
                ev["offset"] = offset
        st.audio(_to_midi(events), format="audio/midi")


if __name__ == "__main__":  # pragma: no cover - UI entry
    main()
