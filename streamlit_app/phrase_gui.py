from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pretty_midi
import streamlit as st

from scripts.segment_phrase import load_model, segment_bytes


def main() -> None:
    st.title("Phrase Segmenter")
    arch = st.sidebar.selectbox("Architecture", ["transformer", "lstm"], key="arch")
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, key="thr")
    st.sidebar.button("Refine & Re-train", key="retrain")

    file = st.file_uploader(
        "MIDI or MusicXML", type=["mid", "midi", "xml"], key="upload"
    )

    if "_model" not in st.session_state or st.session_state.get("_arch") != arch:
        st.session_state["_model"] = load_model(arch, Path("phrase.ckpt"))
        st.session_state["_arch"] = arch

    if file is not None:
        data = file.getvalue()
        boundaries = segment_bytes(data, st.session_state["_model"], float(threshold))
        st.json(boundaries, expanded=False, key="boundary-json")
        try:
            pm = pretty_midi.PrettyMIDI(io.BytesIO(data))
            roll = pm.get_piano_roll(fs=24).sum(axis=0)
            df = pd.DataFrame({"roll": roll})
            st.line_chart(df)
        except Exception:
            st.warning("Could not display piano roll")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
