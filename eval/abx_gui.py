from __future__ import annotations

import base64
import random
from pathlib import Path

import streamlit as st

HTML_PLAYER = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.39/Tone.min.js"></script>
<audio id="player" src="data:audio/midi;base64,{b64}" controls></audio>
"""


def _midi_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run_gui(human_dir: Path, ai_dir: Path, trials: int = 8) -> None:
    st.title("ABX Test")
    human = list(human_dir.glob("*.mid"))
    ai = list(ai_dir.glob("*.mid"))
    if "score" not in st.session_state:
        st.session_state.score = 0
        st.session_state.trial = 0
    if st.session_state.trial >= trials:
        st.success(f"Score: {st.session_state.score}/{trials}")
        return
    a = random.choice(human)
    b = random.choice(ai)
    x = random.choice([a, b])
    st.markdown("### Sample A")
    st.markdown(HTML_PLAYER.format(b64=_midi_b64(a)), unsafe_allow_html=True)
    st.markdown("### Sample B")
    st.markdown(HTML_PLAYER.format(b64=_midi_b64(b)), unsafe_allow_html=True)
    st.markdown("### Sample X")
    st.markdown(HTML_PLAYER.format(b64=_midi_b64(x)), unsafe_allow_html=True)
    if st.button("A"):
        if x == a:
            st.session_state.score += 1
        st.session_state.trial += 1
    if st.button("B"):
        if x == b:
            st.session_state.score += 1
        st.session_state.trial += 1
