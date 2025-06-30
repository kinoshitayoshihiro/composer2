from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional dependency
    st = None

try:
    import altair as alt
    import pandas as pd
except Exception:  # pragma: no cover - altair/pandas optional
    alt = None
    pd = None  # type: ignore

from utilities import groove_sampler_ngram as gs


def _hit_density(events: Iterable[gs.Event]) -> list[dict[str, object]]:
    counts: dict[tuple[str, int], int] = {}
    for ev in events:
        instr = str(ev["instrument"])
        step = int(round((float(ev["offset"]) % 4) * (gs.RESOLUTION / 4)))
        counts[(instr, step)] = counts.get((instr, step), 0) + 1
    data = [
        {"instrument": i, "step": s, "count": c}
        for (i, s), c in sorted(counts.items())
    ]
    return data


def generate_midi(
    model: gs.Model,
    bars: int = 4,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    humanize_vel: bool = False,
    humanize_micro: bool = False,
) -> Path:
    """Return path to a temporary MIDI preview."""
    history: list[gs.State] = []
    events: list[gs.Event] = []
    for bar in range(bars):
        bar_events = gs.generate_bar(
            history,
            model=model,
            temperature=temperature,
            top_k=top_k,
            humanize_vel=humanize_vel,
            humanize_micro=humanize_micro,
        )
        for ev in bar_events:
            ev["offset"] += bar * 4
        events.extend(bar_events)
    pm = gs.events_to_midi(events)
    tmp = Path(tempfile.mkstemp(suffix=".mid")[1])
    pm.write(str(tmp))
    return tmp


if st is not None:

    def _main() -> None:  # pragma: no cover - UI
        st.sidebar.title("Groove Visualiser")
        model_file = st.sidebar.file_uploader("Model", type=["pkl"])
        bars = st.sidebar.slider("Bars", 1, 8, 4)
        temp = st.sidebar.slider("Temperature", 0.0, 2.0, 1.0)
        topk = st.sidebar.number_input("Top-k", min_value=1, value=8)
        human_vel = st.sidebar.checkbox("Humanize velocity", value=False)
        human_micro = st.sidebar.checkbox("Humanize micro", value=False)
        if st.sidebar.button("Generate") and model_file is not None:
            path = Path(model_file.name)
            path.write_bytes(model_file.getbuffer())
            model = gs.load(path)
            midi = generate_midi(
                model,
                bars,
                temperature=temp,
                top_k=int(topk) if topk else None,
                humanize_vel=human_vel,
                humanize_micro=human_micro,
            )
            st.success("Preview generated")
            events = gs.sample(model, bars=bars, temperature=temp)
            data = _hit_density(events)
            if alt is not None and pd is not None:
                df = pd.DataFrame(data)
                chart = alt.Chart(df).mark_rect().encode(
                    x=alt.X("step:O"),
                    y=alt.Y("instrument:O"),
                    color="count:Q",
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write(data)
            with open(midi, "rb") as fh:
                st.download_button("Download MIDI", fh, file_name="preview.mid")


    if __name__ == "__main__":  # pragma: no cover - UI
        _main()

