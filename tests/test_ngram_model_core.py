from pathlib import Path

import numpy as np
import pretty_midi

from utilities.groove_sampler_ngram import (
    _load_events,
    _perplexity,
    auto_select_order,
    train,
)


def _make_loop(path: Path, pitches: list[int]) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i, p in enumerate(pitches):
        start = i * 0.25
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=p, start=start, end=start + 0.05)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_auto_select_and_prob(tmp_path: Path) -> None:
    pitches = [36, 38, 42, 46]
    for i in range(8):
        pattern = [int(np.random.choice(pitches)) for _ in range(8)]
        _make_loop(tmp_path / f"{i}.mid", pattern)
    seqs, *_ = _load_events(tmp_path, ["midi"])
    n = auto_select_order([s for s, _ in seqs])
    assert 2 <= n <= 5
    model = train(tmp_path, order=n)
    for ctx_map in model["prob"].values():
        for dist in ctx_map.values():
            probs = [np.exp(v) for v in dist.values()]
            assert abs(sum(probs) - 1.0) < 1e-6


def test_kneser_ney_perplexity(tmp_path: Path) -> None:
    pitches = [36, 38, 42, 46]
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    for i in range(6):
        pattern = [int(np.random.choice(pitches)) for _ in range(8)]
        _make_loop(train_dir / f"{i}.mid", pattern)
    for i in range(6, 8):
        pattern = [int(np.random.choice(pitches)) for _ in range(8)]
        _make_loop(val_dir / f"{i}.mid", pattern)
    model_add = train(train_dir, order=3, smoothing="add_alpha")
    model_kn = train(train_dir, order=3, smoothing="kneser_ney", discount=0.75)
    val_seqs, *_ = _load_events(val_dir, ["midi"])
    seqs = [s for s, _ in val_seqs]
    ppx_add = _perplexity(model_add["prob"], seqs, model_add["order"])
    ppx_kn = _perplexity(model_kn["prob"], seqs, model_kn["order"])
    assert np.isfinite(ppx_kn) and ppx_kn > 0


def test_short_sequences(tmp_path: Path) -> None:
    """Ensure training works with extremely short loops."""

    pitches = [36, 38]
    for i in range(2):
        _make_loop(tmp_path / f"s{i}.mid", [pitches[i % 2]])
    seqs, *_ = _load_events(tmp_path, ["midi"])
    order = auto_select_order([s for s, _ in seqs], max_order=3)
    model = train(tmp_path, order=order, smoothing="kneser_ney")
    ppx = _perplexity(model["prob"], [s for s, _ in seqs], model["order"])
    assert np.isfinite(ppx)


def test_sparse_data_kneser_ney(tmp_path: Path) -> None:
    """Kneser-Ney should handle sparse contexts without NaNs."""

    pitches = list(range(35, 50))
    for i in range(4):
        pattern = [pitches[(i + j) % len(pitches)] for j in range(2)]
        _make_loop(tmp_path / f"sp{i}.mid", pattern)
    model = train(tmp_path, order=4, smoothing="kneser_ney")
    seqs, *_ = _load_events(tmp_path, ["midi"])
    ppx = _perplexity(model["prob"], [s for s, _ in seqs], model["order"])
    assert np.isfinite(ppx)
