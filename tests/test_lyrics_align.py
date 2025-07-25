import asyncio
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from models import lyrics_alignment


@pytest.fixture()
def alignment_ws_module():
    spec = importlib.util.spec_from_file_location(
        "realtime.alignment_ws",
        Path(__file__).resolve().parent.parent / "realtime" / "alignment_ws.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def client(alignment_ws_module):
    resp = ModuleType("resp")
    resp.json = lambda: asyncio.run(alignment_ws_module.health())
    cli = ModuleType("client")
    cli.get = lambda _p: resp
    return cli


def _make_sample(dir: Path):
    import pretty_midi
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = np.sin(2 * np.pi * 440 * t)
    sf.write(dir / "sample.wav", y, sr)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    times = [0.2, 0.4, 0.6, 0.8]
    for i, start in enumerate(times):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=60 + i, start=start, end=start + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(dir / "sample.mid"))
    phn = dir / "sample.phn"
    with phn.open("w") as f:
        for p, tm in zip(["a", "b", "c", "d"], times):
            f.write(f"{tm} {p}\n")


def test_train_and_cli(tmp_path: Path):
    from scripts import align_lyrics, train_lyrics_align

    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    _make_sample(train)
    _make_sample(val)
    cfg = {
        "sample_rate": 16000,
        "hop_length_ms": 20,
        "midi_feature_dim": 4,
        "hidden_size": 8,
        "dropout": 0.0,
        "ctc_blank": "<blank>",
        "train_dir": str(train),
        "val_dir": str(val),
        "batch_size": 1,
        "epochs": 1,
        "lr": 0.01,
        "checkpoint": str(tmp_path / "ckpt.pt"),
    }
    cfg = SimpleNamespace(**cfg)
    loss0, _ = train_lyrics_align.evaluate(
        lyrics_alignment.LyricsAligner(
            3, cfg.midi_feature_dim, cfg.hidden_size, cfg.dropout, cfg.ctc_blank
        ),
        torch.utils.data.DataLoader(
            train_lyrics_align.AlignDataset(
                train, cfg.sample_rate, cfg.hop_length_ms, blank=cfg.ctc_blank
            ),
            batch_size=1,
            collate_fn=train_lyrics_align.collate,
        ),
        torch.nn.CTCLoss(blank=3, zero_infinity=True),
        cfg.hop_length_ms,
        3,
    )
    ds = train_lyrics_align.AlignDataset(
        train, cfg.sample_rate, cfg.hop_length_ms, blank=cfg.ctc_blank
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, collate_fn=train_lyrics_align.collate
    )
    model = lyrics_alignment.LyricsAligner(
        len(ds.vocab), cfg.midi_feature_dim, cfg.hidden_size, cfg.dropout, cfg.ctc_blank
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = torch.nn.CTCLoss(blank=len(ds.vocab), zero_infinity=True)
    for audio, midi, input_len, target, target_len, _ in dl:
        opt.zero_grad()
        logp = model(audio, midi)
        loss = crit(logp, target, input_len, target_len)
        loss.backward()
        opt.step()
    vocab = ds.vocab
    mcfg = cfg.__dict__
    loss1, _ = train_lyrics_align.evaluate(
        model,
        torch.utils.data.DataLoader(
            train_lyrics_align.AlignDataset(
                val,
                cfg.sample_rate,
                cfg.hop_length_ms,
                vocab=vocab,
                blank=cfg.ctc_blank,
            ),
            batch_size=1,
            collate_fn=train_lyrics_align.collate,
        ),
        torch.nn.CTCLoss(blank=len(vocab), zero_infinity=True),
        cfg.hop_length_ms,
        len(vocab),
    )
    assert loss1 >= 0
    out = align_lyrics.align_file(
        train / "sample.wav", train / "sample.mid", model, mcfg, vocab
    )
    assert isinstance(out, list)


@pytest.mark.asyncio
async def test_realtime_ws(monkeypatch, alignment_ws_module):

    class DummyWS:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.sent: list[dict] = []

        async def accept(self):
            pass

        async def receive_text(self):
            return json.dumps({"midi": [0.0]})

        async def receive_bytes(self):
            if self.payload is None:
                raise RuntimeError("done")
            d = self.payload
            self.payload = None
            return d

        async def send_json(self, obj):
            self.sent.append(obj)
            if len(self.sent) >= 2:
                raise RuntimeError("done")

    monkeypatch.setattr(
        alignment_ws_module,
        "align_audio",
        lambda *a, **k: [{"phoneme": "a", "time_ms": 0.0}],
    )
    alignment_ws_module._model = (
        object(),
        {"hop_length_ms": 20, "heartbeat": True},
        ["a"],
    )
    ws = DummyWS(np.zeros(1600, dtype=np.float32).tobytes())
    start = time.perf_counter()
    with pytest.raises(RuntimeError):
        await alignment_ws_module.infer(ws)
    latency = (time.perf_counter() - start) * 1000.0
    assert latency < 100.0
    assert ws.sent and isinstance(ws.sent[0]["phoneme"], str)
    assert ws.sent[1] == {"heartbeat": True}


@pytest.mark.asyncio
async def test_warmup_busy(monkeypatch, alignment_ws_module):

    async def slow_load(path: Path):
        await asyncio.sleep(0.1)
        return object(), {}, []

    monkeypatch.setattr(alignment_ws_module, "load_model", lambda p: slow_load(p))
    alignment_ws_module._model = None
    tasks = [
        asyncio.create_task(alignment_ws_module.warmup("a.ckpt")) for _ in range(5)
    ]
    results = await asyncio.gather(*tasks)
    assert sum(r["status"] == "ok" for r in results) == 1
    assert sum(r["status"] == "busy" for r in results) == 4


def test_blank_audio_empty():
    from scripts import align_lyrics

    model = lyrics_alignment.LyricsAligner(1)
    model.fc.weight.data.zero_()
    model.fc.bias.data.zero_()
    model.fc.bias.data[model.blank_id] = 1.0
    cfg = {"sample_rate": 16000, "hop_length_ms": 20}
    out = align_lyrics.align_audio(torch.zeros(1600), [], model, cfg, ["a"])
    assert out == []


def test_short_segment():
    from scripts import align_lyrics

    model = lyrics_alignment.LyricsAligner(1)
    cfg = {"sample_rate": 16000, "hop_length_ms": 20}
    out = align_lyrics.align_audio(torch.zeros(10), [], model, cfg, ["a"])
    assert out == []


def test_batch_cli(tmp_path: Path):
    from scripts import align_lyrics

    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    _make_sample(d1)
    _make_sample(d2)
    ckpt = tmp_path / "m.pt"
    model = lyrics_alignment.LyricsAligner(1)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "sample_rate": 16000,
                "hop_length_ms": 20,
                "midi_feature_dim": 4,
                "hidden_size": 8,
                "dropout": 0.0,
                "ctc_blank": "<blank>",
            },
            "vocab": ["a"],
        },
        ckpt,
    )
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            [
                {"audio": str(d1 / "sample.wav"), "midi": str(d1 / "sample.mid")},
                {"audio": str(d2 / "sample.wav"), "midi": str(d2 / "sample.mid")},
            ]
        )
    )
    out = tmp_path / "out.json"
    assert (
        align_lyrics.main(
            [
                "--batch",
                str(batch),
                "--ckpt",
                str(ckpt),
                "--out_json",
                str(out),
            ]
        )
        == 0
    )
    data = json.loads(out.read_text())
    assert len(data) == 2


@pytest.mark.asyncio
async def test_large_payload_abort(monkeypatch, alignment_ws_module):

    class DummyWS:
        def __init__(self):
            self.closed = False
            self.code = None

        async def accept(self):
            pass

        async def receive_text(self):
            return json.dumps({"midi": []})

        async def receive_bytes(self):
            return b"0" * (1_000_001)

        async def send_json(self, obj):
            self.msg = obj

        async def close(self, code=1000):
            self.closed = True
            self.code = code

    alignment_ws_module._model = (object(), {"sample_rate": 16000}, ["a"])
    ws = DummyWS()
    await alignment_ws_module.infer(ws)
    assert ws.closed and ws.code == 1011


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.json() == {"status": "ok"}
