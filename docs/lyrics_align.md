# Lyrics Alignment Model

This module aligns phoneme sequences to sung audio using a lightweight CTC model.

## Installation

```bash
pip install -e .[alignment]
```

Requires `pytorch-lightning>=2.2`, `fastapi>=0.95`, and `uvicorn>=0.22`.

## Training

```bash
python scripts/train_lyrics_align.py --train_dir data/train \
    --val_dir data/val --out outputs/lyrics_align.ckpt
```
TensorBoard logs are stored in `outputs/` and can be viewed with:

```bash
tensorboard --logdir outputs
```

## CLI Inference

```bash
python scripts/align_lyrics.py --audio vocal.wav --midi melody.mid \
    --ckpt outputs/lyrics_align.ckpt
```

### Handling CTCLoss input lengths

When computing the CTC loss, make sure that `input_len` does not exceed the
length of `logp`:

```python
logp = model(audio, midi)
input_len = input_len.clamp(max=logp.size(0))
loss = crit(logp, target, input_len, target_len)
```

## Realtime WebSocket

```bash
uvicorn realtime.alignment_ws:app
```

1. POST `/warmup` with `{"ckpt":"outputs/lyrics_align.ckpt"}`
2. Connect `ws://host/infer`, send `{ "midi": [0, 500, ...] }` then stream raw
   `float32` audio bytes. Each chunk returns alignment JSON.

The config flags `freeze_encoder` and `gradient_checkpointing` allow
fine‑tuning control of the Wav2Vec2 encoder.

## Streaming Parameters

`chunk_ms` query parameter controls slice size, while `heartbeat` sends a JSON
heartbeat after each alignment.

## Colab

[Interactive Tutorial](https://colab.research.google.com/)

![TensorBoard](./tensorboard.png)

## Performance

| model | MAE | loss |
|-------|-----|------|
| small | <50ms | <0.15 |

See `bench/lyrics_align_bench.py` for details.

## 必要な依存

以下は optional dependencies です。事前にインストールしてください：

```bash
pip install -r requirements/alignment.txt
```

テストを実行するには `pytest-asyncio` もインストールしてください：

```bash
pip install pytest-asyncio
```
テスト環境：`tests/conftest.py` に自動でスタブを注入します。
