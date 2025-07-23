# Phrase Segmentation Realtime API

This service exposes a FastAPI WebSocket for low‑latency phrase boundary prediction.

## Requirements

Install the realtime dependencies:

```bash
pip install fastapi uvicorn websockets streamlit pandas torch
pip install -r requirements/realtime.txt
```

See [`requirements/realtime.txt`](../requirements/realtime.txt) for details.

Export an ONNX model for deployment:

```bash
python scripts/export_onnx.py --ckpt outputs/phrase.ckpt --out phrase.onnx
```

```
POST /warmup   # load model
WS   /infer    # send MIDI bytes -> receive JSON boundaries
```

Example usage:

```bash
python -m realtime.phrase_ws &
# warm model
curl -X POST http://localhost:8000/warmup
# predict
python tests/test_realtime_ws.py
```

```
[0, 0.95]
[8, 0.92]
```

```
          +---------+        +-------------+
MIDI ---> | /infer  | ---->  | boundaries  |
          +---------+        +-------------+
```

## Benchmark

```bash
python scripts/bench_ws.py
```

```yaml
- name: Benchmark WS
  run: python scripts/bench_ws.py
```
