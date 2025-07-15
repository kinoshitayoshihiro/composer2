# Realtime WebSocket Streaming

This server exposes a lightweight WebSocket endpoint for low latency note delivery.
It can be started with:

```bash
python -m utilities.ws_bridge --port 8765 --buffer 256 --config config/main_cfg.yml
```

Connect using any WebSocket client and listen on `/groove`:

```bash
wscat -c ws://localhost:8765/groove
```

Packets have the following JSON form:

```json
{ "type": "note", "t_rel_ms": 12.0,
  "midi": 36, "vel": 108, "len_ms": 95, "ch": 10 }
```

Control messages can adjust the tempo or section during playback:

```json
{ "type": "control", "tempo": 95, "section": "chorus" }
```

### Latency Tuning

Use smaller `--buffer` sizes for lower latency and monitor the 99‑percentile
in `tests/test_ws_latency.py`. When `--midi-port` is specified the same packets
are sent as virtual MIDI instead of WebSocket data.
