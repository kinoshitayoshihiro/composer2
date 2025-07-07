# Piano Epsilon

This release introduces a lightweight cloud API and an updated JUCE plugin.

## Live Plugin Overview

The updated JUCE plugin offers a simplified interface for real-time MIDI generation.

## API Usage Example

Start the server with Docker Compose and request four bars of accompaniment:

```bash
docker compose up -d
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"piano_companion","chords":["Cm7","F7"],"bars":4}'
```

## Latency Benchmarks

| Backend | Avg Latency (ms) |
|---------|-----------------|
| PG-ε    | 45              |
| PG-δ    | 60              |

