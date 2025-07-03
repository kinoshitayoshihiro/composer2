# AI Features

This project optionally integrates a Transformer-based bass generator via
Hugging Face Transformers.

## Usage

Install the dependencies:

```bash
pip install transformers torch mido python-rtmidi
```

Generate with the new backend:

```bash
modcompose live model.pkl --ai-backend transformer --model-name gpt2-medium
```

Or produce a short JSON sample:

```bash
modcompose sample model.pkl --ai-backend transformer --model-name gpt2-medium
```

Add rhythm style tokens with `--rhythm-schema`:

```bash
modcompose sample model.pkl --ai-backend transformer \
  --model-name gpt2-medium --rhythm-schema <swing16>
```

Combine with the phrase diversity filter to avoid repetition:

```bash
modcompose sample model.pkl --ai-backend transformer \
  --model-name gpt2-medium --rhythm-schema <straight8> | \
  modcompose diversity-filter --n 8 --max-sim 0.8
```

Enable feedback from previous sessions with `--use-history`. Generation
statistics are stored in `~/.otokotoba/history.jsonl` and loaded on start.

For real-time interaction use the interactive mode:

```bash
modcompose interact --backend transformer --model-name gpt2-medium \
  --midi-in "Device In" --midi-out "Device Out" --bpm 120
```

Incoming MIDI notes trigger the `TransformerBassGenerator` and forward events to
the selected MIDI output.
