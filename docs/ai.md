# AI Features

This project optionally integrates a Transformer-based bass generator via
Hugging Face Transformers.

## Usage

Install the dependency:

```bash
pip install transformers
```

Generate with the new backend:

```bash
modcompose live model.pkl --ai-backend transformer --model-name gpt2-music
```

Enable feedback from previous sessions with `--use-history`. Generation
statistics are stored in `~/.modcompose_history.json` and loaded on start.

For real-time interaction use the interactive mode:

```bash
modcompose interact --backend transformer --bpm 120
```

Incoming MIDI notes trigger the `TransformerBassGenerator` and forward events to
the selected MIDI output.
