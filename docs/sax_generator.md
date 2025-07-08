# Sax Generator

The `SaxGenerator` creates improvised alto sax phrases. It inherits from
`MelodyGenerator` and accepts the same global parameters.

## Initialization Parameters

- `seed` *(int | None)* – RNG seed for repeatable phrases.
- `staccato_prob` *(float)* – chance of staccato articulation per note.
- `slur_prob` *(float)* – probability to tie consecutive notes with a slur.
- `vibrato_depth` *(float)* – pitch‑wheel depth for vibrato.
- `vibrato_rate` *(float)* – vibrato frequency in Hz.

Instantiate directly or via configuration:

```python
from generator.sax_generator import SaxGenerator

sax = SaxGenerator(seed=42)
```

## `compose(section_data)` Inputs

`section_data` follows the standard schema with optional overrides:

```yaml
part_params:
  melody:
    rhythm_key: sax_basic_swing
musical_intent:
  emotion: joy
  intensity: high
```

`rhythm_key` selects a phrase pattern. Emotion and intensity map to a default
pattern when the key is omitted.

### Section Data Fields

- `tonic_of_section` – root note for scale selection.
- `mode` – scale mode such as `major` or `dorian`.
- `musical_intent.emotion` – maps to a phrase bucket.
- `musical_intent.intensity` – controls velocity curve.
- `part_params.melody.rhythm_key` – explicit pattern name.
- `part_params.melody.growl` – enable growl articulation.
- `part_params.melody.altissimo` – extend the range an octave higher.

## MIDI CC & Vibrato

Articulation events are emitted as CC1/CC2 based on staccato and slur
markers. Vibrato is approximated with pitch‑wheel events whose depth and
rate can be customised.

## API

The FastAPI server exposes `/generate_sax`:

```bash
uvicorn api.sax_server:app
```

Send a request containing `growl` and `altissimo` flags:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"growl": true, "altissimo": false}' \
     http://localhost:8000/generate_sax
```

Each note dict in the response contains `note`, `velocity`, `offset` and the
requested flags.

To render MIDI from a config file:

```bash
python modular_composer.py --main-cfg config/main_cfg.yml --dry-run
```

Include a `Sax Solo` section in `sections_to_generate` to produce a demo
`demos/demo_Sax_Solo.mid`.
