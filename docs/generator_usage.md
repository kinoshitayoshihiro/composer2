# Generator Usage

This document describes how to instantiate part generators with the optional
`key`, `tempo` and `emotion` parameters.  These options allow generators to
adapt to the harmonic context, playback speed and emotional intent of each
section, while remaining fully backward compatible if omitted.

## Basic Example

```python
from generator import PianoGenerator

piano = PianoGenerator(
    global_settings={},
    default_instrument=None,  # uses music21 defaults
    global_time_signature="4/4",
    key=("C", "major"),
    tempo=100,
    emotion="hope_dawn",
)

part = piano.compose(section_data={"section_name": "Verse"})
```

All other instrument generators (BassGenerator, DrumGenerator, GuitarGenerator,
StringsGenerator, MelodyGenerator and its subclasses such as SaxGenerator) now
accept the same trio of parameters and forward them to the shared
`BasePartGenerator`.

## ML Velocity and Duration

If an ML velocity model trained via `train_velocity.py` is supplied through
`ml_velocity_model_path`, `_apply_ml_velocity` will refine note velocities.
Future duration models (e.g. the Duration Transformer) can be provided via the
`duration_model` argument and will be invoked automatically to adjust note
lengths.

Custom duration predictors should subclass
`utilities.duration_model_base.DurationModelBase` and implement the
`predict()` method returning new quarterLength values.

```python
piano = PianoGenerator(
    global_settings={},
    default_instrument=None,
    global_time_signature="4/4",
    key="A minor",
    tempo=90,
    emotion="quiet_pain",
    ml_velocity_model_path="ml_models/vel_model.pt",
    duration_model=my_duration_model,
)
```

The generators also draw on `/data/chordmap.yaml` and
`/data/rhythm_library.yaml` to shape their output according to the requested
emotion profile.
