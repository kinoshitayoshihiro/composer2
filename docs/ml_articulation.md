# ML Articulation Tagger

This model predicts note articulation labels from MIDI features.

## Training

```bash
python -m scripts.train_articulation data=conf/data model=conf/model
```

| feature | description |
| ------- | ----------- |
| pitch   | MIDI note number |
| duration | bucketed duration index |
| velocity | note velocity (0-127) |
| pedal_state | 0=off,1=on,2=half |

Label vocabulary:

| label | id |
| ----- | -- |
| legato | 0 |
| staccato | 1 |
| accent | 2 |
| tenuto | 3 |
| marcato | 4 |
| trill | 5 |
| sustain_pedal | 6 |

Early stopping monitors `val_loss` with patience provided via `--patience`.

## Inference

```python
from utilities import ml_articulation
model = ml_articulation.load('models/artic_tagger.ckpt')
tags = ml_articulation.predict(score, model)
# in modcompose sample
modcompose sample model.pkl --backend piano_template \
    --artic-model models/artic_tagger.ckpt
```

See `examples/artic_demo.ipynb` for a walkthrough.

### ABX Listening Test

Generate an interactive ABX page comparing dry vs articulated renditions:

```bash
python -m scripts.abx_articulation dry_midis/ articulated_midis/ --trials 12
```

Open the produced `abx.html` in a browser and submit your choices. A
`results.json` file will accumulate responses. ![abx](abx_usage.gif)
