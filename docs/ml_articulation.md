# ML Articulation Tagger

The tagger predicts articulations per note. Durations are stored in **quarterLength** units.
Labels defined in `articulation_schema.yaml`:

| Label | ID |
|-------|----|
| legato | 0 |
| staccato | 1 |
| accent | 2 |
| tenuto | 3 |
| marcato | 4 |
| trill | 5 |
| tremolo | 6 |
| sustain_pedal | 7 |
| none | 8 |

Duration buckets come from `duration_bucket.to_bucket()`.


| Bucket | qlen < |
|-------:|-------|
| 0 | 0.25 |
| 1 | 0.5 |
| 2 | 1.5 |
| 3 | 3.0 |
| 4 | 6.0 |
| 5 | 12.0 |
| 6 | 24.0 |
| 7 | otherwise |

## Tempo-aware onset

Onset and duration are computed in quarter lengths using the tempo map. The
`seconds_to_qlen` helper iterates through tempo segments and accumulates beats.

## Pedal merge

All sustain pedal CC64 events are merged across instruments before feature
extraction. Values \>= 64 mark "on", 40-63 produce a "half" state.

## Sequence dataset

`SeqArticulationDataset` pads each track to `(B, L)` and returns a batch dictionary:

```
{
  "pitch": LongTensor(B, L),
  "bucket": LongTensor(B, L),
  "pedal": LongTensor(B, L),
  "velocity": FloatTensor(B, L),
  "qlen": FloatTensor(B, L),
  "labels": LongTensor(B, L),
  "pad_mask": BoolTensor(B, L),
}
```

## Training with LightningCLI

Hydra configuration files reside in `conf/`. Start training with:

```bash
python -m scripts.train_articulation \
    data.csv_path=data/train.csv \
    trainer.max_epochs=30
```
Set `data.weighted=true` to enable a weighted sampler balancing label
frequencies.

## Predicting on GPU

```python
score = music21.converter.parse("score.mid")
model = MLArticulationModel.load("model.pt", Path("articulation_schema.yaml"))
pairs = predict(score, model, Path("articulation_schema.yaml"), flat=False)
```
`predict()` now returns a list of `(note, label)` pairs by default, moving
batches to the model's device so GPU inference works automatically.

## Evaluating a Checkpoint

```bash
python -m scripts.eval_articulation ckpt=outputs/last.ckpt \
    data.csv_path=data/val.csv
```
This writes `metrics.json` with accuracy and F1 scores and saves the
normalized confusion matrix as both `cm_normalized.svg` and
`cm_normalized.png`.
