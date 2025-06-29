# Groove Sampler

Train an n-gram model from a folder of MIDI or WAV loops.

Install optional dependencies to enable WAV extraction and the CLI:

```bash
pip install click
pip install -e .[audio]
```

```bash
modcompose groove train loops/ --ext midi --out model.pkl
```

Generate a four bar MIDI groove:

```bash
modcompose groove sample model.pkl -l 4 --temperature 0.8 --seed 42 > out.mid
```

An experimental RNN baseline can be trained from the cached loops:

```bash
modcompose rnn train loops.json --epochs 6 --out rnn.pt
modcompose rnn sample rnn.pt -l 4 > rnn.mid
```

## Aux-feature training / sampling

You can condition groove generation on section type, vocal heatmap bin and
intensity bucket. Provide a JSON file mapping each loop filename to auxiliary
data when training and use `--cond` at sampling time:

```bash
modcompose groove train loops/ --aux aux.json
modcompose groove sample model.pkl --cond '{"section":"chorus","intensity":"high"}' > out.mid
```

### Backward compatibility

If you omit the `--aux` option during training, the sampler behaves exactly as
in version 1.0 and ignores auxiliary conditions.

Models generated prior to commit 608fdda no longer include the
deprecated `aux_dims` field and should be retrained.
