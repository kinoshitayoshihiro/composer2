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
