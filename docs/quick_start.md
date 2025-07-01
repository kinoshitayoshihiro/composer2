# Quick Start

Install locally and run the training pipeline.

```bash
pip install -e .[dev]
python -m modcompose.auto_tag input.mid tags.yaml
python -m modcompose.train config.yaml
python -m modcompose.play live
```

For live playback, run:

```bash
modcompose realtime
```
