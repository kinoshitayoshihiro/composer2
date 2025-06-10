# Composer Project

This project generates music and poetry with dynamic arrangements.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```
These requirements include packages such as `pydantic`, `music21`, and `PyYAML`. Ensure they are installed so that all generators work correctly.

Run the helper script to verify configuration and execute the small test suite:

```bash
python tools/test_timesig.py
pytest -q
```

Use `modular_composer.py` to generate MIDI after installing dependencies.
