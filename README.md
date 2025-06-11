# Composer Project

This repository combines Japanese poetry readings with emotional original music. The goal is to divide stories or poems into chapters and automatically generate and integrate chord progressions, melodies, arrangements, and human-like expression for each section.

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt pyyaml
   ```
3. Run the test script and the full test suite:
   ```bash
   python tools/test_timesig.py
   pytest -q
   ```

These steps will ensure the environment is configured to run the project tools and tests.
