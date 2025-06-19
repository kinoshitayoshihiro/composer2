#!/bin/bash
# Run demo MIDI generation and confirm drum pattern durations

python tools/generate_demo_midis.py -m config/main_cfg.yml && \
  echo "drum_patterns の duration 欠損が解消されました"
