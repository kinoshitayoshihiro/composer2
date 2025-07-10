# Project Architecture

This project generates musical arrangements from YAML chord maps. Each section is
rendered by individual part generators. The new `utilities.arrangement_builder`
module merges these per-section parts into one track per generator while keeping
section markers. CLI tools can convert the merged score to MIDI via
`score_to_pretty_midi`.

