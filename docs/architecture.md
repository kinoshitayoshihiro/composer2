# Project Architecture

This project generates musical arrangements from YAML chord maps. Each section is
rendered by individual part generators. The new `utilities.arrangement_builder`
module merges these per-section parts into one track per generator while keeping
section markers. CLI tools can convert the merged score to MIDI via
`score_to_pretty_midi`.

## Part Integration Logic

The `compose()` function assembles streams returned from every part generator.
When multiple sections are processed, items with the same part ID are merged
into a single `music21.stream.Part`.  Each event's offset is adjusted by the
section start time before insertion, so the resulting part contains a continuous
timeline across the entire song.  Instruments are inserted only once per part,
and additional events from later sections simply extend the existing stream.

## Studio One Track Layout

Below is a simplified view of how the exported MIDI appears when imported into
Studio&nbsp;One:

```text
Track 1  Piano          [Verse][Chorus][Bridge]
Track 2  Bass           [Verse][Chorus][Bridge]
Track 3  Drums          [Verse][Chorus][Bridge]
```

Each instrument occupies a single track across the song, matching the section
order defined in the YAML chord map.

## Future Direction

The current integration logic will eventually move into an
`ArrangementBuilder` class to centralize score assembly and prepare for more
advanced editing features.

## Vocal-sync data flow

```
vocal MIDI -> analyse_section -> {onsets, rests, peaks}
        \-> BassGenerator/PianoGenerator
```

