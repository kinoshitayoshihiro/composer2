# Advanced Bass Techniques

This page explains optional features for shaping the bass part.

## mirror_melody
When enabled, the bass inverts the vocal melody around the tonic, creating a mirrored line.

```yaml
part_defaults:
  bass:
    mirror_melody: true
```

## kick-lock
Boost bass note velocity on beats shared with kick drum hits.

```yaml
section_overrides:
  "Chorus 1":
    bass:
      velocity_shift_on_kick: 12
```

## II–V build-up
Use a subdominant–dominant approach on the fourth beat to lead into the next chord.

```yaml
section_overrides:
  "Bridge":
    bass:
      options:
        approach_style_on_4th: subdom_dom
```

## velocity-envelope
Gradually change dynamics by specifying a velocity envelope.

```yaml
section_overrides:
  "Verse 2":
    bass:
      velocity_envelope:
        - [0.0, 60]
        - [2.0, 90]
        - [4.0, 70]
```

Combine these features to craft expressive bass lines.
