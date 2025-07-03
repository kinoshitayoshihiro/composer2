# Groove Enhancements

## Late Humanization

`apply_late_humanization` randomly shifts note offsets by a few milliseconds right before playback. Enable this in the live CLI with `--late-humanize` to add extra looseness.

## HH Leak Jitter

When `kick_leak_velocity_jitter` is set in the humanizer, hi-hat hits within ±60&nbsp;ms of a kick drum have their velocities randomly varied. This mimics microphone bleed from the kick into the hi-hat channel.
