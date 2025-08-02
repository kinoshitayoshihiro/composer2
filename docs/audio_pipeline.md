# Audio Pipeline Enhancements

The convolution engine now performs fast FFT convolution with optional
oversampling and high quality sample‑rate conversion. Specify the
quality level (`fast`, `high`, or `ultra`) to control the resampler.
For oversampling, signals are resampled up by 2× or 4× before the IR is
applied and filtered back down to minimise aliasing.

Tail handling detects when the convolved signal falls below the given
`tail_db_drop` threshold (default `-60` dB) and applies a short
cross‑fade to silence. When exporting to 16‑, 24‑ or 32‑bit PCM, TPDF dither
is applied automatically.

Use the new options via `render_wav` or the command line interface to
fine‑tune quality and bit depth.

Bit depth can be `16`, `24`, or `32` (float). Dithering is skipped when
normalization is disabled or with `--no-dither`.

| quality | resampler | window/setting |
| ------- | --------- | -------------- |
| fast    | soxr `q` or Kaiser 8 | quick |
| high    | soxr `hq` or Kaiser 8 | high quality |
| ultra   | soxr `vhq` or Kaiser 16 | very high quality |

## Batch audio-to-MIDI conversion

`utilities.audio_to_midi_batch` converts directories of stems into multi-track
MIDI, using `crepe` for pitch detection with an onset-based fallback when
`crepe` or `librosa` are unavailable. The CLI supports parallel jobs, custom
file extensions, and minimum note duration filtering.

For example, to process both WAV and FLAC stems:

```bash
python -m utilities.audio_to_midi_batch input/ output/ --ext wav,flac
```

When running with `--jobs > 1`, each worker loads the CREPE model. On GPUs this
can quickly exhaust available memory, so large batches may require a smaller
`--jobs` value. Non-WAV formats (e.g. FLAC or MP3) depend on `librosa` and the
system's audio codecs; missing codec support will result in read warnings.
