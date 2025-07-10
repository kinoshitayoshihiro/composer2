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
