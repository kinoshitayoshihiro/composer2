# Changelog

## [Unreleased]
### Added
- Lightweight module stubs for tests in `tests/_stubs.py`
- Instrument filtering for duration CSV via `--instrument` flag
- Phrase training visualizations (PR curve & confusion matrix), tag-wise metrics,
  deterministic and scheduler flags, weighted sampling, and optional DUV embeddings
- Headless-safe `--viz` plots, robust tag-aware evaluation, CSV tag filters and run
  metadata (git commit, env, sampler stats)
- `--strict-tags` option, transformer hyper-parameters/seed flags, CSV bucket
  emission, and sampler weight logging
- Standardize `duration_bucket`/`velocity_bucket` columns (legacy names warn),
  track CSV filtering stats, and record visualization usage in run metadata
### Fixed
- Harmonize DUV bucket column names and apply transformer nhead/layer/dropout
  flags while avoiding invalid LSTM kwargs
### Changed
- ⚠️ Breaking change note removed – parameters are now optional
- Unified to **numba>=0.60.0** across requirements

## [0.6.1] - 2025-07-25
### Fixed
- handle missing `pad_mask` in `decode_batch`
- error when no loops found in `_load_loops`

## [0.1.0] - 2025-07-21
### Added
- Initial dataset builder and CLI
- Unit tests and CI configuration

## [3.0.0] - 2025-07-15

### Added
- Modular plugin architecture
- Percussion sampler and groove utilities
- Style and auxiliary tag conditioning
- WebSocket bridge for realtime generation
- フェーズ0: 基盤機能とCLI整理
- フェーズ2: PercGenerator 試作
- フェーズ3: Style/Auxタグ対応
- フェーズ4: WebSocket ブリッジ
- フェーズ5: GrooveSamplerロードマップ完遂

### Changed
- Unified generator APIs and configuration loading
- Updated documentation and examples
- フェーズ1: ジェネレーターAPI統合
- フェーズ4: Hydra設定への移行

### Fixed
- Assorted stability fixes across generators and tests
- フェーズ移行時の互換性バグを修正
## [1.0.0] - 2025-07-22

### Added
- Breath Control module v1.0 with keep / attenuate / remove modes.
- ONNX inference option & energy_percentile configurability.

### Fixed
- Pop artefacts on micro breath segments.
