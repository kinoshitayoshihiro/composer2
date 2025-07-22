# Changelog

## [Unreleased]

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
