from __future__ import annotations

from pytorch_lightning.cli import LightningCLI

from data.articulation_datamodule import ArticulationDataModule
from ml_models import TaggerModule


def main() -> None:
    LightningCLI(TaggerModule, ArticulationDataModule, save_config_callback=None)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
