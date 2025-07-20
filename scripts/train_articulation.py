from __future__ import annotations
from pathlib import Path
from pytorch_lightning.cli import LightningCLI

from data.articulation_datamodule import ArticulationDataModule
from ml_models.tagger_module import TaggerModule


def main() -> None:
    LightningCLI(
        TaggerModule,
        ArticulationDataModule,
        seed_everything_default=42,
        parser_kwargs={
            "parser_mode": "omegaconf",
        },
        save_config_callback=None,
        run=True,
        config_path=str(
            Path(__file__).resolve().parent.parent / "conf"
        ),
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()