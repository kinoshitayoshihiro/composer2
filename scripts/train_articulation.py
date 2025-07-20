from __future__ import annotations

from pathlib import Path

from pytorch_lightning.cli import LightningCLI

from data.articulation_data import ArticulationDataModule
from ml_models.articulation_tagger import ArticulationTagger


def main() -> None:
    LightningCLI(
        ArticulationTagger,
        ArticulationDataModule,
        seed_everything_default=42,
        parser_kwargs={"parser_mode": "omegaconf"},
        run=True,
        config_path=str(Path(__file__).resolve().parent.parent / "conf"),
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
