import argparse
from pathlib import Path

from utilities.velocity_model import KDEVelocityModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train velocity KDE model")
    parser.add_argument("data_dir", type=Path, help="Directory of MIDI loops")
    parser.add_argument("--parts", nargs="*", default=None, help="part names")
    parser.add_argument("-o", "--out", type=Path, default=Path("velocity_model.pkl"))
    args = parser.parse_args()

    KDEVelocityModel.train(args.data_dir, parts=args.parts, out_path=args.out)
    print(f"model saved to {args.out}")


if __name__ == "__main__":
    main()
