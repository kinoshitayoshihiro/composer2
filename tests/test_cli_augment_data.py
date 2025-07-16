import sys
from pathlib import Path

from scripts import train_velocity


def _make_wav(path: Path) -> None:
    path.write_bytes(b"RIFF0000WAVEfmt ")


def test_missing_drums_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = train_velocity.main([
        "augment-data",
        "--drums-dir",
        str(tmp_path / "missing"),
        "--out-dir",
        str(out_dir),
    ])
    assert rc == 1


def test_auto_create_out_dir(tmp_path: Path) -> None:
    drums = tmp_path / "drums"
    drums.mkdir()
    _make_wav(drums / "a.wav")
    out_dir = tmp_path / "out"
    rc = train_velocity.main([
        "augment-data",
        "--drums-dir",
        str(drums),
        "--out-dir",
        str(out_dir),
        "--snrs",
        "1",
        "--shifts",
        "2",
        "--rates",
        "1.0",
        "2.0",
    ])
    assert rc == 0
    assert len(list(out_dir.rglob("*.wav"))) >= 1


def test_seed_reproducible(tmp_path: Path) -> None:
    drums = tmp_path / "drums"
    drums.mkdir()
    _make_wav(drums / "a.wav")
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    rc1 = train_velocity.main([
        "--seed",
        "42",
        "augment-data",
        "--drums-dir",
        str(drums),
        "--out-dir",
        str(out1),
    ])
    rc2 = train_velocity.main([
        "--seed",
        "42",
        "augment-data",
        "--drums-dir",
        str(drums),
        "--out-dir",
        str(out2),
    ])
    assert rc1 == 0 and rc2 == 0
    files1 = sorted(p.name for p in out1.glob("*.wav"))
    files2 = sorted(p.name for p in out2.glob("*.wav"))
    assert files1 == files2


def test_param_variations(tmp_path: Path) -> None:
    drums = tmp_path / "d"
    drums.mkdir()
    _make_wav(drums / "a.wav")
    out_dir = tmp_path / "out"
    rc = train_velocity.main(
        [
            "--seed",
            "1",
            "augment-data",
            "--drums-dir",
            str(drums),
            "--out-dir",
            str(out_dir),
            "--snrs",
            "0",
            "1",
            "--shifts",
            "0",
            "1",
            "--rates",
            "1.0",
            "1.5",
        ]
    )
    assert rc == 0
    files = sorted(out_dir.glob("*.wav"))
    assert len(files) == 8
    sizes = {f.stat().st_size for f in files}
    assert len(sizes) > 1


def test_out_dir_not_writable(tmp_path: Path) -> None:
    drums = tmp_path / "drums"
    drums.mkdir()
    _make_wav(drums / "a.wav")
    out_dir = tmp_path / "out"
    out_dir.write_text("not a dir")
    rc = train_velocity.main(
        [
            "augment-data",
            "--drums-dir",
            str(drums),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 1
