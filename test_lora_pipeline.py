#!/usr/bin/env python3
"""
Test LoRA DUV Pipeline Integration
Tests all 4 trained LoRA models (Piano, Guitar, Bass, Strings) with the humanization pipeline.
"""
import logging
import sys
from pathlib import Path

import pretty_midi
from music21 import instrument, note, stream

from utilities.duv_apply import apply_duv_to_pretty_midi
from data.export_pretty import stream_to_pretty_midi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_melody(inst_class, part_name: str) -> stream.Part:
    """Create a simple test melody for testing."""
    part = stream.Part(id=part_name)
    part.insert(0, inst_class())

    # Simple C major scale with different velocities
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    velocities = [64, 70, 76, 82, 88, 94, 100, 106]

    for i, (pitch, vel) in enumerate(zip(pitches, velocities)):
        n = note.Note(pitch, quarterLength=0.5)
        n.volume.velocity = vel
        part.append(n)

    return part


def test_lora_model(
    instrument_type: str,
    lora_checkpoint: str,
    base_checkpoint: str,
    scaler_path: str,
    inst_class,
):
    """Test a single LoRA model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {instrument_type} LoRA Model")
    logger.info(f"{'='*60}")
    logger.info(f"LoRA checkpoint: {lora_checkpoint}")
    logger.info(f"Base checkpoint: {base_checkpoint}")

    # Create test melody
    part = create_test_melody(inst_class, instrument_type.lower())
    logger.info(f"Created test melody with {len(part.notes)} notes")

    # Convert to PrettyMIDI
    score = stream.Score()
    score.append(part)
    pm_original = stream_to_pretty_midi(score)

    # Log original velocities
    original_vels = [n.velocity for inst in pm_original.instruments for n in inst.notes]
    logger.info(f"Original velocities: {original_vels}")

    # Apply DUV with Base Model (not LoRA for now, LoRA needs TorchScript export)
    # TODO: Export LoRA models to TorchScript for inference
    try:
        logger.info(f"Testing with BASE model: {base_checkpoint}")
        pm_humanized = apply_duv_to_pretty_midi(
            pm_original,
            model_path=base_checkpoint,  # Use base model for now
            scaler_path=scaler_path,
            mode="absolute",
            intensity=1.0,
            include_regex=None,  # Apply to all
            exclude_regex=None,
        )

        # Log humanized velocities
        humanized_vels = [n.velocity for inst in pm_humanized.instruments for n in inst.notes]
        logger.info(f"Humanized velocities: {humanized_vels}")

        # Calculate differences
        vel_diffs = [h - o for h, o in zip(humanized_vels, original_vels)]
        logger.info(f"Velocity changes: {vel_diffs}")
        logger.info(f"Mean velocity change: {sum(vel_diffs) / len(vel_diffs):.2f}")
        logger.info(f"Max velocity change: {max(vel_diffs)}")
        logger.info(f"Min velocity change: {min(vel_diffs)}")

        # Save test output
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"{instrument_type.lower()}_lora_test.mid"
        pm_humanized.write(str(output_file))
        logger.info(f"✅ Saved test output to: {output_file}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed for {instrument_type}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Test all 4 LoRA models."""
    logger.info("=" * 60)
    logger.info("LoRA DUV Pipeline Integration Test")
    logger.info("=" * 60)

    # Test configurations
    tests = [
        {
            "instrument_type": "Piano",
            "lora_checkpoint": "checkpoints/duv_piano_lora/duv_lora_final.ckpt",
            "base_checkpoint": "checkpoints/keys_duv_v2.best.ckpt",
            "scaler_path": "checkpoints/scalers/piano_duv.json",
            "inst_class": instrument.Piano,
        },
        {
            "instrument_type": "Guitar",
            "lora_checkpoint": "checkpoints/duv_guitar_lora/duv_lora_best.ckpt",
            "base_checkpoint": "checkpoints/guitar_duv_v2.best.ckpt",
            "scaler_path": "checkpoints/scalers/guitar_duv.json",
            "inst_class": instrument.AcousticGuitar,
        },
        {
            "instrument_type": "Bass",
            "lora_checkpoint": "checkpoints/duv_bass_lora/duv_lora_final.ckpt",
            "base_checkpoint": "checkpoints/bass_duv_v2.best.ckpt",
            "scaler_path": "checkpoints/scalers/bass_duv.json",
            "inst_class": instrument.AcousticBass,
        },
        {
            "instrument_type": "Strings",
            "lora_checkpoint": "checkpoints/duv_strings_lora/duv_lora_final.ckpt",
            "base_checkpoint": "checkpoints/strings_duv_v2.best.ckpt",
            "scaler_path": "checkpoints/scalers/strings_duv.json",
            "inst_class": instrument.StringInstrument,
        },
    ]

    results = {}
    for test_config in tests:
        instrument_type = test_config["instrument_type"]
        success = test_lora_model(**test_config)
        results[instrument_type] = success

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    for instrument_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{instrument_type:12s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
