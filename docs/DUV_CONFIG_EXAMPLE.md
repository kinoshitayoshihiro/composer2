# Example: Using DUV Humanization in Song Generation

## Configuration Structure

Add to your `main_cfg.yml` or section configuration:

```yaml
song:
  sections:
    - name: verse
      instruments:
        - type: guitar
          generator: guitar_part_generator
          
          # Controls: Onset micro-timing adjustments
          controls:
            enable: true
            strum:
              direction: auto      # auto/down/up
              span_s: 0.018        # Strum span (18ms)
              chord_window_s: 0.030
            jitter:
              onset_std_ms: 3.0    # Random onset jitter
            
          # DUV: ML-based velocity/duration humanization
          duv:
            enable: true
            model_path: checkpoints/duv_guitar_lora/duv_lora_best.ckpt
            intensity: 0.9         # 0.0 (off) - 1.0 (full)
            include_regex: "(?i)guitar|gtr"
            exclude_regex: "(?i)drum"
            
        - type: bass
          generator: bass_part_generator
          controls:
            enable: true
            ghost_notes:
              probability: 0.15
              velocity_ratio: 0.3
          duv:
            enable: true
            model_path: checkpoints/duv_bass_lora/duv_lora_best.ckpt
            intensity: 0.85
            
        - type: piano
          generator: piano_part_generator
          controls:
            enable: true
            chord_spread:
              direction: up        # up/down/random
              span_ms: 15
            sustain:
              enable: true
              cc64_value: 127
          duv:
            enable: true
            model_path: checkpoints/duv_piano_lora/duv_lora_best.ckpt
            intensity: 0.8
            
        - type: drums
          generator: drum_part_generator
          controls:
            enable: true
            flam:
              probability: 0.1
              span_ms: 10
            hi_hat:
              closed_velocity: 80
              open_velocity: 100
          duv:
            enable: true
            model_path: checkpoints/duv_drums_lora/duv_lora_best.ckpt
            intensity: 0.7
```

## Pipeline Order

The humanization pipeline runs in this order:

```
1. Generate symbolic MIDI (BasePartGenerator)
   ↓
2. Apply Controls (onset micro-timing)
   - Guitar: strum, jitter
   - Piano: chord spread, sustain
   - Bass: ghost notes
   - Drums: flams, hi-hat variations
   ↓
3. Apply DUV (velocity/duration humanization)
   - Load LoRA-adapted model
   - Predict velocity/duration adjustments
   - Blend with original (controlled by intensity)
   ↓
4. Final MIDI output (humanized)
```

## Intensity Parameter

Controls the blend between original and humanized values:

```python
# intensity = 0.0: No humanization (original values)
# intensity = 0.5: 50/50 blend
# intensity = 1.0: Full humanization (100% model predictions)

final_velocity = original * (1 - intensity) + predicted * intensity
```

**Recommended values:**
- **Guitar/Bass:** 0.8-0.9 (strong humanization)
- **Piano:** 0.7-0.8 (moderate humanization)
- **Drums:** 0.6-0.7 (subtle humanization)

## Per-Section Control

Different sections can have different humanization settings:

```yaml
sections:
  - name: intro
    instruments:
      - type: guitar
        duv:
          intensity: 0.5  # Subtle for intro
          
  - name: chorus
    instruments:
      - type: guitar
        duv:
          intensity: 1.0  # Full humanization for energy
          
  - name: outro
    instruments:
      - type: guitar
        duv:
          intensity: 0.3  # Minimal for fade-out
```

## Disable Humanization

To disable for testing/comparison:

```yaml
instruments:
  - type: guitar
    controls:
      enable: false  # No onset adjustments
    duv:
      enable: false  # No velocity/duration adjustments
```

## Advanced: Custom Model Per Track

```yaml
instruments:
  - type: guitar
    name: "Lead Guitar"
    duv:
      model_path: checkpoints/duv_guitar_lora_aggressive/duv_lora_best.ckpt
      intensity: 0.95
      
  - type: guitar
    name: "Rhythm Guitar"
    duv:
      model_path: checkpoints/duv_guitar_lora_subtle/duv_lora_best.ckpt
      intensity: 0.7
```

## Validation

Test the pipeline:

```bash
# Generate with humanization
python modular_composer.py --config examples/song_with_duv.yml --out test.mid

# Compare with baseline (no humanization)
python modular_composer.py --config examples/song_no_duv.yml --out baseline.mid

# Listen to both and compare
```

## Monitoring

Check if DUV is being applied:

```python
# Add logging to base_part_generator.py
if section_data.get('duv', {}).get('enable'):
    logger.info(f"Applying DUV: {section_data['duv']['model_path']}")
    logger.info(f"Intensity: {section_data['duv'].get('intensity', 0.9)}")
```

## Troubleshooting

### No audible difference

1. Check intensity: `intensity: 1.0` for maximum effect
2. Verify model loaded: Check logs for "Applying DUV"
3. Check track name regex: Ensure `include_regex` matches
4. Listen to specific notes, not overall mix

### Too much humanization

1. Reduce intensity: `intensity: 0.5`
2. Train with less aggressive data
3. Adjust LoRA rank down: `r: 4`

### Performance issues

1. Disable DUV for non-critical tracks
2. Use smaller models (`r: 4` instead of `r: 16`)
3. Cache model loading (load once, reuse)

## Best Practices

1. **Start subtle:** Begin with `intensity: 0.5`, increase gradually
2. **Test incremental:** Enable DUV for one instrument at a time
3. **A/B compare:** Always generate both humanized and baseline versions
4. **Genre-specific:** Rock/blues may need higher intensity than classical
5. **Mix context:** DUV works best with other humanization (Controls, groove)
