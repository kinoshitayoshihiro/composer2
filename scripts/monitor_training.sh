#!/bin/bash
# Monitor all training logs with summary

echo "=== Training Progress Summary ==="
echo ""

for instrument in guitar bass piano strings drums; do
    log_file="logs/${instrument}_duv_raw.log"
    
    if [ -f "$log_file" ]; then
        echo ">>> $instrument <<<"
        
        # Get batch count
        batch_info=$(grep "train_batches/epoch" "$log_file" | tail -1)
        if [ ! -z "$batch_info" ]; then
            echo "  $batch_info"
        fi
        
        # Get latest epoch info
        latest_epoch=$(grep "INFO:root:epoch [0-9]" "$log_file" | tail -1)
        if [ ! -z "$latest_epoch" ]; then
            echo "  $latest_epoch"
        fi
        
        echo ""
    else
        echo ">>> $instrument <<<"
        echo "  Log not found: $log_file"
        echo ""
    fi
done

echo "=== Live Monitoring Commands ==="
echo "  tail -f logs/guitar_duv_raw.log"
echo "  tail -f logs/bass_duv_raw.log"
echo "  tail -f logs/piano_duv_raw.log"
echo "  tail -f logs/strings_duv_raw.log"
echo "  tail -f logs/drums_duv_raw.log"
