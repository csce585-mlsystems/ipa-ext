#!/bin/bash

MODEL_WEIGHTS=(
  "./model_checkpoints/sam_vit_b_01ec64.pth"
  "./model_checkpoints/sam_vit_h_4b8939.pth"
  "./model_checkpoints/sam_vit_l_0b3195.pth"
)

MODEL_TYPES=(
  "vit_b"
  "vit_h"
  "vit_l"
)

# Ensure directory exist
mkdir -p sam_energy_log

for i in "${!MODEL_WEIGHTS[@]}"; do 
    model_weight="${MODEL_WEIGHTS[$i]}"
    model_type="${MODEL_TYPES[$i]}"
    
    model_name=$(basename "$model_weight" .pth)
    log_file="sam_energy_log/${model_name}_energy_log.txt"
    save_folder="./segmentation_benchmark/results_${model_type}"

    mkdir -p "$save_folder"

    echo "Starting energy monitoring for $model_name..."
    # Start the Python processing script
    python batch_process_sam.py \
        --image_dir="./segmentation_benchmark/input/" \
        --model_weights="$model_weight" \
        --model_type="$model_type" \
        --save_folder="$save_folder" &
    
    SCRIPT_PID=$!

    # Monitor energy consumption while the Python script is running
    echo "Logging energy data to $log_file"
    sudo perf stat -e power/energy-pkg/ -I 1000 2>> "$log_file" &
    PERF_PID=$!

    # Wait for the Python script to finish
    wait $SCRIPT_PID

    # Stop energy monitoring
    kill $PERF_PID 2>/dev/null

    echo "Finished processing with model weights: $model_weight"
    echo "Results saved to $save_folder"
done
