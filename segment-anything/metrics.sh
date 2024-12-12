#!/bin/bash

RESULTS_FOLDERS=(
  "./segmentation_benchmark/results_vit_b"
  "./segmentation_benchmark/results_vit_b"
  "./segmentation_benchmark/results_vit_b"
)

for i in "${!RESULTS_FOLDERS[@]}"; do
    folder_path="${RESULTS_FOLDERS[$i]}"
    # Calculate IoU for each of the results_folder compared to the ground truth segmentation masks
    python calculate_iou.py --gt_folder="./segmentation_benchmark/gt" --pred_folder="${folder_path}"
done
