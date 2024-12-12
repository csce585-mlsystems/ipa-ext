import os
import numpy as np
from PIL import Image
import argparse

def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def get_file_key(filename, delimiter="_", index=2):
    parts = filename.split(delimiter)
    return delimiter.join(parts[:index + 1])

def average_iou(gt_folder, pred_folder):
    gt_files = {get_file_key(f): os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.endswith(".png")}
    pred_files = {get_file_key(f): os.path.join(pred_folder, f) for f in os.listdir(pred_folder) if f.endswith(".png")}

    common_keys = set(gt_files.keys()).intersection(set(pred_files.keys()))
    if not common_keys:
        print("No matching files found!")
        return 0.0

    iou_scores = []
    for key in common_keys:
        gt_path = gt_files[key]
        pred_path = pred_files[key]

        gt_image = np.array(Image.open(gt_path).convert("L")) > 0  # Convert to binary mask
        pred_image = np.array(Image.open(pred_path).convert("L")) > 0  # Convert to binary mask

        iou = compute_iou(gt_image, pred_image)
        iou_scores.append(iou)

    avg_iou = np.mean(iou_scores)
    return avg_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average IoU between ground truth and predicted masks.")
    parser.add_argument("--gt_folder", type=str, help="Path to the ground truth folder.")
    parser.add_argument("--pred_folder", type=str, help="Path to the predicted masks folder.")
    
    args = parser.parse_args()

    # Compute and print the average IoU
    average_iou_score = average_iou(args.gt_folder, args.pred_folder)
    print(f"Average IoU: {average_iou_score:.4f}")
