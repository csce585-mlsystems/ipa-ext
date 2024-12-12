import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import time

def apply_masks(image, masks):
    # Create an RGBA image to hold the mask overlay
    img_with_mask = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
    img_with_mask[:, :, 3] = 255  # Set alpha to opaque

    for ann in masks:
        # Create a color mask with random colors and transparency
        color_mask = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8).flatten()
        overlay = np.zeros_like(img_with_mask, dtype=np.uint8)
        overlay[:, :, :3] = color_mask
        overlay[:, :, 3] = int(0.35 * 255)  # Set mask transparency

        # Apply the segmentation mask
        mask = ann['segmentation'].astype(bool)
        img_with_mask[mask] = cv2.addWeighted(img_with_mask[mask], 1 - 0.35, overlay[mask], 0.35, 0)

    return img_with_mask

def save_mask(image, masks, name):
    img_with_mask = apply_masks(image, masks)
    cv2.imwrite(f'first_results/{name}.png', cv2.cvtColor(img_with_mask, cv2.COLOR_RGBA2BGRA)) # Save the image

def main():
    image = cv2.imread('notebooks/images/dog.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    sam_checkpoint = "./model_checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    name = "test"
    save_mask(image, masks, name)

if __name__ == "__main__":
    main()