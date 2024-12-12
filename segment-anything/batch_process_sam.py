import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import time

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]


def process_with_sam(sam_model, image):
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    masks = mask_generator.generate(image)
    return masks

def save_mask(masks, name):
    color_mask = apply_masks(masks)
    cv2.imwrite(f'results/{name}.png', color_mask)

def apply_masks(masks):
    # Create an empty RGB image for the mask
    mask_height, mask_width = masks[0]['segmentation'].shape
    color_mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
    
    for ann in masks:
        # Generate a random color for each mask
        random_color = np.random.randint(0, 255, 3, dtype=np.uint8)
        mask = ann['segmentation'].astype(bool)
        
        # Apply the color to the mask
        color_mask[mask] = random_color

    return color_mask

def save_mask(masks, name, save_folder):
    color_mask = apply_masks(masks)
    save_path = os.path.join(save_folder, f'{name}_mask.png')
    cv2.imwrite(save_path, color_mask)

def main():
    parser = argparse.ArgumentParser(description='Process images with SAM')
    parser.add_argument('--image_dir', type=str, help='Path to the image directory')
    parser.add_argument('--model_weights', type=str, help='Path to the model weights')
    parser.add_argument('--model_type', type=str, help='Type of SAM model')
    parser.add_argument('--save_folder', type=str, default='./segmentation_benchmark/results', help='Folder to save the masks')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for computations')

    args = parser.parse_args()

    # Ensure the save folder exists
    os.makedirs(args.save_folder, exist_ok=True)

    sam = sam_model_registry[args.model_type](checkpoint=args.model_weights)
    sam.to(device=args.device)

    dataset = ImageDataset(image_dir=args.image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for image, filename in dataloader:
        image = image[0].numpy()
        masks = process_with_sam(sam, image)
        save_mask(masks, filename[0].split('.')[0], args.save_folder)

if __name__ == "__main__":
    main()
