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

def save_mask(image, masks, name):
    img_with_mask = apply_masks(image, masks)
    cv2.imwrite(f'results/{name}.png', cv2.cvtColor(img_with_mask, cv2.COLOR_RGBA2BGRA))

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
    cv2.imwrite(f'results/{name}.png', cv2.cvtColor(img_with_mask, cv2.COLOR_RGBA2BGRA)) # Save the image

def main():
    image_dir = './img_dataset'
    sam_checkpoint = "./model_checkpoints/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    dataset = ImageDataset(image_dir=image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for image, filename in dataloader:
        print(f"Processing:\t{filename[0].split('.')[0]}")
        image = image[0].numpy()
        masks = process_with_sam(sam, image)
        save_mask(image, masks, filename[0].split('.')[0])

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
