import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

class BuildingDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        # Load the data from the JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Root directory for actual images
        self.root_dir = root_dir

        # Transformations (albumentations)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image data and bounding box information
        image_data = self.data[idx]
        image_path = os.path.join(self.root_dir, image_data["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Convert bounding boxes from [x, y, width, height] to [x_min, y_min, x_max, y_max]
        bboxes = []
        labels = []
        for bbox in image_data["annotations"]:
            x, y, width, height = bbox
            x_min, y_min = x, y
            x_max, y_max = x + width, y + height
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # Assign label '1' to all buildings

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=np.array(image), bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        # Convert bounding boxes and labels to tensors
        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, target
    
class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            # Apply transformations
            if self.transform:
                image = self.transform(image=np.array(image))['image']
            return image, image_file
        except Exception as e:
            # Handle the exception (e.g. log it) and return None
            print(f"Error loading image {image_file}: {e}")
            pass
            # return None  # Indicate that this sample should be skipped
    
# With Augmentation
def get_transforms():
    train_transforms = A.Compose([
        A.Resize(512, 512),                             # Resize to 512x512
        A.HorizontalFlip(p=0.3),                        # Horizontal flip
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),  # Color jitter
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),       # Blurring
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),    # Gaussian noise
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),  # Brightness & Contrast
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.1),  # Color shift
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalization
        ToTensorV2()                                    # Convert to tensor
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))   # Bounding boxes in [x_min, y_min, x_max, y_max]

    val_transforms = A.Compose([
        A.Resize(512, 512),                             # Resize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    return train_transforms, val_transforms

def get_inference_transform():
    return A.Compose([
        A.Resize(512, 512),  # Resize to match training dimensions
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def collate_fn(batch):
    images, image_files = zip(*batch)
    images = list(images)
    image_files = list(image_files)
    return images, image_files

def get_test_transform():
    return A.Compose([
        A.Resize(512, 512),                             # Resize to 512x512
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalization
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))