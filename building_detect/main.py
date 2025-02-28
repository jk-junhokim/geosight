import os
import json
import random
from torch.utils.data import DataLoader, Dataset
import argparse

from dataloader import *
from model import fasterBuilding
from trainer import *

# Arguments
parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("--num_epochs", type=int, default=25)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=0.0005)

args = parser.parse_args()


# Define directory
data_dir = '/scratch/user/junhokim/data/'
folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
print(folders)

beauty_dir = data_dir + "BEAUTY1.0_COCOLike/Split0/JPEGImages"
# print(beauty_dir)
# print(os.path.isdir(beauty_dir))


# Split Dataset
# TODO: Load Dataset (JSON) as `combined data`
random.shuffle(combined_data)

# Experiment ID
experiment_id = "exp4"

# Split Ratios
train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

train_count = int(len(combined_data) * train_ratio)
val_count = int(len(combined_data) * val_ratio)
test_count = int(len(combined_data) * test_ratio)

# Split the data
train_data = combined_data[:train_count]
val_data = combined_data[train_count:train_count + val_count]
test_data = combined_data[train_count + val_count:]

# Data directories
data_dir = f'/home/junhokim/code/datasplit_{experiment_id}'

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# File paths
train_data_file = os.path.join(data_dir, 'train_data.json')
val_data_file = os.path.join(data_dir, 'val_data.json')
test_data_file = os.path.join(data_dir, 'test_data.json')

# Save each split
with open(train_data_file, 'w') as f:
    json.dump(train_data, f, indent=2)

with open(val_data_file, 'w') as f:
    json.dump(val_data, f, indent=2)

with open(test_data_file, 'w') as f:
    json.dump(test_data, f, indent=2)
    
    
train_transforms, val_transforms = get_transforms()
root_dir = beauty_dir

# Initialize Datasets
train_dataset = BuildingDataset(json_file=train_data_file, 
                                root_dir=root_dir, 
                                transform=train_transforms)

val_dataset = BuildingDataset(json_file=val_data_file, 
                              root_dir=root_dir, 
                              transform=val_transforms)


# Load Data
train_loader = DataLoader(train_dataset, 
                          batch_size=4, 
                          shuffle=True, 
                          collate_fn=lambda x: tuple(zip(*x)))

val_loader = DataLoader(val_dataset, 
                        batch_size=4, 
                        shuffle=False, 
                        collate_fn=lambda x: tuple(zip(*x)))


net = fasterBuilding()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.model_ft.to(device)

# Checkpoint directory
checkpoint_dir = f'/scratch/user/junhokim/data/Checkpoints/checkpoints_{experiment_id}'

os.makedirs(checkpoint_dir, exist_ok=True)

train(net.model_ft,
      train_loader,
      val_loader,
      args.num_epochs,
      args.learning_rate,
      args.weight_decay,
      checkpoint_dir,
      device,)