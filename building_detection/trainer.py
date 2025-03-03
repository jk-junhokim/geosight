import torch.optim as optim
from tqdm import tqdm
import os, sys
import numpy as np
import torch

def train(net,
          train_loader,
          val_loader,
          num_epochs, 
          learning_rate, 
          weight_decay, 
          checkpoint_dir,
          device):

    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_val_loss = float('inf')  
    
    for epoch in range(num_epochs):
        print(f"---------- Epoch {epoch + 1}/{num_epochs} ----------")
        net.train()
        train_loss = 0
        tqdm_train = tqdm(train_loader, desc="Training", leave=False)

        for images, targets in tqdm_train:

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            optimizer.zero_grad()

            loss_dict = net(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            tqdm_train.set_postfix({"Batch Loss": losses.item()})
            
        # Avg train loss
        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        # Val
        net.eval()
        val_loss = 0
        tqdm_val = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, targets in tqdm_val:

                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

                net.train()
                loss_dict = net(images, targets)
                net.eval()

                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                tqdm_val.set_postfix({"Batch Loss": losses.item()})

        # Avg val loss for the epoch
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save ckpt at the end of the epoch
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, epoch_ckpt_path)
        print(f"Checkpoint saved at {epoch_ckpt_path}")

        # Compare with best val loss, save the best model ckpt
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_ckpt_path)
            print(f"Best checkpoint saved at {best_ckpt_path} with validation loss: {best_val_loss:.4f}")

        # Update the learning rate
        lr_scheduler.step()
    
    print("\nTraining Complete!")
    
    
def run_inference(model, data_loader, device, num_images_to_visualize=30):
    model.eval()
    model.to(device)
    predictions = []
    images_to_visualize = []
    image_files_to_visualize = []
    images_collected = 0

    with torch.no_grad():
        # Add tqdm progress bar here
        for images, image_files in tqdm(data_loader, desc="Running Inference"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                # Collect images and filenames for visualization
                if images_collected < num_images_to_visualize:
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    # Unnormalize
                    img = img * np.array([0.229, 0.224, 0.225]) + \
                          np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    images_to_visualize.append(img)
                    image_files_to_visualize.append(image_files[i])
                    images_collected += 1

                # Collect predictions
                boxes = outputs[i]['boxes'].cpu().numpy()
                labels = outputs[i]['labels'].cpu().numpy()
                scores = outputs[i]['scores'].cpu().numpy()
                prediction = {
                    'image_file': image_files[i],
                    'boxes': boxes.tolist(),
                    'labels': labels.tolist(),
                    'scores': scores.tolist()
                }
                predictions.append(prediction)

    return images_to_visualize, image_files_to_visualize, predictions