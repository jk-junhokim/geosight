import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_predictions_grid(images, predictions, num_rows, num_cols, confidence_threshold):
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*5, num_rows*5))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= len(images):
            ax.axis('off')
            continue

        img = images[idx]
        prediction = predictions[idx]
        boxes = np.array(prediction['boxes'])
        scores = np.array(prediction['scores'])

        ax.imshow(img)
        ax.axis('off')

        # Viz predicted boxes
        for i, box in enumerate(boxes):
            score = scores[i]
            if score >= confidence_threshold:  # threshold
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # score
                ax.text(x_min, y_min - 5, f'{score:.2f}', color='red', fontsize=8, backgroundcolor='white')
    plt.tight_layout()
    plt.show()

# Set the grid 9x9
num_rows, num_cols = 9, 9
confidence_threshold = 0.5

# visualize_predictions_grid(images_to_visualize, predictions[:81], num_rows, num_cols, confidence_threshold)


# BBox viz
def visualize_predictions(model, test_loader, device, num_images=21):
    model.eval()
    images_so_far = 0

    # Figure (3, 7)
    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(10, 20))
    axes = axes.flatten()

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                if images_so_far >= num_images:
                    break
                
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                gt_boxes = targets[i]['boxes'].cpu().numpy()
                pred_boxes = outputs[i]['boxes'].cpu().numpy()
                pred_scores = outputs[i]['scores'].cpu().numpy()
                
                ax = axes[images_so_far]
                ax.imshow(img)
                ax.axis('off')

                # Ground truth boxes (green)
                for box in gt_boxes:
                    x_min, y_min, x_max, y_max = box
                    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                             linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)

                # Rredicted boxes (red)
                for idx, box in enumerate(pred_boxes):
                    score = pred_scores[idx]
                    if score < 0.8: # skip low confidence boxes
                        continue
                    x_min, y_min, x_max, y_max = box
                    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                             linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    # Confidence score
                    ax.text(x_min, y_min, f'{score:.2f}', color='white', fontsize=8, backgroundcolor='red')

                images_so_far += 1
                if images_so_far >= num_images:
                    break

            if images_so_far >= num_images:
                break

    plt.tight_layout()
    plt.show()