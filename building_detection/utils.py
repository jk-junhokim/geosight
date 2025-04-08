import torch
import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from dataloader import *
from utils import *


# Calculate Intersection over Union between two boxes
def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def compute_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    # Sort predictions by descending confidence
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = np.zeros(len(gt_boxes))

    for i, pred_box in enumerate(pred_boxes):
        ious = np.array([calculate_iou(pred_box, gt_box) for gt_box in gt_boxes])
        max_iou = np.max(ious) if len(ious) > 0 else 0
        max_iou_idx = np.argmax(ious) if len(ious) > 0 else -1

        if max_iou >= iou_threshold and matched_gt[max_iou_idx] == 0:
            tp[i] = 1
            matched_gt[max_iou_idx] = 1  # mark GT as matched
        else:
            fp[i] = 1

    # Compute cumulative true positives and false positives
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    # recalls = cum_tp / len(gt_boxes)
    # precisions = cum_tp / (cum_tp + cum_fp)

    recalls = cum_tp / len(gt_boxes) if len(gt_boxes) > 0 else np.array([0])
    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)

    # Ensure PR curve range
    recalls = np.concatenate(([0.], recalls))
    precisions = np.concatenate(([1.], precisions))

    # Compute AP as area under the PR curve
    ap = np.trapz(precisions, recalls)
    return ap, precisions, recalls

# Eval model on the test set and calculate mAP
def evaluate_model_on_test_set(model, test_loader, device, iou_threshold=0.5):
    model.eval()
    all_gt_boxes = []
    all_pred_boxes = []
    all_pred_scores = []

    binary_threshold = 0.8
    binary_true = []
    binary_pred = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                all_gt_boxes.extend(gt_boxes)

                pred_boxes = outputs[i]['boxes'].cpu().numpy()
                pred_scores = outputs[i]['scores'].cpu().numpy()

                all_pred_boxes.extend(pred_boxes)
                all_pred_scores.extend(pred_scores)

                # TODO: Test this evaluation methods
                gt_label = 1 if len(gt_boxes) > 0 else 0
                # Predict positive if any predicted box score > threshold
                pred_label = 1 if np.any(pred_scores > binary_threshold) else 0

                binary_true.append(gt_label)
                binary_pred.append(pred_label)

    all_gt_boxes = np.array(all_gt_boxes)
    all_pred_boxes = np.array(all_pred_boxes)
    all_pred_scores = np.array(all_pred_scores)

    # Compute AP
    ap, precisions, recalls = compute_ap(all_gt_boxes,
                                        all_pred_boxes, 
                                        all_pred_scores, 
                                        iou_threshold)
    # return ap, precisions, recalls

    # Confusion Matrix, ROC
    cm = confusion_matrix(binary_true, binary_pred)
    fpr, tpr, _ = roc_curve(binary_true, binary_pred)
    prec_curve, rec_curve, _ = precision_recall_curve(binary_true, binary_pred)

    # return ap, precisions, recalls, cm, (fpr, tpr), (prec_curve, rec_curve)
    return ap, precisions, recalls, cm, (fpr, tpr), (prec_curve, rec_curve), binary_true, binary_pred


def plot_precision_recall_curve(precisions, recalls):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()