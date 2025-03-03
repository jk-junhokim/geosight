import os, sys
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import zipfile
import argparse

import wandb
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from dataloader import *
from utils import *
from trainer import run_inference


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="gsv", required=True)
args = parser.parse_args()


def main(args,):
    if args.data == "gsv":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)

        checkpoint_path = '/content/drive/My Drive/IDRT_MODEL/IDRT/building_detection/best_model.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_ft = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
        model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

        # Load wseights
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        model_ft.to(device)

        gsv_path = '/content/drive/My Drive/IDRT_MODEL/IDRT/GSV_final.zip'

        # Define the extraction directory
        gsv_data_dir = '/content/gsv_original_images'
        gsv_data_buildings = '/content/gsv_buildings'
        os.makedirs(gsv_data_buildings, exist_ok=True)

        with zipfile.ZipFile(gsv_path, 'r') as zip_ref:
            zip_ref.extractall(gsv_data_dir)
        # print(f'Files extracted to: {gsv_data_dir}')


        inference_dataset = InferenceDataset(root_dir=gsv_final_dir, transform=get_inference_transform())
        inference_loader = DataLoader(inference_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images_to_visualize, image_files_to_visualize, predictions = run_inference(
                model_ft,
                inference_loader,
                device,
                num_images_to_visualize=100,)
        
    elif args.data == "noaa":
        experiment_id = 'exp4'
        checkpoint_dir = f'/scratch/user/junhokim/data/Checkpoints/checkpoints_{experiment_id}'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        checkpoint_dir = checkpoint_dir + '/best_model.pth'
        checkpoint = torch.load(checkpoint_dir, map_location=device)

        model_ft = fasterrcnn_resnet50_fpn(weights=True)
        in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
        model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        model_ft.to(device)
        

        test_file = f'/scratch/user/junhokim/code/datasplits/datasplit_{experiment_id}/test_data.json'

        test_transform = get_test_transform()
        test_dataset = BuildingDataset(json_file=test_file, root_dir=beauty_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        
        
        wandb.init(project="geolocation-building-detection-results", name="building_detection_eval")
        
        iou_threshold = 0.5
        
        def test(model, model_name, iou_threshold):
            # print(f"Evaluating {model_name} on Test Set...")

            ap, precisions, recalls, cm, (fpr, tpr), (prec_curve, rec_curve), binary_true, binary_pred = evaluate_model_on_test_set(model, test_loader, device, iou_threshold)

            print(f"{model_name} AP @ IoU={iou_threshold}: {ap:.4f}")

            # Log AP to W&B
            wandb.log({f"{model_name}_AP": ap})

            # Plot PR curve
            # plt.figure(figsize=(8, 6))
            # plt.plot(recalls, precisions, marker='.', label=f"{model_name}")
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title('Precision-Recall Curve')
            # plt.grid(True)

            pr_fig = plt.gcf()
            
            wandb.log({f"{model_name}_PR_Curve": wandb.Image(pr_fig)})
            plt.close(pr_fig)

            plt.figure()
            plt.plot(fpr, tpr, label=f"{model_name} ROC")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            roc_fig = plt.gcf()
            wandb.log({f"{model_name}_ROC_Curve": wandb.Image(roc_fig)})
            plt.close(roc_fig)

            return ap, precisions, recalls
        
        ap_ft, precisions_ft, recalls_ft = test(model_ft, "Fine_Tuned", iou_threshold)
        
    else:
        print("Invalid Data Type")
        
        
if __name__ == "__main__":
    main()