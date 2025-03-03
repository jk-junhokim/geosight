from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class fasterBuilding():
    def __init__(self):
        
        self.model_ft = fasterrcnn_resnet50_fpn(pretrained=True)

        self.in_features = self.model_ft.roi_heads.box_predictor.cls_score.in_features

        self.model_ft.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, num_classes=2)