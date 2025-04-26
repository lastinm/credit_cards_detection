import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import (
#    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
import warnings
import torch
warnings.filterwarnings("ignore", category=UserWarning)
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform


# 1. Кастомный AnchorGenerator для реквизитов карт
def get_anchor_generator():
    # 2. Генератор якорей
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))  # По одному размеру на уровень
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    return AnchorGenerator(anchor_sizes, aspect_ratios)

def create_model(num_classes,checkpoint=None,device='cpu'):
    """
    Create a model for object detection using the Faster R-CNN architecture.

    Parameters:
    - num_classes (int): The number of classes for object detection. (Total classes + 1 [for background class])
    - checkpoint (str) : checkpoint path for the pretrained custom model
    - device (str) : cpu / cuda
    Returns:
    - model : The created model for object detection.
    """

    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        #weights='COCO_V1')
        pretrained=True,
        pretrained_backbone=True)
    
    # Заменяем стандартный AnchorGenerator
    model.rpn.anchor_generator = get_anchor_generator()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = model.to(device)
    return model
