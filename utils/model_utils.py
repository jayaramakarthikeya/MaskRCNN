import numpy as np
import torch
import torchvision
from sklearn import metrics
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes
from PIL import ImageColor

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

def pretrained_models_680(checkpoint_file,eval=True):
    import torchvision
    model_fpn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    if(eval):
        model_fpn.eval()
    model_fpn.to(device)

    backbone = model_fpn.backbone
    rpn = model_fpn.rpn

    if(eval):
        backbone.eval()
        rpn.eval()

    rpn.nms_thresh=0.6
    checkpoint = torch.load(checkpoint_file, device)

    backbone.load_state_dict(checkpoint['backbone'])
    rpn.load_state_dict(checkpoint['rpn'])

    return backbone, rpn