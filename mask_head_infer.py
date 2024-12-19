
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model import BoxHead
from datasets import BuildDataset, BuildDataLoader
import warnings
from utils import  visualizer_top_proposals, prec_rec
from torchvision.models.detection.image_list import ImageList
from torchvision import ops
from model import MaskHead
from utils import visualize_raw_processor

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

path = {}
path["images"] = "./data/hw3_mycocodata_img_comp_zlib.h5" 
path["labels"] = "./data/hw3_mycocodata_labels_comp_zlib.npy"
path["bboxes"] = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
path["masks"]  = "./data/hw3_mycocodata_mask_comp_zlib.h5"

torch.manual_seed(42)
dataset = BuildDataset(path)
full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 4
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

model_mh = MaskHead()
model_mh.load_state_dict(torch.load('model_trained_maskhead.pth', map_location=device))
model_mh.eval().to(device)

count = 0
temp_flag = True

for i, each_set in enumerate(test_loader):
  if i < 5:
    continue
  images=each_set['images'].to(device)
  index=each_set['index']
  bounding_boxes=each_set['bboxes']
  gt_label = each_set['labels']
  masks = [ mask.to(device) for mask in each_set['masks']]

  model_mh.backbone.eval()
  model_mh.rpn.eval()
  with torch.no_grad():
    backout = model_mh.backbone(images)
    im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    rpnout = model_mh.rpn(im_lis, backout)

  proposals=[proposal[0:model_mh.keep_topK,:] for proposal in rpnout[0]]
  fpn_feat_list = list(backout.values())
  feature_vectors = model_mh.model_boxhead.MultiScaleRoiAlign_boxhead(fpn_feat_list,proposals)
  with torch.no_grad():
    class_logits, box_pred = model_mh.model_boxhead.forward(feature_vectors, eval=True)

  boxes, scores, labels, gt_masks = model_mh.preprocess_ground_truth_creation(proposals, class_logits, box_pred, gt_label,bounding_boxes ,masks , IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=3)
  if len(boxes) == 0: 
    continue
  mask_feature_vectors = model_mh.MultiScaleRoiAlign_maskhead(fpn_feat_list,boxes)
  mask_feature_vectors_flattened = model_mh.flatten_inputs(mask_feature_vectors)

  mask_preds = model_mh.forward(mask_feature_vectors_flattened)

  output_masks_final = model_mh.postprocess_mask(mask_preds, labels)

  for k in range(len(output_masks_final)):
  
    visualize_raw_processor(images[k], output_masks_final[k], labels[k])
    plt.show()
    count += 1
    if count > 10 :
      temp_flag = False
      break

  if temp_flag == False:
    break
