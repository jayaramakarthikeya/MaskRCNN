
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
from model import RPNHead

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

model = RPNHead()
model.load_state_dict(torch.load('model_rpn_trial.pth', map_location=device))
model.eval()
model.to(device)

count = 0
temp_flag = True

for i, each_set in enumerate(test_loader):
  images=each_set['images'].to(device)
  index=each_set['index']
  bounding_boxes=each_set['bboxes']
  with torch.no_grad():
    backbone = model.resnet50_fpn(images)
    backbone_list = [backbone["0"], backbone["1"], backbone["2"], backbone["3"], backbone["pool"]]
    logits, bbox_regs = model.forward(backbone_list)

  nms_clas_list, nms_prebox_list = model.postprocess(logits, bbox_regs, IOU_thresh=0.5, keep_num_preNMS=2000, keep_num_postNMS=100)
  for i in range(4):
    visualizer_top_proposals(images[i].permute(1,2,0), nms_prebox_list[i])
    count += 1
  if count > 10 :
    temp_flag = False
    break