

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

model_bh = BoxHead()
model_bh.load_state_dict(torch.load('model_trained_boxhead.pth', map_location="cuda:2"))
model_bh.eval().to(device)

cnt = 0
cnt_fig = 1
plt.figure(figsize=(25,10))
for i, batch in enumerate(test_loader):
    if i < 7:
        continue
    images=batch['images'].to(device)
    index=batch['index']
    bounding_boxes=batch['bboxes']
    gt_label = batch['labels']

    model_bh.backbone.eval()
    model_bh.rpn.eval()

    fpn_feat_list = list(model_bh.backbone(images).values())

    plt.subplot(3,(len(fpn_feat_list)+1),cnt_fig)
    plt.imshow(images[0].cpu().permute(1,2,0).numpy())
    plt.title("Original Image")
    plt.axis('off')
    cnt_fig += 1

    for i in range(len(fpn_feat_list)):
        plt.subplot(3,len(fpn_feat_list)+1,cnt_fig)
        plt.title("FPN"+str(i))
        x = model_bh.rpn.head.conv(fpn_feat_list[i])
        plt.imshow(x[0,1,:,:].detach().cpu().numpy())
        plt.axis('off')
        cnt_fig += 1

    cnt += 1

    if cnt == 3:
        break
plt.show()

count = 0
temp_flag = True

for i, each_set in enumerate(test_loader):

  images=each_set['images'].to(device)
  index=each_set['index']
  bounding_boxes=each_set['bboxes']
  gt_label = each_set['labels']

  model_bh.backbone.eval()
  model_bh.rpn.eval()
  with torch.no_grad():
    backout = model_bh.backbone(images)
    im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    rpnout = model_bh.rpn(im_lis, backout)
  proposals=[proposal[0:model_bh.keep_topK,:] for proposal in rpnout[0]]
  fpn_feat_list = list(backout.values())
  #print()
  feature_vectors = model_bh.MultiScaleRoiAlign_boxhead(fpn_feat_list,proposals)
      
  class_logits, box_pred = model_bh.forward(feature_vectors, eval=True)

  boxes, scores, labels = model_bh.postprocess_detections(class_logits, box_pred, proposals,conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=3)
  if len(boxes) == 0:
    continue
  for i in range(4):
    visualizer_top_proposals(images[i].permute(1,2,0), boxes[i], labels[i])

    count += 1
    if count > 10 :
      temp_flag = False
      break

dataset_map = BuildDataset(path)
dataset_size = len(dataset_map)
required_size = 1000 + 200
subset_indices = torch.randperm(dataset_size)[:required_size]
dataset_map = torch.utils.data.Subset(dataset_map, subset_indices) # type: ignore
train_size = 1000
test_size = 200

train_dataset, test_dataset = torch.utils.data.random_split(dataset_map, [train_size, test_size])

batch_size_map = 1
train_build_loader_map = BuildDataLoader(train_dataset, batch_size=batch_size_map, shuffle=False, num_workers=0)
train_loader_map = train_build_loader_map.loader()
test_build_loader_map = BuildDataLoader(test_dataset, batch_size=batch_size_map, shuffle=False, num_workers=0)
test_loader_map = test_build_loader_map.loader()


tp_all = dict({'1':[],'2':[],'3':[]})
scores_all = dict({'1':[],'2':[],'3':[]})
gt_label_all = dict({'1':[],'2':[],'3':[]})

for i, each_set in enumerate(test_loader_map):
  images=each_set['images'].to(device)
  index=each_set['index']
  bounding_boxes=each_set['bboxes']
  gt_label = each_set['labels']

  model_bh.backbone.eval()
  model_bh.rpn.eval()
  with torch.no_grad():
    backout = model_bh.backbone(images)
    im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    rpnout = model_bh.rpn(im_lis, backout)
  proposals=[proposal[0:model_bh.keep_topK,:] for proposal in rpnout[0]]
  fpn_feat_list = list(backout.values())
  feature_vectors = model_bh.MultiScaleRoiAlign_boxhead(fpn_feat_list,proposals)
      
  class_logits, box_pred = model_bh.forward(feature_vectors, eval=True)
  
  boxes, scores, labels = model_bh.postprocess_detections(class_logits, box_pred, proposals,conf_thresh=0.3, keep_num_preNMS=500, keep_num_postNMS=3)
  gt_label_one = gt_label[0] - 1
  for j in gt_label_one:
    label_idx = j.detach().cpu().item() + 1
    gt_label_all[str(label_idx)].append(label_idx)

  iou_all = ops.box_iou(boxes[0], bounding_boxes[0])

  for j in range(labels[0].shape[0]):
    one_pred_iou = iou_all[j]
    clas_check = gt_label_one.detach().cpu() == labels[0][j]
    pred_ious_clas = one_pred_iou[clas_check]

    check = pred_ious_clas > 0.5
    idx = labels[0][j].detach().cpu().item() + 1
    if check.sum() == 1:
      tp_all[str(idx)].append(1)
      scores_all[str(idx)].append(scores[0][j].detach().cpu().item())
    else:
      tp_all[str(idx)].append(0)
      scores_all[str(idx)].append(scores[0][j].detach().cpu().item())

map_val_1, rec1, prec1 = prec_rec(scores_all['1'], gt_label_all['1'],tp_all['1'])
map_val_2, rec2, prec2 = prec_rec(scores_all['2'], gt_label_all['2'],tp_all['2'])
map_val_3, rec3, prec3 = prec_rec(scores_all['3'], gt_label_all['3'],tp_all['3'])

print("MAP 1: ", map_val_1)
print("MAP 2: ", map_val_2)
print("MAP 3: ", map_val_3)

print("Average MAP : ", (map_val_1+map_val_2+map_val_3)/3)

fig = plt.figure(figsize =(11, 9))
plt.title("Precision vs. Recall", size=20)
plt.xlabel('Recall', size=15)
plt.ylabel('Precision', size =15)

plt.plot(rec1, prec1, label ="Vehicle (Class 0)", color="blue")
plt.plot(rec2, prec2, label = "Human (Class 1)", color="green")
plt.plot(rec3, prec3, label = "Animals (Class 2)", color="red")
plt.legend(loc="upper right")
plt.show()

