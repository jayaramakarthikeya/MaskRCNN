
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import metrics
import torchvision
from PIL import ImageColor

def visualizer_top_proposals_first(image, boxes, labels):

    boxes = boxes.detach().cpu().numpy()
    image = image.detach().cpu().numpy()
    image = np.clip(image, 0., 255.)

    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot()

    ax.imshow(image)
    for i in range(len(boxes)):
      if labels[i] == 0 :
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='white')
      elif labels[i] == 1:
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='blue')
      elif labels[i] == 2:
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='green')
      elif labels[i] == 3:
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='red')
      ax.add_patch(rect)

    plt.show()


def visualizer_top_proposals(image, boxes, labels):
    labels = np.array(labels)
    boxes = boxes.detach().cpu().numpy()

    image = image.detach().cpu().numpy()
    image = np.clip(image, 0., 255.)

    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot()

    ax.imshow(image)
    for i in range(len(boxes)):

      if labels[i] == 0 :
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='blue')
        ax.add_patch(rect)
      elif labels[i] == 1:
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='green')
        ax.add_patch(rect)
      elif labels[i] == 2:
        rect = patches.Rectangle((boxes[i][0], boxes[i][1],), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1] , fill=False,color='red')
        ax.add_patch(rect)

def prec_rec(scores_each, gt_labels_each, tp_each):
  sorted_scores_idx = np.argsort(np.array(scores_each))
  sorted_tp = np.array(tp_each)[sorted_scores_idx][::-1]
  rec = []
  pre = []

  fp_tot = 0
  tp_tot = 0
  rec = []
  prec = []
  for i,one_val in enumerate(sorted_tp):
    if one_val == 0:
      fp_tot += 1
    elif one_val == 1:
      tp_tot += 1

    one_prec = tp_tot/(tp_tot+fp_tot)
    one_rec = tp_tot/len(gt_labels_each)
    rec.append(one_rec)
    prec.append(one_prec)
    if one_rec == 1:
      break

  return metrics.auc(rec,prec), rec, prec

def visualize_raw_processor(img, mask,label, alpha=0.5):
    processed_mask = mask.clone().detach().squeeze().bool()
    img = img.clone().detach()[:, :, 11:-11]

    inv_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[ 0., 0., 0. ], std=[1/0.229, 1/0.224, 1/0.255]), torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    pad = torch.nn.ZeroPad2d((11,11,0,0))
    processed_img = pad(inv_transform(img))

    processed_img = processed_img.detach().cpu().numpy()

    processed_img = np.clip(processed_img, 0, 1)
    processed_img = torch.from_numpy((processed_img * 255.).astype(np.uint8))

    img_to_draw = processed_img.detach().clone()

    if processed_mask.ndim == 2:
        processed_mask = processed_mask[None, :, :]
    for mask, lab in zip(processed_mask, label):
        if lab.item() == 0 :            # vehicle
            colored = 'blue'
            color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
            img_to_draw[:, mask] = color[:, None]
        if lab.item() == 1 :            # person
            colored = 'green'
            color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
            img_to_draw[:, mask] = color[:, None]
        if lab.item() == 2 :            # animal
            colored = 'red'
            color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
            img_to_draw[:, mask] = color[:, None]
 
    out = (processed_img * (1 - alpha) + img_to_draw * alpha).to(torch.uint8)
    # out = draw_bounding_boxes(out, bbox, colors='red', width=2)
    final_img = out.numpy().transpose(1,2,0)
    plt.figure()
    plt.imshow(final_img)
    return final_img