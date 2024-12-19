import torch
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes
from PIL import ImageColor
import pytorch_lightning as pl
from utils import pretrained_models_680
import torch.nn as nn
from utils import MultiApply, conv_box_to_corners, conv_box_to_xywh, output_decoding_postprocess
from torchvision.models.detection.image_list import ImageList
from datasets import BuildDataLoader, BuildDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.loggers import tensorboard

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class BoxHead(pl.LightningModule):
    _default_cfg = {
        'classes': 3,
        'P': 7
    }

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in {**self._default_cfg, **kwargs}.items():
            setattr(self, k, v)

        self.C=self._default_cfg['classes']
        self.P=self._default_cfg['P']
        self.image_size = (800,1088)
        self.pretrained_path = 'checkpoint680.pth'
        with torch.no_grad():
          self.backbone, self.rpn = pretrained_models_680(self.pretrained_path)
        self.keep_topK = 200

        self.train_loss_epoch = []
        self.train_class_loss_epoch = []
        self.train_reg_loss_epoch = []
        self.training_step_outputs = []

        self.val_loss_epoch = []
        self.val_class_loss_epoch = []
        self.val_reg_loss_epoch = []
        self.val_outputs = []

        # TODO initialize BoxHead
        self.intermediate_layer = nn.Sequential(
            nn.Linear(in_features=256*self.P*self.P, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU()
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=self.C+1)
    
        )

        self.regressor_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4*self.C)
        )
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, padding='same'),
            nn.Sigmoid()
        )
        
    def create_ground_truth(self,proposals,gt_labels,bbox):
          """     
          This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
          Input:
            proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            gt_labels: list:len(bz) {(n_obj)}
            bbox: list:len(bz){(n_obj, 4)}
          Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
            labels: (total_proposals,1) (the class that the proposal is assigned)
            regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
          """        
          all_labels = []
          all_regressor_targets = []
          for per_image_proposal, per_image_gt_label, per_image_bbox in zip(proposals, gt_labels, bbox):
              per_image_gt_label = per_image_gt_label.clone().detach().to(device).float()
              each_labels = (-1.)*torch.ones(per_image_proposal.shape[0]).float().to(device)
              each_regressor_target = torch.ones(per_image_proposal.shape[0], 4).to(device)
              per_image_bbox = per_image_bbox.to(device)
              per_image_proposal = per_image_proposal.to(device)
              iou = torchvision.ops.box_iou(per_image_proposal, per_image_bbox)
              per_image_proposal = conv_box_to_xywh(per_image_proposal)
              per_image_bbox = conv_box_to_xywh(per_image_bbox)

              max_iou, max_iou_idx = torch.max(iou, dim=1)
              max_iou_idx = max_iou_idx.long()
              above_thres = torch.where(max_iou > 0.4)

              if len(above_thres[0]) != 0. : 
                  each_labels[above_thres[0].long()] = torch.stack([per_image_gt_label[i] for i in max_iou_idx[above_thres[0].long()]])
                  # print("Inside loss iteration : ", torch.where(each_labels < 0.)[0].shape)
                  each_regressor_target[above_thres[0].long()] = torch.stack([per_image_bbox[i] for i in max_iou_idx[above_thres[0].long()]])

                  conv_each_regressor_target = torch.zeros_like(per_image_proposal)
                  conv_each_regressor_target[:,0] = (each_regressor_target[:,0] - per_image_proposal[:,0]) / per_image_proposal[:,2]
                  conv_each_regressor_target[:,1] = (each_regressor_target[:,1] - per_image_proposal[:,1]) / per_image_proposal[:,3]
                  conv_each_regressor_target[:,2] = torch.log(each_regressor_target[:,2]/per_image_proposal[:,2])
                  conv_each_regressor_target[:,3] = torch.log(each_regressor_target[:,3]/per_image_proposal[:,3])

                  all_labels.append(each_labels)
                  all_regressor_targets.append(conv_each_regressor_target)
              else : 
                  # print("Entering inside")
                  all_labels.append(each_labels)
                  all_regressor_targets.append(each_regressor_target)


          labels = torch.cat(all_labels, dim=0)
          background_mask = labels < 0.
          regressor_target = torch.cat(all_regressor_targets, dim=0)
          regressor_target[background_mask] = 0.
          return labels,regressor_target



    def MultiScaleRoiAlign_boxhead(self, fpn_feat_list,proposals,P=7):
        """    
        This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
        a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
        Input:
            fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
            proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            P: scalar
        Output:
            feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        """
        roi_alls = []
        for i in range(len(proposals)):
            each_proposal= conv_box_to_xywh(proposals[i])
            k = (torch.log2(torch.sqrt(each_proposal[:,2]*each_proposal[:,3])/224.) + 4).int()
            k = torch.clamp(k, min=2., max=5.).int()
            each_proposal = conv_box_to_corners(each_proposal)

            scaling_vals = torch.pow(2,k).reshape(-1,1)*torch.ones_like(each_proposal)
            scaled_proposals = each_proposal / scaling_vals

            fpn_list_each_proposal = [fpn_feat_list[j][i].unsqueeze(0) for j in range(5)]
            roi_vals = torch.stack([torchvision.ops.roi_align(fpn_list_each_proposal[k[n]-2], [scaled_proposals[n].view(1,4)], (P,P)).squeeze(0) for n in range(k.shape[0])], dim=0).reshape(-1,256*P*P)
            roi_alls.append(roi_vals)
        feature_vectors = torch.cat(roi_alls, dim=0)

        return feature_vectors

    def MultiScaleRoiAlign_maskhead(self, fpn_feat_list,proposals,P=14):
        """    
        This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
        a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
        Input:
            fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
            proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            P: scalar
        Output:
            feature_vectors: (total_proposals, 256,P,P)  (make sure the ordering of the proposals are the same as the ground truth creation)
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        """
        roi_alls = []
        for i in range(len(proposals)):
            each_proposal= conv_box_to_xywh(proposals[i])
            k = (torch.log2(torch.sqrt(each_proposal[:,2]*each_proposal[:,3])/224.) + 4).int()
            k = torch.clamp(k, min=2., max=5.).int()
            each_proposal = conv_box_to_corners(each_proposal)

            scaling_vals = torch.pow(2,k).reshape(-1,1)*torch.ones_like(each_proposal)
            scaled_proposals = each_proposal / scaling_vals

            fpn_list_each_proposal = [fpn_feat_list[j][i].unsqueeze(0) for j in range(5)]
            roi_vals = torch.stack([torchvision.ops.roi_align(fpn_list_each_proposal[k[n]-2], [scaled_proposals[n].view(1,4)], (P,P)).squeeze(0) for n in range(k.shape[0])], dim=0)
            roi_alls.append(roi_vals)
        feature_vectors = torch.cat(roi_alls, dim=0)
        return feature_vectors



    def greedy_nms(self,clases,boxes,scores, IOU_thres=0.5, keep_num_postNMS=100):
        # Input:
        #       clases: (num_preNMS, )
        #       boxes:  (num_preNMS, 4)
        #       scores: (num_preNMS,)
        # Output:
        #       boxes:  (post_NMS_boxes_per_image,4) ([x1,y1,x2,y2] format)
        #       scores: (post_NMS_boxes_per_image)   ( the score for the top class for the regressed box)
        #       labels: (post_NMS_boxes_per_image)  (top category of each regressed box)
        ###########################################################

        scores_all = [[],[],[]]
        boxes_all = [[],[],[]]
        clas_all = [[],[],[]]

        for i in range(3):
            each_label_idx = torch.where(clases == i)[0]
            if len(each_label_idx) == 0:
              continue
            each_clas_boxes = boxes[each_label_idx]
            each_clas_score = scores[each_label_idx]

            start_x_torched = each_clas_boxes[:, 0]
            start_y_torched = each_clas_boxes[:, 1]
            end_x_torched   = each_clas_boxes[:, 2]
            end_y_torched   = each_clas_boxes[:, 3]

            areas_torched = (end_x_torched - start_x_torched + 1) * (end_y_torched - start_y_torched + 1)

            order_torched = torch.argsort(each_clas_score)

            while len(order_torched) > 0:
                # The index of largest confidence score
                index = order_torched[-1]
                
                # Pick the bounding box with largest confidence score
                boxes_all[i].append(boxes[index].detach())
                scores_all[i].append(each_clas_score[index].detach())

                if len(boxes_all[i]) == keep_num_postNMS:
                    break

                # Compute ordinates of intersection-over-union(IOU)
                x1 = torch.maximum(start_x_torched[index], start_x_torched[order_torched[:-1]])
                x2 = torch.minimum(end_x_torched[index], end_x_torched[order_torched[:-1]])
                y1 = torch.maximum(start_y_torched[index], start_y_torched[order_torched[:-1]])
                y2 = torch.minimum(end_y_torched[index], end_y_torched[order_torched[:-1]])

                # Compute areas of intersection-over-union
                w = torch.maximum(torch.tensor([0]), x2 - x1 + 1)
                h = torch.maximum(torch.tensor([0]), y2 - y1 + 1)
                intersection = w * h

                # Compute the ratio between intersection and union
                ratio = intersection / (areas_torched[index] + areas_torched[order_torched[:-1]] - intersection)
                left = torch.where(ratio < IOU_thres)[0]
                order_torched = order_torched[left]
            clas_all[i] = [i]*len(scores_all[i])
            
        fin_scores = torch.cat([torch.tensor(one_score).reshape(-1,1) for one_score in scores_all if len(one_score)!=0],dim=0).reshape(-1,1)
        fin_boxes = torch.cat([torch.stack(one_box) for one_box in boxes_all if len(one_box)!=0]).reshape(-1,4)
        fin_clas = torch.cat([torch.tensor(one_clas) for one_clas in clas_all if len(one_clas)!=0]).reshape(-1,1)
        return fin_clas, fin_scores, fin_boxes



    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=20):
        # This function does the post processing for the results of the Box Head for a batch of images
        # Use the proposals to distinguish the outputs from each image
        # Input:
        #       class_logits: (total_proposals,(C+1))
        #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
        #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
        #       conf_thresh: scalar
        #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
        #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
        # Output:
        #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
        #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
        #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        
        class_logits = class_logits.cpu()
        box_regression = box_regression.cpu()
        num_proposals = proposals[0].shape[0]
        boxes = []
        scores = []
        labels = []
        for i, each_proposal in enumerate(proposals):
            each_proposal = each_proposal.cpu()
            one_image_boxes = box_regression[i*num_proposals:(i+1)*num_proposals]          # Shape (num_proposals, 12)
            one_image_logits = class_logits[i*num_proposals:(i+1)*num_proposals]           # Shape (num_proposals, 4)
            one_image_scores, one_image_label = torch.max(one_image_logits, dim=1)
            one_image_label = one_image_label.clone().int() - 1
            non_bg_label_idx = torch.where(one_image_label >= 0)[0].cpu()


            if len(non_bg_label_idx) != 0: 
                class_labels = one_image_label[non_bg_label_idx]
                all_class_boxes = one_image_boxes[non_bg_label_idx]
                class_boxes =  torch.stack([all_class_boxes[i, x*4:(x+1)*4] for i, x in enumerate(class_labels)])      # Shape(filtered_labels, 4) ([t_x,t_y,t_w,t_h])
                
                decoded_boxes = output_decoding_postprocess(class_boxes, each_proposal[non_bg_label_idx])                          # (x1,y1,x2,y2)
                decoded_boxes = decoded_boxes.cpu()

                valid_boxes_idx = torch.where((decoded_boxes[:,0] >= 0) & (decoded_boxes[:,2] < 1088) & (decoded_boxes[:,1] > 0) & (decoded_boxes[:,3] < 800))

                valid_boxes = decoded_boxes[valid_boxes_idx]
                valid_clases = one_image_label[non_bg_label_idx][valid_boxes_idx]
                valid_scores = one_image_scores[non_bg_label_idx][valid_boxes_idx]
                sorted_scores_pre_nms, sorted_idx = torch.sort(valid_scores, descending=True)
                sorted_clases_pre_nms = valid_clases[sorted_idx]
                sorted_boxes_pre_nms = valid_boxes[sorted_idx]
                
                if len(sorted_clases_pre_nms) > keep_num_preNMS:
                    clases_pre_nms = sorted_clases_pre_nms[:keep_num_preNMS]
                    boxes_pre_nms = sorted_boxes_pre_nms[:keep_num_preNMS]
                    scores_pre_nms = sorted_scores_pre_nms[:keep_num_preNMS]
                else:
                    clases_pre_nms = sorted_clases_pre_nms
                    boxes_pre_nms = sorted_boxes_pre_nms
                    scores_pre_nms = sorted_scores_pre_nms
                clases_post_nms, scores_post_nms, boxes_post_nms = self.greedy_nms(clases_pre_nms, boxes_pre_nms, scores_pre_nms, IOU_thres=conf_thresh, keep_num_postNMS=keep_num_postNMS)
            boxes.append(boxes_post_nms)
            scores.append(scores_post_nms)
            labels.append(clases_post_nms)

        return boxes, scores, labels


    def training_step(self, batch, batch_idx):
        images=batch['images'].to(device)
        index=batch['index']
        bounding_boxes=batch['bboxes']
        gt_label = batch['labels']

        self.backbone.eval()
        self.rpn.eval()
        with torch.no_grad():
          backout = self.backbone(images)
          im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
          rpnout = self.rpn(im_lis, backout)
        proposals=[proposal[0:self.keep_topK,:] for proposal in rpnout[0]]
        fpn_feat_list = list(backout.values())
        feature_vectors = self.MultiScaleRoiAlign_boxhead(fpn_feat_list,proposals)
        lbls, rgrssor_targ = self.create_ground_truth(proposals, gt_label, bounding_boxes)
            
        class_logits, box_pred = self.forward(feature_vectors)

        loss, loss_class, loss_regr = self.compute_loss(class_logits.to(device), box_pred.to(device), lbls.to(device), rgrssor_targ.to(device), l=0.2, effective_batch=100)
    
        del lbls, bounding_boxes, index, gt_label
        del rgrssor_targ
        torch.cuda.empty_cache()

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("loss_class", loss_class, prog_bar=True)
        self.log("loss_regr", loss_regr, prog_bar=True)

        self.training_step_outputs.append({"loss":loss.detach().cpu().item(), "classifier_loss":loss_class.detach().cpu().item(), "regressor_loss":loss_regr.detach().cpu().item()})

        return {"loss":loss, "classifier_loss":loss_class, "regressor_loss":loss_regr}

    def on_train_epoch_end(self):
        avg_train_loss = 0
        avg_class_loss = 0
        avg_reg_loss = 0
        for i in range(len(self.training_step_outputs)):
            avg_train_loss += self.training_step_outputs[i]["loss"]
            avg_class_loss += self.training_step_outputs[i]["classifier_loss"]
            avg_reg_loss += self.training_step_outputs[i]["regressor_loss"]

        avg_train_loss = avg_train_loss / len(self.training_step_outputs)
        avg_class_loss = avg_class_loss / len(self.training_step_outputs)
        avg_reg_loss = avg_reg_loss / len(self.training_step_outputs)

        self.train_loss_epoch.append(avg_train_loss)
        self.train_class_loss_epoch.append(avg_class_loss)
        self.train_reg_loss_epoch.append(avg_reg_loss)

        self.training_step_outputs = []


    def validation_step(self, batch, batch_idx):
        images=batch['images']
        index=batch['index']
        bounding_boxes=batch['bboxes']
        gt_label = batch['labels']

        self.backbone.eval()
        self.rpn.eval()
        with torch.no_grad():
          backout = self.backbone(images)
          im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
          rpnout = self.rpn(im_lis, backout)
        proposals=[proposal[0:self.keep_topK,:] for proposal in rpnout[0]]
        fpn_feat_list = list(backout.values())
        feature_vectors = self.MultiScaleRoiAlign_boxhead(fpn_feat_list,proposals)
        lbls, rgrssor_targ = self.create_ground_truth(proposals, gt_label, bounding_boxes)
            
        class_logits, box_pred = self.forward(feature_vectors)

        val_loss, loss_class, loss_regr = self.compute_loss(class_logits.to(device), box_pred.to(device), lbls.to(device), rgrssor_targ.to(device), l=0.2, effective_batch=100)

        del lbls, bounding_boxes, index, gt_label
        del rgrssor_targ
        torch.cuda.empty_cache()

        self.log("val_loss", val_loss)
        self.val_outputs.append({"loss":val_loss, "classifier_loss":loss_class.detach().cpu().item(), "regressor_loss":loss_regr.detach().cpu().item()})
        
        return {"loss":val_loss, "classifier_loss":loss_class, "regressor_loss":loss_regr}

    def on_validation_epoch_end(self):
        avg_train_loss = 0
        avg_class_loss = 0
        avg_reg_loss = 0
        for i in range(len(self.val_outputs)):
            avg_train_loss += self.val_outputs[i]["loss"]
            avg_class_loss += self.val_outputs[i]["classifier_loss"]
            avg_reg_loss += self.val_outputs[i]["regressor_loss"]

        avg_train_loss /= len(self.val_outputs)
        avg_class_loss /= len(self.val_outputs)
        avg_reg_loss /= len(self.val_outputs)
        
        self.val_loss_epoch.append(avg_train_loss)
        self.val_class_loss_epoch.append(avg_class_loss)
        self.val_reg_loss_epoch.append(avg_reg_loss)

        self.val_outputs = []


    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(),lr = 0.0005,weight_decay=1.0e-4,momentum=0.90)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[22,28,35], gamma=0.1)

        return [optimizer] , [scheduler]


        #################################################
   # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=0.2,effective_batch=70):
        labels_all = labels.flatten()
        # print("labels all : ", labels_all)
        neg_mask = labels_all < 0.
        pos_mask = labels_all >= 0.
        labels_all = torch.where(labels_all < 0., 0., labels_all)

        num_neg_ind = neg_mask.sum().item()
        num_pos_ind = labels_all.shape[0] - num_neg_ind
        # print("Num neg ind : ", num_neg_ind)
        # print("Num pos ind : ", num_pos_ind)

        if num_pos_ind > (3*effective_batch/4):
            rand_pos_idx = torch.randperm(num_pos_ind)[:int(3*effective_batch/4)]
            rand_neg_idx = torch.randperm(num_neg_ind)[:int(effective_batch/4)]
            pos_clas_tar = labels_all[pos_mask][rand_pos_idx]
            neg_clas_tar = labels_all[neg_mask][rand_neg_idx]
            pos_clas_pred = class_logits[pos_mask][rand_pos_idx]
            neg_clas_pred = class_logits[neg_mask][rand_neg_idx]
            pos_box_pred = box_preds[pos_mask][rand_pos_idx]
            pos_box_tar = regression_targets[pos_mask][rand_pos_idx]
        else:
            rand_neg_idx = torch.randperm(num_neg_ind)
            pos_clas_tar = labels_all[pos_mask]
            neg_clas_tar = labels_all[neg_mask][rand_neg_idx][:(effective_batch - num_pos_ind)]
            pos_clas_pred = class_logits[pos_mask]
            neg_clas_pred = class_logits[neg_mask][rand_neg_idx][:(effective_batch - num_pos_ind)]
            pos_box_pred = box_preds[pos_mask]
            pos_box_tar = regression_targets[pos_mask]


        clas_preds = torch.vstack((pos_clas_pred,neg_clas_pred)).float()
        clas_tar   = torch.cat((pos_clas_tar,neg_clas_tar)).reshape(-1).long()
        one_hot_clas_tar = torch.nn.functional.one_hot(clas_tar, num_classes=4).float()

        clas_criterion = torch.nn.CrossEntropyLoss()
        clas_loss = clas_criterion(clas_preds, one_hot_clas_tar)

        class_labels_cor = (pos_clas_tar.clone() - 1).int()
        if num_pos_ind == 0:
          loss_regr = 0.
        else:
          box_regression_preds = torch.stack([one_box_regression[4*(class_labels_cor[i]):4*class_labels_cor[i]+4] for i, one_box_regression in enumerate(pos_box_pred)])
          reg_criterion = torch.nn.SmoothL1Loss(reduction = 'sum')
          loss_regr = reg_criterion(box_regression_preds,pos_box_tar) 

        loss = clas_loss + l*loss_regr 

        return loss, clas_loss, loss_regr
        ###################################################



    def forward(self, feature_vectors, eval=False):
        """
        Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
        Input:
              feature_vectors: (total_proposals, 256*P*P)
        Outputs:
              class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, 
                            notice if you want to use CrossEntropyLoss you should not pass the output through softmax here)
              box_pred:     (total_proposals,4*C)
        """

        #TODO forward through the Intermediate layer
        X = self.intermediate_layer(feature_vectors)

        #TODO forward through the Classifier Head
        class_logits = self.classifier_head(X)

        if eval==True:
          softmax = torch.nn.Softmax(dim = 1)
          class_logits = softmax(class_logits)

        #TODO forward through the Regressor Head
        box_pred = self.regressor_head(X)

        return class_logits, box_pred




