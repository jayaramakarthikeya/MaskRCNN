
import torch
import torchvision

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def IOU(boxA, boxB):
    # This function compute the IOU between two set of boxes 
    boxA = boxA.to(device)
    boxB = boxB.to(device)
    iou = torchvision.ops.box_iou(boxA, boxB)
    return iou

def conv_box_to_xywh(box):
    fin_box = torch.zeros_like(box)
    fin_box[:,0] = (box[:,0] + box[:,2]) / 2
    fin_box[:,1] = (box[:,1] + box[:,3]) / 2
    fin_box[:,2] = (box[:,2] - box[:,0])
    fin_box[:,3] = (box[:,3] - box[:,1])
    return fin_box

# This function computes the IOU between two set of boxes
def IOU_gt(boxA, boxB):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes

    boxA_conv = conv_box_to_corners(boxA)
    iou_torched = IOU(boxA_conv, boxB) 
    ##################################
    return iou_torched

# This function converts x,y,w,h to x1,y1,x2,y2
def conv_box_to_corners(box):
    fin_box = torch.zeros_like(box)
    fin_box[:,0] = box[:,0] - (box[:,2]/2)
    fin_box[:,1] = box[:,1] - (box[:,3]/2)
    fin_box[:,2] = box[:,0] + (box[:,2]/2)
    fin_box[:,3] = box[:,1] + (box[:,3]/2)
    return fin_box

# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding_postprocess(flatten_out,flatten_anchors, device=device):
    #######################################
    # TODO decode the output
    flatten_anchors = conv_box_to_xywh(flatten_anchors)
    conv_box = torch.zeros_like(flatten_anchors).to(device)

    
    conv_box[:,3] = torch.exp(flatten_out[:,3]) * flatten_anchors[:,3]
    conv_box[:,2] = torch.exp(flatten_out[:,2]) * flatten_anchors[:,2]
    conv_box[:,1] = (flatten_out[:,1] * flatten_anchors[:,2]) + flatten_anchors[:,1]
    conv_box[:,0] = (flatten_out[:,0] * flatten_anchors[:,3]) + flatten_anchors[:,0]

    box = conv_box_to_corners(conv_box)

    #######################################
    return box

def output_decoding(flatten_out, flatten_anchors, device='cpu'):
    # This function decodes the output that are given in the [t_x,t_y,t_w,t_h] format
    # into box coordinates where it returns the upper left and lower right corner of the bbox
    # Input:
    #       flatten_out: (total_number_of_anchors*bz,4)
    #       flatten_anchors: (total_number_of_anchors*bz,4)
    # Output:
    #       box: (total_number_of_anchors*bz,4)
    conv_box = torch.zeros_like(flatten_anchors)
    conv_box[:,3] = torch.exp(flatten_out[:,3]) * flatten_anchors[:,3]
    conv_box[:,2] = torch.exp(flatten_out[:,2]) * flatten_anchors[:,2]
    conv_box[:,1] = (flatten_out[:,1] * flatten_anchors[:,3]) + flatten_anchors[:,1]
    conv_box[:,0] = (flatten_out[:,0] * flatten_anchors[:,2]) + flatten_anchors[:,0]

    box = conv_box_to_corners(conv_box)
    return box

def output_flattening(out_r, out_c, anchors):
    # This function flattens the output of the network and the corresponding anchors
    # in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
    # the FPN levels from all the images into 2D matrices
    # Each row correspond of the 2D matrices corresponds to a specific grid cell
    # Input:
    #       out_r: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}
    #       out_c: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
    #       anchors: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
    # Output:
    #       flatten_regr: (total_number_of_anchors*bz,4)
    #       flatten_clas: (total_number_of_anchors*bz)
    #       flatten_anchors: (total_number_of_anchors*bz,4)
    flatten_regr_all = []
    flatten_clas_all = []
    flatten_anchors_all = []
    for level_idx in range(5):
        bz = out_r[level_idx].shape[0]
        flatten_regr = out_r[level_idx].reshape(bz,3,4,out_r[level_idx].shape[-2],out_r[level_idx].shape[-1]).permute(0,1,3,4,2).reshape(-1,4)
        flatten_clas = out_c[level_idx].reshape(-1)
        flatten_anchors = anchors[level_idx].reshape(-1,4).repeat(1,bz)
        flatten_regr_all.append(flatten_regr)
        flatten_clas_all.append(flatten_clas)
        flatten_anchors_all.append(flatten_anchors)
    
    flatten_regr_all = torch.cat(flatten_regr_all)
    flatten_clas = torch.cat(flatten_clas_all)
    flatten_anchors = torch.cat(flatten_anchors_all)

    return flatten_regr_all, flatten_clas, flatten_anchors
