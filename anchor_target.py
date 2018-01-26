from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import anchor_generator
from util import cython_overlaps

BASE_NETS = ['vgg_16']
MODELS = ['faster_rcnn','ssd']

def get_target(model,base_net,image_size,gt_boxes):
    
    if model not in MODELS:
        raise ValueError('Unknown model.')
    if base_net not in BASE_NETS:
        raise ValueError('Unknown base net.')


    neg_max_iou = 0.3
    pos_min_iou = 0.7

    if model == 'ssd':
        neg_max_iou = 0.5
        pos_min_iou = 0.5

    height = image_size
    width = image_size

    all_anchors = anchor_generator.get_anchor(model = model,base_net = base_net,image_size = image_size)

    num_anchor = all_anchors.shape[0]
    num_box = gt_boxes.shape[0]

    _allowed_border = 100
    
    inds_inside = np.where(
        (all_anchors[:,0] >= -_allowed_border) &
        (all_anchors[:,1] >= -_allowed_border) &
        (all_anchors[:,2] < width + _allowed_border) & 
        (all_anchors[:,3] < height + _allowed_border)
    )[0]
    
    anchors = all_anchors[inds_inside,:]
    
    labels = np.empty((len(inds_inside),),dtype = np.float32)
    labels.fill(-1)
    targets = np.empty((len(inds_inside),),dtype = np.float32)
    targets.fill(-1)

    overlaps = cython_overlaps.bbox_overlaps(
                    np.ascontiguousarray(anchors,dtype = np.float),
                    np.ascontiguousarray(gt_boxes,dtype = np.float))
    
    argmax_overlaps = overlaps.argmax(axis = 1)
    max_overlaps = overlaps[np.arange(len(inds_inside)),argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis = 0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                            np.arange(overlaps.shape[1])]
    arg_gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    
    labels[max_overlaps < neg_max_iou] = 0
    labels[arg_gt_argmax_overlaps] = 1
    labels[max_overlaps >= pos_min_iou] = 1

    check = np.count_nonzero(labels) - len(labels)
    if check == 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return check
    
    ones = np.ones((len(labels)),dtype = np.float32)
    targets = np.where(labels == ones,argmax_overlaps,0)
    targets[gt_argmax_overlaps] = np.arange(num_box)

    labels = _unmap(labels,num_anchor,inds_inside,fill = -1)
    targets = _unmap(targets,num_anchor,inds_inside,fill = 0)
    return labels,targets



def _unmap(data,count,inds,fill = 0,dtype = np.float32):
    if len(data.shape) == 1:
        ret = np.empty((count,),dtype = dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,),dtype = dtype)
        ret.fill(fill)
        ret[inds,:] = data
    return ret
