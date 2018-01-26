from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
from util import anchor_util


def produce_anchor(pool_num,
                    image_size,
                    num_per_cell,
                    base_size,
                    ratios,
                    scales,
                    k = -1,
                    offset = False):
    """
        produce anchors

        Args:
            pool_num: an integer indicating number of pooling layers before this layer
            image_size: an integer indicating image height and width
            num_per_cell: an integer indicating number of boxes per cell
            base_size: an integer indicating basic size of anchor
            ratios: an array for ratios
            scales: an array for scalse
            k: an integer, if k != -1, k is the kth feature map in ssd(i.e. with k == 0 as layer conv4_3)


        Returns:
            anchors: an array [num_anchors,4], coordinates of anchors,coordinates in [image_size,image_size]
    """
    base_anchor = np.array([1.,1.,base_size,base_size]) - 1
    ratio_anchors = anchor_util.ratio_enum(base_anchor,ratios)
    anchors = np.vstack([anchor_util.scale_enum(ratio_anchors[i,:],scales)
                            for i in range(ratio_anchors.shape[0])])
    

    stride = 2 ** pool_num
    height = int(np.ceil(image_size / stride))
    width = int(np.ceil(image_size / stride))
    
    ca = base_size / 2
    cp = stride / 2
    cp = np.minimum(image_size/2,cp)
    
    base_center = cp
    
    stride = (image_size * 1.0) / (height + 1.0)
    cp = stride 
    #ssd adds a default box for the aspect ratio of 1
    if k != -1:
        extra_anchor = np.array([[1.,1.,base_size,base_size]]) - 1
        increment = image_size * anchor_util.ssd_scale(k+1,5) - base_size 
        increment /= 2
        extra_anchor[0,0:2] -= increment
        extra_anchor[0,2:4] += increment
        anchors = np.concatenate((anchors,extra_anchor),axis = 0)

    shift_matrix = anchor_util.generate_shift_matrix(height,width,stride,num_per_cell)
    
    anchors = np.expand_dims(anchors,axis = 0)
    anchors = np.repeat(anchors,height * width,axis = 0)
    anchors += shift_matrix
    
    anchors = np.reshape(anchors,(-1,4))
    
    if offset:

        anchors += (cp - ca)

    return anchors

def get_anchor(model,
            base_net,
            image_size,
            base_size = 0):
    """
        produce anchors for a model

        Args:
            model: a string indicating what detection model is used
            base_net: a string indicating what base net is used
            image_size: an integer indicating image size and width
            base_size: an integer indicating basic size if anchor
    """
    
    if model == 'faster_rcnn':
        if base_size == 0:
            base_size = 16
        return produce_anchor(pool_num = 4,
                            image_size = image_size, 
                            num_per_cell = 9,
                            base_size = base_size,
                            ratios = [0.5,1,2],
                            scales = 2**np.arange(3,6))
    elif model == 'ssd':
        #follow the settings in the paper, conv4_3,conv7,conv8_2,conv9_2,conv10_2,conv11_2 are used to predict. 
        #specifically conv4_3, conv 10_2 and conv11_2 has 4 boxes per cell, omitting aspect retio 1/3 and 3
        #conv4_3's  default box with scale 0.1
        
        ratios = [1,2,3,0.5,0.333]
        special_ratios = [1,2,0.5]
        scales = np.ones((1))

        anchors_in_conv4_3 = produce_anchor(pool_num = 3,
                                        image_size = image_size,
                                        num_per_cell = 4,
                                        base_size = math.ceil(0.1*image_size),
                                        ratios = special_ratios,
                                        scales = scales,
                                        k = 0,
                                        offset = True)
        anchors_in_conv7 = produce_anchor(pool_num = 4,
                                        image_size = image_size,
                                        num_per_cell = 6,
                                        base_size = math.ceil(image_size*anchor_util.ssd_scale(1,5)),
                                        ratios = ratios,
                                        scales = scales,
                                        k = 1,
                                        offset = True)

        anchors_in_conv8_2 = produce_anchor(pool_num = 5,
                                            image_size = image_size,
                                            num_per_cell = 6,
                                            base_size = math.ceil(image_size*anchor_util.ssd_scale(2,5)),
                                            ratios = ratios,
                                            scales = scales,
                                            k = 2,
                                            offset = True)

        anchors_in_conv9_2 = produce_anchor(pool_num = 6,
                                            image_size = image_size,
                                            num_per_cell = 6,
                                            base_size = math.ceil(image_size*anchor_util.ssd_scale(3,5)),
                                            ratios = ratios,
                                            scales = scales,
                                            k = 3,
                                            offset = True)

        anchors_in_conv10_2 = produce_anchor(pool_num = 7,
                                            image_size = image_size,
                                            num_per_cell = 4,
                                            base_size = math.ceil(image_size*anchor_util.ssd_scale(4,5)),
                                            ratios = special_ratios,
                                            scales = scales,
                                            k = 4,
                                            offset = True)

        anchors_in_conv11_2 = produce_anchor(pool_num = 9,
                                            image_size = image_size,
                                            num_per_cell = 4,
                                            base_size = math.ceil(image_size*anchor_util.ssd_scale(5,5)),
                                            ratios = special_ratios,
                                            scales = scales,
                                            k = 5,
                                            offset = True)
                                                                            
        anchors = np.concatenate((anchors_in_conv4_3,
                                anchors_in_conv7,
                                anchors_in_conv8_2,
                                anchors_in_conv9_2,
                                anchors_in_conv10_2,
                                anchors_in_conv11_2),
                                axis = 0)

        return anchors

    else:
        raise ValueError('Unknown model.')

