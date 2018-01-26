import tensorflow as tf

def objectness_loss(logits,
                labels,
                weights):
    """
        class score for rpn

        Args:
            logits: a 2-D tensor of [num_anchors,2]
            labels: a 1-D tensor of [num_anchros]
            weights: a 1-D tensor of [num_anchors], loss computed for weights = 1

        Returns:
            loss: a scalar loss
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,
                                                            logits = logits)
    weighted_cross_entropy = cross_entropy * weights
    
    return weighted_cross_entropy

def softmax_loss(logits,
                labels,
                weights,
                num_samples):

    return objectness_loss(logits,lables,weights,num_samples)

def smoothl1loss(x,weights):
    """
        smmoth l1 loss from fast rcnn

        Args:
            x: a 1-D tensor
            weights: a 1-D tensor with same dimension as x
        Returns:
            loss: a scalar loss
    """
    abs_diff = tf.abs(x)
    abs_diff_lt_1 = tf.less(abs_diff,1)
    anchorwise_smooth_l1norm = tf.where(abs_diff_lt_1,0.5*tf.square(abs_diff),abs_diff - 0.5)
    weighted_loss = anchorwise_smooth_l1norm * weights

    return weighted_loss

def bbox_loss(x,
            x_pred,
            y,
            y_pred,
            w,
            w_pred,
            h,
            h_pred,
            weights,
            scale = [10.,10.,5.,5.]):
    """
        compute box prediction loss of rpn

        Args:
            x,y: 1-D tensors [num_gt_boxes], parameterized center coordinates of gt boxes
            x_pred,y_pred: 1-D tensors [num_anchors], parameterized center coordinates of predictions
            w,h: 1-D tensors [num_gt_boxes], parameterized width and height of gt boxes
            w_pred,h_pred: 1-D tensors [num_anchors], parameterized width and height of predictions
            weights: 1-D tensor [num_anchors], deciding which anchors to compute loss
            normalizer: a scalar indicating the normalizer, equalling num anchors when None
            scale: default scaling factor
        Returns:
            loss: a scalar loss
    """
    sum_loss = smoothl1loss(scale[0] * x - x_pred,weights) + smoothl1loss(scale[1] * y - y_pred,weights) + smoothl1loss(scale[2] * w - w_pred,weights) + smoothl1loss(scale[3] * h - h_pred,weights)

    return sum_loss
