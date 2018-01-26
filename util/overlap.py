import tensorflow as tf


def get_overlap(box,xmin,ymin,xmax,ymax):
    """
        get overlaps
    """

    area0 = (xmax - xmin + 1) * (ymax - ymin + 1)
    area1 = (box[:,2] - box[:,0] + 1) * (box[:,3] - box[:,1] + 1)

    min_x = tf.maximum(tf.expand_dims(box[:,0],-1),tf.transpose(tf.expand_dims(xmin,-1)))
    max_x = tf.minimum(tf.expand_dims(box[:,2],-1),tf.transpose(tf.expand_dims(xmax,-1)))
    iw = tf.maximum(0.0,max_x - min_x)

    min_y = tf.maximum(tf.expand_dims(box[:,1],-1),tf.transpose(tf.expand_dims(ymin,-1)))
    max_y = tf.minimum(tf.expand_dims(box[:,3],-1),tf.transpose(tf.expand_dims(ymax,-1)))
    ih = tf.maximum(0.0,max_y - min_y)

    intersections = ih* iw
    unions = tf.expand_dims(area1,-1) + area0 - intersections

    iou = intersections / unions

    return iou



