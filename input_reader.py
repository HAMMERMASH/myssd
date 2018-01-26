"""
    read input from TFRecord of PASCAL VOC datasets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import tensorflow as tf
import io
import scipy.misc as misc
#Basic model parameters as external flags
#FLAGS = None

save_image = 0

def read_and_decode(filename_queue,image_size):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
        'image/image':tf.FixedLenFeature((),tf.string,default_value=''), 
        'image/format':tf.FixedLenFeature((),tf.string,default_value='jpeg'),
        'image/filename':tf.FixedLenFeature((),tf.string,default_value=''),
        'image/source_id':tf.FixedLenFeature((),tf.string,default_value=''),
        'image/height':tf.FixedLenFeature((),tf.int64,1),
        'image/width':tf.FixedLenFeature((),tf.int64,1),
        'image/object/bbox/xmin':tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':tf.VarLenFeature(tf.float32),
        'image/labels':tf.VarLenFeature(tf.float32),
        'image/targets':tf.VarLenFeature(tf.float32),
        'image/object/class/label':tf.VarLenFeature(tf.int64),
        'image/object/difficult':tf.VarLenFeature(tf.int64)
    })
    
    image = tf.decode_raw(features['image/image'],tf.uint8)
    image = tf.reshape(image,[image_size,image_size,3])
    labels = tf.cast(features['image/labels'],tf.float32)
    targets = tf.cast(features['image/targets'],tf.int32)
    xmin = tf.cast(features['image/object/bbox/xmin'],tf.float32)
    ymin = tf.cast(features['image/object/bbox/ymin'],tf.float32)
    xmax = tf.cast(features['image/object/bbox/xmax'],tf.float32)
    ymax = tf.cast(features['image/object/bbox/ymax'],tf.float32)
    classes = tf.cast(features['image/object/class/label'],tf.int32)

    return image,labels,targets,xmin,ymin,xmax,ymax,classes

def input(record_path,dataset_size,batch_size,image_size,num_epochs=1):
    """
        read input data num_epochs times
    """

    if not num_epochs: 
        num_epochs = None
    filename = record_path

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename])
        image,label,target,xmin,ymin,xmax,ymax,classes = read_and_decode(filename_queue,image_size)
        image.set_shape([image_size,image_size,3])
        images,labels,targets,b_xmin,b_ymin,b_xmax,b_ymax,b_classes = tf.train.shuffle_batch([image,label,target,xmin,ymin,xmax,ymax,classes],batch_size = batch_size,num_threads = 8,capacity = dataset_size + 3*batch_size,min_after_dequeue=100)
        dense_labels = tf.sparse_tensor_to_dense(labels)
        dense_targets = tf.sparse_tensor_to_dense(targets)
        dense_xmin = tf.sparse_tensor_to_dense(b_xmin)
        dense_ymin = tf.sparse_tensor_to_dense(b_ymin)
        dense_xmax = tf.sparse_tensor_to_dense(b_xmax)
        dense_ymax = tf.sparse_tensor_to_dense(b_ymax)
        dense_classes = tf.sparse_tensor_to_dense(b_classes)
        images = tf.cast(images,tf.float32)

        dense_box = tf.stack(
            [dense_xmin,dense_ymin,dense_xmax,dense_ymax],axis = 2)
    return images,dense_labels,dense_targets,dense_box,dense_classes
