"""Label map utility functions."""

import logging

import tensorflow as tf

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "diningtable", "dog", "horse", "motorbike", "person","pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def get_label_map_dict():
  label_map_dict = {}
  i = 0
  for item in CLASSES:
    label_map_dict[item] = i
    i+=1
  return label_map_dict
