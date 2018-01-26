import numpy as np
import anchor_target

gt_boxes = np.array([[100,100,300,300],[50,50,100,100]],dtype= np.float32)
labels,targets = anchor_target.get_target(model = 'faster_rcnn',base_net = 'vgg16',gt_boxes = gt_boxes,image_size = 300)
print(labels.shape)
print(targets.shape)
print(np.count_nonzero(labels == 0))
