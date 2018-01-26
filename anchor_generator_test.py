import anchor_generator

anchors = anchor_generator.get_anchor(model = 'faster_rcnn',base_net = 'vgg16',image_size = 300)
print(anchors)
