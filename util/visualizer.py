import numpy as np
from util import label_map_util
from PIL import Image,ImageDraw

def draw_bounding_box(image,box_list,color,name,class_id):
    
    x1 = box_list[:,0]
    y1 = box_list[:,1]
    x2 = box_list[:,2]
    y2 = box_list[:,3]
    num_box = len(x1)
    height = image.shape[0]
    width = image.shape[1]
    p_image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(p_image)

    for i in range(num_box):
        draw.rectangle((x1[i],y1[i],x2[i],y2[i]))
        class_name = label_map_util.get_label_name(class_id[i])
        draw.text((x1[i],y1[i]),'{}'.format(class_name))

    p_image.save('./{}.jpg'.format(name),'JPEG')
