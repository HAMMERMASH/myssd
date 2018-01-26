from __future__ import division
from __future__ import print_function

import tensorflow as tf
import hashlib
import io
import logging
import os
import scipy.misc as misc
from lxml import etree
import numpy as np
from util import dataset_util
from util import label_map_util
import anchor_target
import random

flags = tf.app.flags
flags.DEFINE_string('data_dir','../../hdd/pascalvoc/VOCdevkit/','Root directory to raw PASCAL VOC dataset')
flags.DEFINE_string('set','trainval','Convert training set,validation set or merged set')
flags.DEFINE_string('annotations_dir','Annotations','(Relative) path to annotations directory')
flags.DEFINE_string('year','VOC2012','Desired challenge year')
flags.DEFINE_string('output_path','../../hdd/tfrecord/pascal_300_0.record','Path to output TFRecord')
flags.DEFINE_boolean('ignore_difficult_instances',False,'Whether to ignore difficult instances')
flags.DEFINE_integer('image_size',300,'shortest size of a image')
FLAGS = flags.FLAGS

SETS = ['train','val','trainval','test']
YEARS = ['VOC2007','VOC2012','merged']




def dict_to_tf_example(data,
                    dataset_directory,
                    label_map_dict,
                    ignore_difficult_instances=False,
                    image_subdirectory='JPEGImages',
                    augment = 0):
    """
        Produce example of tfrecord. Match anchors for each input image and produce target in the same time.
        Args:
            data: a tree like dict parsed from xmls of Pascal dataset.
            dataset_directory: a string, folder containing the xml file.
            label_map_dict: a dict [class_string, class_int], mapping class name to a integer.
        Returns:
            example: return the tf.train.Example.
    """
    img_path = os.path.join(data['folder'],image_subdirectory,data['filename'])
    full_path = os.path.join(dataset_directory,img_path)
    image = misc.imread(full_path)
    image = misc.imresize(image,[FLAGS.image_size,FLAGS.image_size,3])

    width = int(data['size']['width'])
    height = int(data['size']['height'])
    
    filename = data['filename'].encode('utf8')

    ymin = []
    xmin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue
        
        difficult_obj.append(int(difficult))
        
        xmin.append(np.round(float(obj['bndbox']['xmin']) / width * FLAGS.image_size))
        ymin.append(np.round(float(obj['bndbox']['ymin']) / height * FLAGS.image_size))
        xmax.append(np.round(float(obj['bndbox']['xmax']) / width * FLAGS.image_size))
        ymax.append(np.round(float(obj['bndbox']['ymax']) / height * FLAGS.image_size))
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))
    


    #rotate
    if augment == 1:
        
        r = image[:,:,0]
        g = image[:,:,1]
        b = image[:,:,2]
        r = np.transpose(r)
        g = np.transpose(g)
        b = np.transpose(b)

        image = np.stack((r,g,b),axis = 2).astype('uint8')

        
        image_raw = image.tostring()
        
        xmin = np.array(xmin)
        ymin = np.array(ymin)
        xmax = np.array(xmax)
        ymax = np.array(ymax)

        _xmin = ymin
        _ymin = FLAGS.image_size - xmax
        _xmax = ymax
        _ymax = FLAGS.image_size - xmin
        
        return make_tf_example(filename,height,width,image_raw,_xmin,_ymin,_xmax,_ymax,classes_text,classes,difficult_obj,truncated,poses)

    #crop
    if augment == 2:
        

        ind = random.randint(0,len(xmin)-1)
        xmin = np.array(xmin)
        ymin = np.array(ymin)
        xmax = np.array(xmax)
        ymax = np.array(ymax)

        cx = (xmin[ind] + xmax[ind]) / 2.0
        cy = (ymin[ind] + ymax[ind]) / 2.0
        cw = xmax[ind] - xmin[ind] + 1
        ch = ymax[ind] - ymin[ind] + 1

        crop_size = np.maximum(cw,ch) + random.randint(0,FLAGS.image_size - np.maximum(cw,ch))
        assert crop_size <= 300,'crop size too large'

        cx_min = np.round(cx - crop_size / 2)
        cy_min = np.round(cy - crop_size / 2)
        cx_max = np.round(cx + crop_size / 2)
        cy_max = np.round(cy + crop_size / 2)

        if cx_min < 0:
            cx_min = 0
        if cy_min < 0:
            cy_min = 0
        if cx_max > FLAGS.image_size - 1:
            cx_max = FLAGS.image_size - 1
        if cy_max > FLAGS.image_size - 1:
            cy_max = FLAGS.image_size - 1

        _h = cy_max - cy_min + 1
        _w = cx_max - cx_min + 1
        
        _xmin = xmin
        _ymin = ymin
        _xmax = xmax
        _ymax = ymax
        
        

        keep_inds = []
        for i in range(len(xmin)):
            
            t_xmin = _xmin[i]
            t_ymin = _ymin[i]
            t_xmax = _xmax[i]
            t_ymax = _ymax[i]
            if _xmin[i] < cx_min:
                t_xmin = cx_min
            if _ymin[i] < cy_min:
                t_ymin = cy_min
            if _xmax[i] > cx_max:
                t_xmax = cx_max
            if _ymax[i] > cy_max:
                t_ymax = cy_max
            
            t_w = t_xmax - t_xmin + 1
            t_h = t_ymax - t_ymin + 1

            t_a = t_w * t_h
            o_a = (_xmax[i] - _xmin[i] + 1) * (_ymax[i] - _ymin[i] + 1)

            if t_w > 0 and t_h > 0 and o_a * 1.0 / t_a < 2:

                _xmin[i] = t_xmin
                _xmax[i] = t_xmax
                _ymin[i] = t_ymin
                _ymax[i] = t_ymax
                keep_inds.append(i)

        assert len(keep_inds) != 0,'no box in image'

        _xmin = np.round((_xmin[keep_inds] - cx_min) / _w * FLAGS.image_size)
        _ymin = np.round((_ymin[keep_inds] - cy_min) / _h * FLAGS.image_size)
        _xmax = np.round((_xmax[keep_inds] - cx_min) / _w * FLAGS.image_size)
        _ymax = np.round((_ymax[keep_inds] - cy_min) / _h * FLAGS.image_size)
        
        _classes = np.array(classes)[keep_inds]

        image = image[int(cy_min):int(cy_max),int(cx_min):int(cx_max)]
        image = misc.imresize(image,[FLAGS.image_size,FLAGS.image_size,3])
        
        image_raw = image.tostring()

        return make_tf_example(filename,height,width,image_raw,_xmin,_ymin,_xmax,_ymax,classes_text,_classes,difficult_obj,truncated,poses)

        if augment == 3:
            image = image[:,::-1]
            _xmin = FLAGS.image_size - xmax
            _ymin = ymin
            _xmax = FLAGS.image_size - xmin
            _ymax = ymax
            image_raw = image.tostring()
            return make_tf_example(filename,height,width,image_raw,_xmin,_ymin,_xmax,_ymax,classes_text,classes,difficult_obj,truncated,poses)
        
        if augment == 4:
            channel = random.randint(0,2)
            for c in range(3):
                image[:,:,c] = image[:,:,channel]
            image_raw = image.tostring()
            return make_tf_example(filename,height,width,image_raw,xmin,ymin,xmax,ymax,classes_text,classes,difficult_obj,truncated,poses)
            

        
    image_raw = image.tostring()
    return make_tf_example(filename,height,width,image_raw,xmin,ymin,xmax,ymax,classes_text,classes,difficult_obj,truncated,poses)

def make_tf_example(filename,height,width,image_raw,xmin,ymin,xmax,ymax,classes_text,classes,difficult_obj,truncated,poses):

    boxes = np.c_[xmin,ymin,xmax,ymax]
    labels,targets = anchor_target.get_target(model = 'ssd',base_net = 'vgg_16',image_size = FLAGS.image_size,gt_boxes = boxes)

    example = tf.train.Example(features = tf.train.Features(feature = {
        'image/height':dataset_util.int64_feature(height),
        'image/width':dataset_util.int64_feature(width),
        'image/filename':dataset_util.bytes_feature(filename),
        'image/source_id':dataset_util.bytes_feature(filename),
        'image/image':dataset_util.bytes_feature(image_raw),
        'image/format':dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/labels':dataset_util.float_list_feature(labels),
        'image/targets':dataset_util.float_list_feature(targets),
        'image/object/bbox/xmin':dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':dataset_util.float_list_feature(ymax),
        'image/object/class/text':dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label':dataset_util.int64_list_feature(classes),
        'image/object/difficult':dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated':dataset_util.int64_list_feature(truncated),
        'image/object/view':dataset_util.bytes_list_feature(poses),
    }))
    num_match = np.count_nonzero(labels)
    print('{}, num matched: {}'.format(filename,num_match))
    return example


def main(_):

    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))
    if FLAGS.year not in YEARS:
        raise ValueError('Year must be in : {}'.format(YEARS))

    data_dir = FLAGS.data_dir
    years = ['VOC2007','VOC2012']
    if FLAGS.year != 'merged':
        years = [FLAGS.year]

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict()


    for year in years:
        logging.info('Reading from PASCAL %s dataset.',year)
        examples_path = os.path.join(data_dir,year,'ImageSets','Main','aeroplane_'+FLAGS.set+'.txt')
        annotations_dir = os.path.join(data_dir,year,FLAGS.annotations_dir)
        examples_list = dataset_util.read_examples_list(examples_path)
        for idx,example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d',idx,len(examples_list))
            path = os.path.join(annotations_dir,example + '.xml')
            with tf.gfile.GFile(path,'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data,FLAGS.data_dir,label_map_dict,FLAGS.ignore_difficult_instances)
            writer.write(tf_example.SerializeToString())
            #tf_example = dict_to_tf_example(data,FLAGS.data_dir,label_map_dict,FLAGS.ignore_difficult_instances,augment = 1)
            #writer.write(tf_example.SerializeToString())
            #tf_example = dict_to_tf_example(data,FLAGS.data_dir,label_map_dict,FLAGS.ignore_difficult_instances,augment = 2)
            #writer.write(tf_example.SerializeToString())
            #tf_example = dict_to_tf_example(data,FLAGS.data_dir,label_map_dict,FLAGS.ignore_difficult_instances,augment = 3)
            #writer.write(tf_example.SerializeToString())
            #tf_example = dict_to_tf_example(data,FLAGS.data_dir,label_map_dict,FLAGS.ignore_difficult_instances,augment = 4)
            #writer.write(tf_example.SerializeToString())



    writer.close()

if __name__ == '__main__':
    tf.app.run()
