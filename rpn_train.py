import tensorflow as tf
import sub_sampler
import input_reader
import losses
import anchor_generator
import feature_extractor
import rpn
import numpy as np
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

flags = tf.app.flags
flags.DEFINE_integer('batch_size',1,'batch_size for input_queue')
flags.DEFINE_integer('dataset_size',11540,'dataset size')
flags.DEFINE_integer('image_size',600,'image height and width')
flags.DEFINE_string('record_path','./pascal.record','record path')
flags.DEFINE_integer('num_samples',256,'num of samples per image')
flags.DEFINE_integer('base_size',16,'base size for anchors')
flags.DEFINE_integer('k',9,'number of prediction per cell')
flags.DEFINE_string('model','faster_rcnn','detection model')
flags.DEFINE_string('base_net','vgg16','base net for rpn')
flags.DEFINE_float('lamb',10,'lambda in loss')
flags.DEFINE_string('checkpoint_path','../../pretrained_models/','path for trained base net parameter')
flags.DEFINE_string('training_checkpoint_path','./','path for trained parameter')
flags.DEFINE_float('learning_rate',0.001,'initial learning rate')
flags.DEFINE_integer('save_step',1000,'steps for saving parameters')
flags.DEFINE_integer('log_step',10,'steps for log training process')
flags.DEFINE_integer('decay_steps',11540,'steps for decaying learning rate')
FLAGS = flags.FLAGS


def parameterize(xmin,xmin_pred,
                ymin,ymin_pred,
                xmax,xmax_pred,
                ymax,ymax_pred,
                anchors):
    """
        parameterize coordinates of predictions and gt boxes

        Args:
            xmin,ymin,xmax,ymax: 1-D tensors [num_anchors], coordinates of gt boxes of each anchor
            xmin_pred,ymin_pred,xmax_pred,ymax_pred: 1-D tensors [num_anchors], coordinaters of predictions
            anchors: a 2-D tensor [num_anchors,4], four coordinates of default anchors with [:,0] to [:,3]: xmin,ymin,xmax,ymax
            targets: a 1-D tensor [num_anchors], indicating gt box targets for each anchor
        
        Returns:
            x,y: 1-D tensors [num_gt_boxes], parameterized gt box centers
            x_pred,y_pred: 1-D tensors [num_anchors], parameteraized prediction centers
            w,h: 1-D tensors [num_gt_boxes], parameterized gt box width and height
            w_pred,h_pred: 1-D tensors [num_anchors], parameterized prediction width and height
    """
    wa = anchors[:,2] - anchors[:,0]
    ha = anchors[:,3] - anchors[:,1]
    xa = (anchors[:,2] - anchors[:,0]) / 2.0
    ya = (anchors[:,3] - anchors[:,1]) / 2.0

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx_pred = (xmin_pred + xmax_pred) / 2.0
    cy_pred = (ymin_pred + ymax_pred) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    width_pred = xmax_pred - xmin_pred
    height_pred = ymax_pred - ymin_pred

    x = (cx - xa) / wa
    y = (cy - ya) / ha
    x_pred = (cx_pred - xa) / wa
    y_pred = (cy_pred - ya) / ha

    zeros = tf.zeros([tf.size(x)])
    w = tf.log(tf.abs(width/wa) + 0.1) #initially w = tf.log(width/wa) will course nan loss. for width initially random
    h = tf.log(tf.abs(height/ha) + 0.1)
    w_pred = tf.log(tf.abs(width_pred/wa) + 0.1)
    w_pred = tf.where(tf.less(w_pred,-10000),zeros,w_pred)
    h_pred = tf.log(tf.abs(height_pred/ha) + 0.1)
    h_pred = tf.where(tf.less(h_pred,-10000),zeros,h_pred)

    return x,y,x_pred,y_pred,w,h,w_pred,h_pred
        

def batch_process(labels,targets,xmin,ymin,xmax,ymax,num_samples):
    """
        Read from decoded input, prepare sub-sampled object target and weights for loss_op
        x/y coordintes: horizontal/vertical direction

        only support batch size = 1
        Args:
            labels: a 2-D tensor [batch_size,num_anchors], objectness label
            targets: a 2-D tensor [batch_size,num_anchors], indicating which gt boxes the anchors is assigned to
            xmin: a 2-D tensor [batch_size,num_gt_boxes], min x coordinates of gt boxes
            ymin: a 2-D tensor [batch_size,num_gt_boxes], min y coordinates of gt boxes
            xmax: a 2-D tensor [batch_size,num_gt_boxes], max x coordinates of gt boxes
            ymax: a 2-D tensor [batch_size,num_gt_boxes], max y coordinates of gt boxes
            num_samples: a scalar, an even number for 1:1 positive samples and negative samples

        Return:
            pos_inds: a 1-D tensor [num_anchors], targets for objectness softmax cross entropy calculation, 1 for positive, 0 for negative
            b_targets: a 1-D tensor [num_anchors], targets[0]
            b_xmin: a 1-D tensor [num_anchors], anchor-wise corresponding min x coordinates targets
            b_ymin: a 1-D tensor [num_anchors], anchor-wise corresponding min y coordinates targets
            b_xmax: a 1-D tensor [num_anchors], anchor-wise corresponding max x coordinates targets
            b_ymax: a 1-D tensor [num_anchors], anchor-wise corresponding max y coordinates targets
            weights: a 1-D tensor [num_anchors], 1 for samples, 0 for the rest
            num_pos: a scalar indicating number of positive samples left after sampling
            num_neg: a scalar indicating number of negative samples left after sampling
    """
    positive_samples = tf.cast(num_samples / 2,dtype = tf.int32)

    pos_indices_in_batch = tf.equal(labels[0],1)
    positive_in_batch = tf.size(tf.where(pos_indices_in_batch))
    
    num_pos = tf.minimum(positive_in_batch,positive_samples)

    pos_inds = sub_sampler.sub_sample(indicator = pos_indices_in_batch,
                                    num_samples = num_pos,
                                    indices_value = 1,
                                    default_value = 0)

    weights = pos_inds
    neg_indices_in_batch = tf.equal(labels,0)
    negative_in_batch = tf.size(tf.where(neg_indices_in_batch))

    num_neg = tf.minimum(negative_in_batch,num_samples - num_pos)
    neg_inds = sub_sampler.sub_sample(indicator = neg_indices_in_batch,
                                    num_samples = num_neg,
                                    indices_value = 1,
                                    default_value = 0)

    weights += neg_inds
    num_sampled_elements = num_pos + num_neg
    
    b_targets = targets[0]
    b_targets = tf.cast(b_targets,tf.int32)

    b_xmin = tf.gather(xmin[0],b_targets)
    b_ymin = tf.gather(ymin[0],b_targets)
    b_xmax = tf.gather(xmax[0],b_targets)
    b_ymax = tf.gather(ymax[0],b_targets)

    pos_inds = tf.cast(pos_inds,tf.int64)
    
    return pos_inds,b_xmin,b_ymin,b_xmax,b_ymax,weights,num_pos,num_neg

def loss_op(cls_logits,
        box_logits,
        objectness,
        xmin,
        ymin,
        xmax,
        ymax,
        weights,
        num_pos,
        num_neg,
        k,
        lamb,
        anchors):
    """
        rpn loss
        
        Args:
            cls_logits: a 4-D tensor [batch_size,height,width,2*k], indicating 2K objectness score
            box_logits: a 4-D tensor [batch_size,height,width,4*k], indicating 4K predicted bounding box coordinates
            objectness: a 1-D tensor [num_anchors], indicating objectness targets for each anchor
            targets: a 1-D tensor [num_anchors], indicating which gt box the anchor is assigned to
            xmin: a 1-D tensor [num_gt_boxes], indicating min x coordinates of a gt box
            ymin: a 1-D tensor [num_gt_boxes], indicating min y coordinates of a gt box
            xmax: a 1-D tensor [num_gt_boxes], indicating max x coordinates of a gt box
            ymax: a 1-D tensor [num_gt_boxes], indicating max y coordinates of a gt box
            weights: a 1-D tensor [num_anchors], deciding which anchors to compute loss
            k: a scalar indicating number of anchors per cell
            lamb: lambda to balance loss for object and box
            anchors: a 2-D tensor [num_anchors,4], default anchor coordinates

        Returns:
            loss: a scalar loss value
    """

    cls_score = tf.reshape(cls_logits,[-1,2])
    box_pred = tf.reshape(box_logits,[-1,4])
    xmin_pred = box_pred[:,0]
    ymin_pred = box_pred[:,1]
    xmax_pred = box_pred[:,2]
    ymax_pred = box_pred[:,3]
    
    cls_loss = losses.objectness_loss(logits = cls_score,
                                    labels = objectness,
                                    weights = weights,
                                    num_samples = num_pos + num_neg)
    
    x,y,x_pred,y_pred,w,h,w_pred,h_pred = parameterize(xmin,xmin_pred,
                                                    ymin,ymin_pred,
                                                    xmax,xmax_pred,
                                                    ymax,ymax_pred,
                                                    anchors)

    pos_weights = tf.cast(objectness,tf.float32)
    normalizer = tf.size(objectness)
    box_loss = losses.bbox_loss(x = x,
                                x_pred = x_pred,
                                y = y,
                                y_pred = y_pred,
                                w = w,
                                w_pred = w_pred,
                                h = h,
                                h_pred = h_pred,
                                weights = pos_weights,
                                normalizer = normalizer)

    return cls_loss + lamb * box_loss, cls_loss, box_loss


def main(_):
    with tf.Graph().as_default():

        global_step = tf.train.get_or_create_global_step()
        
        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step = global_step,decay_steps = FLAGS.decay_steps,decay_rate = 0.9995)
        
        opt = tf.train.MomentumOptimizer(lr,0.9)
        
        with tf.device('/cpu:0'):
            anchors_array = anchor_generator.get_anchor(model = FLAGS.model,
                                                    base_net = FLAGS.base_net,
                                                    image_size = FLAGS.image_size,
                                                    base_size = FLAGS.base_size)

            anchors = tf.constant(anchors_array,dtype = tf.float32)

            image,label,target,xmin,ymin,xmax,ymax,classes = input_reader.input(record_path = FLAGS.record_path,
                                                                            dataset_size = FLAGS.dataset_size,
                                                                            batch_size = FLAGS.batch_size,
                                                                            image_size = FLAGS.image_size)

            objectness_targets,xmins,ymins,xmaxes,ymaxes,weights,num_pos,num_neg = batch_process(label,target,xmin,ymin,xmax,ymax,FLAGS.num_samples)
        
        with tf.device('/gpu:0'):
            feature_map = feature_extractor.extract(base_net = FLAGS.base_net,image = image)

            obj_logits,box_logits = rpn.inference(feature_map,k = FLAGS.k)
        
        rpn_loss, cls_loss, box_loss = loss_op(cls_logits = obj_logits,
                                box_logits = box_logits,
                                objectness = objectness_targets,
                                xmin = xmins,
                                ymin = ymins,
                                xmax = xmaxes,
                                ymax = ymaxes,
                                weights = weights,
                                num_pos = num_pos,
                                num_neg = num_neg,
                                k = FLAGS.k,
                                lamb = FLAGS.lamb,
                                anchors = anchors)
        
        train_var = [var for var in tf.global_variables() if 'Momentum' in var.name or 'rpn' in var.name or 'conv3' in var.name or 'conv4' in var.name or 'conv5' in var.name]

        rpn_train = opt.minimize(rpn_loss,global_step = global_step,var_list = train_var)

        var_list = [var for var in tf.global_variables() if not 'Momentum' in var.name and not 'rpn' in var.name]
        restore_saver = tf.train.Saver(var_list)
        
        saver = tf.train.Saver(max_to_keep = 1000)
        
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path,'checkpoint')
            restore_saver.restore(sess,ckpt.model_checkpoint_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)

            start_time = datetime.now()

            iter_start_time = datetime.now()
            for i in range(10):
                _,test0,test1,test2 = sess.run([rpn_train,rpn_loss,cls_loss,box_loss])
                iter_end_time = datetime.now()
                print("iteration {}: loss: {},class loss: {},regression loss: {}. duration: {} ".format(i+1,test0,test1,test2,iter_end_time - iter_start_time))
                if i%FLAGS.save_step == 0:
                    saver.save(sess,'./rpn.ckpt',global_step = global_step)
                if i%FLAGS.log_step == 0:
                    log_file = open('./log_file','a')
                    log_file.write('{},{},{}\n'.format(i,test1,test2))
                    log_file.close()

                iter_start_time = datetime.now()

            end_time = datetime.now()
            print('total training time: {}'.format(end_time-start_time))
            coord.request_stop()
            coord.join(threads)
            


if __name__ == '__main__':
    tf.app.run()
