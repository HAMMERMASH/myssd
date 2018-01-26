import tensorflow as tf
import sub_sampler
import input_reader
import losses
import anchor_generator
import feature_extractor
import ssd
import numpy as np
from datetime import datetime
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.app.flags
flags.DEFINE_integer('batch_size',8,'batch_size for input_queue')
flags.DEFINE_integer('dataset_size',11540,'dataset size')
flags.DEFINE_integer('image_size',300,'image height and width')
flags.DEFINE_string('record_path','../../hdd/tfrecord/pascal_300_0.record','record path')
flags.DEFINE_integer('neg_pos_ratio',3,'ratio of neg and pos after hard negative mining')
flags.DEFINE_integer('base_size',16,'base size for anchors')
flags.DEFINE_string('model','ssd','detection model')
flags.DEFINE_string('base_net','vgg_16','base net for rpn')
flags.DEFINE_float('lamb',1,'lambda in loss')
flags.DEFINE_string('checkpoint_path','./','path for trained base net parameter')
flags.DEFINE_string('training_checkpoint_path','./','path for trained parameter')
flags.DEFINE_float('learning_rate',0.001,'initial learning rate')
flags.DEFINE_integer('num_anchor',8732,'total num anchors')
flags.DEFINE_integer('save_step',10000,'steps for saving parameters')
flags.DEFINE_integer('log_step',10,'steps for log training process')
flags.DEFINE_integer('decay_steps',23080,'steps for decaying learning rate')
FLAGS = flags.FLAGS


def encode(boxes,anchors):
    """
        parameterize boxes

        Args:
            boxes: 3-D tensors [batch_size,num_anchors,4], 
                coordinates of gt boxes of each anchor
            anchors: a 2-D tensor [num_anchors,4], 
                four coordinates of default anchors with [:,0] to [:,3]: xmin,ymin,xmax,ymax
        
        Returns:
            encoded_box: a 3-D tensor [batch_size,num_anchors,4],encoded boxes
    """
    anchors = tf.expand_dims(anchors,axis = 0)
    anchors = tf.tile(anchors,[FLAGS.batch_size,1,1])
    wa = anchors[:,:,2] - anchors[:,:,0] + 1
    ha = anchors[:,:,3] - anchors[:,:,1] + 1
    xa = (anchors[:,:,2] + anchors[:,:,0]) / 2.0
    ya = (anchors[:,:,3] + anchors[:,:,1]) / 2.0

    cx = (boxes[:,:,0] + boxes[:,:,2]) / 2.0
    cy = (boxes[:,:,1] + boxes[:,:,3]) / 2.0
    width = boxes[:,:,2] - boxes[:,:,0] + 1
    height = boxes[:,:,3] - boxes[:,:,1] + 1
    
    x = (cx - xa) / wa
    y = (cy - ya) / ha
    
    w = tf.log(width/wa)
    h = tf.log(height/ha)

    return tf.stack([x,y,w,h],axis = 2)
        

def batch_process(objectness,targets,classes,gt_box,anchors):
    """
        Read from decoded input, prepare targets and weights for loss_op
        x/y coordintes: horizontal/vertical direction

        Args:
            objectness: a 2-D tensor [batch_size,num_anchors], objectness label
            targets: a 2-D tensor [batch_size,num_anchors], 
                indicating which gt boxes the anchors is assigned to
            classes: a 2-D tensor [batch_size,num_gt_boxes,num_classes], 
                indicating classes of gt boxes, num_classes == 20 for pascal voc
            gt_box: a 3-D tensor [batch_size,num_gt_boxes,4], 
                xmin,ymin,xmax,ymax coordinates of gt boxes

        Return:
            cls_targets: a 3-D tensor [batch_size,num_anchors,num classes + 1], 
                class targets with back ground labeled 20 for pascal voc
            target_boxes: a 3-D tensor [batch_size,num_anchors,4], 
                anchor-wise corresponding coordinates targets
    """

    batch_size = FLAGS.batch_size
    num_anchor = FLAGS.num_anchor 
    
    #process the targets so that it can index classes and gt_box in tf.gather_nd
    batch_inds = tf.cast(tf.range(batch_size),tf.int32)
    batch_inds = tf.expand_dims(batch_inds,axis = 1)
    batch_inds = tf.tile(batch_inds,[1,num_anchor])
    targets = tf.stack([batch_inds,targets],axis = 2)

    ones = tf.ones([batch_size,num_anchor],tf.int32)
    zeros = tf.zeros([batch_size,num_anchor],tf.int32)
    background = ones * 20#background id
    class_only = tf.gather_nd(classes,targets)
    cls_target = tf.where(tf.equal(objectness,1),class_only,background)
    
    target_boxes = tf.gather_nd(gt_box,targets)
    target_boxes = encode(target_boxes,anchors)
    
    pos_weights = tf.where(tf.equal(objectness,1),ones,zeros)
    neg_weights = tf.where(tf.equal(objectness,0),ones,zeros)
    
    return cls_target,objectness,target_boxes,tf.cast(pos_weights,tf.float32),tf.cast(neg_weights,tf.float32)



def loss_op(cls_logits,
        box_logits,
        cls_targets,
        target_boxes,
        pos_weights,
        neg_weights):
    """
        rpn loss
        
        Args:
            cls_logits: a 4-D tensor [batch_size,height,width,2*k], 
                        indicating 2K objectness score
            box_logits: a 4-D tensor [batch_size,height,width,4*k], 
                        indicating 4K predicted bounding box coordinates
            cls_targets: a 2-D tensor [batch_size,num_anchor], 
                        indicating which gt box the anchor is assigned to
            target_boxes: a 3-D tensor [batch_size,num_anchor,4],
                        indicating anchor-wise gt box regression targets
            pos_weights: a 2-D tensor [batch_size,num_anchor],
                        indicating positive matched anchors
            neg_weights: a 2-D tensor [batch_size,num_anchor],
                        indicating negative anchors
        Returns:
            loss: a scalar loss value
    """
    
    num_pos = tf.reduce_sum(pos_weights,axis = 1)
    weights = pos_weights + neg_weights
    cls_loss = losses.objectness_loss(logits = cls_logits,
                                    labels = cls_targets,
                                    weights = weights)
    
    zeros = tf.zeros([FLAGS.batch_size,FLAGS.num_anchor])
    neg_sample_loss = tf.where(tf.equal(neg_weights,1),cls_loss,zeros)
    
    num_neg = tf.reduce_min(num_pos) * FLAGS.neg_pos_ratio
    num_neg = tf.to_int32(num_neg)
    neg_loss,_ = tf.nn.top_k(neg_sample_loss,num_neg)
    pos_loss = cls_loss * pos_weights
    sampled_loss = tf.reduce_sum(pos_loss,axis = 1) + tf.reduce_sum(neg_loss,axis = 1)
    sampled_loss = sampled_loss / num_pos

    box_loss = losses.bbox_loss(x = target_boxes[:,:,0],
                                x_pred = box_logits[:,:,0],
                                y = target_boxes[:,:,1],
                                y_pred = box_logits[:,:,1],
                                w = target_boxes[:,:,2],
                                w_pred = box_logits[:,:,2],
                                h = target_boxes[:,:,3],
                                h_pred = box_logits[:,:,3],
                                weights = pos_weights)
    
    box_loss = tf.reduce_sum(box_loss,axis = 1) / num_pos
    ssd_loss = tf.reduce_sum(sampled_loss + FLAGS.lamb * box_loss)
    return ssd_loss,sampled_loss,box_loss


def main(_):
    with tf.Graph().as_default():

        global_step = tf.train.get_or_create_global_step()
        
        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step = global_step,decay_steps = FLAGS.decay_steps,decay_rate = 0.95)
        
        opt = tf.train.AdamOptimizer(lr,epsilon=1)
        
        
        anchors_array = anchor_generator.get_anchor(
                    model = FLAGS.model,
                    base_net = FLAGS.base_net,
                    image_size = FLAGS.image_size,
                    base_size = FLAGS.base_size)

        anchors = tf.constant(anchors_array,dtype = tf.float32)

        image,label,target,gt_box,classes = input_reader.input(
                                             record_path = FLAGS.record_path,
                                             dataset_size = FLAGS.dataset_size,
                                             batch_size = FLAGS.batch_size,
                                             image_size = FLAGS.image_size)

        
        cls_targets,objectness,target_boxes,pos_weights,neg_weights = batch_process(
            label,target,classes,gt_box,anchors)
    
    
        
        cls_logits,box_logits = ssd.inference(image,norm = True)
    
        ssd_loss, cls_loss, box_loss = loss_op(
                            cls_logits = cls_logits,
                            box_logits = box_logits,
                            cls_targets = cls_targets,
                            target_boxes = target_boxes,
                            pos_weights = pos_weights,
                            neg_weights = neg_weights)
        
                    
        
        train_var = [var for var in tf.global_variables() if 'ssd' in var.name]
        
        for var in train_var:
            print(var.name)

        ssd_train = opt.minimize(ssd_loss,global_step = global_step,var_list = train_var)
        
    
        loss_display = tf.reduce_sum(ssd_loss)
        cls_loss_display = tf.reduce_sum(cls_loss)
        box_loss_display = tf.reduce_sum(box_loss)
        #var_list = [var for var in tf.global_variables() if not 'global' in var.name  and not 'Momentum' in var.name  and not 'Adam' in var.name and not 'beta' in var.name]  
        #restore_saver = tf.train.Saver(var_list)

        var_list = [var for var in tf.global_variables() if not 'global' in var.name and not 'ssd' in var.name and not 'Momentum' in var.name and not 'Adam' in var.name and not 'beta' in var.name]
        base_net_saver = tf.train.Saver(var_list)
        
        saver = tf.train.Saver(max_to_keep = 1000)
        
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

        with tf.Session() as sess:
            start_time = datetime.now()
            sess.run(init_op)
            
            #ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path,'_checkpoint')
            #restore_saver.restore(sess,ckpt.model_checkpoint_path)

            base_net_saver.restore(sess,'vgg_16.ckpt')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)


            iter_start_time = datetime.now()
            ave_loss = 0
            ave_cls_loss = 0
            ave_box_loss = 0

            display_step = 500
            for i in range(500001):

                _,test0,test1,test2,test3 = sess.run([ssd_train,loss_display,cls_loss_display,box_loss_display,lr])
                ave_loss += test0
                ave_cls_loss += test1
                ave_box_loss += test2
                if i%display_step == 0:
                    iter_end_time = datetime.now()
                    print("iteration {}: loss: {},class loss: {},regression loss: {},lr: {}, duration: {} ".format(i+1,ave_loss/display_step,ave_cls_loss/display_step,ave_box_loss/display_step,test3,iter_end_time - iter_start_time))
                    ave_loss = 0
                    ave_cls_loss = 0
                    ave_box_loss = 0
                    iter_start_time = datetime.now()
                if i%FLAGS.save_step == 0:
                    saver.save(sess,'./ssd.ckpt',global_step = global_step)
                if i%FLAGS.log_step == 0:
                    log_file = open('./log_file','a')
                    log_file.write('{},{},{}\n'.format(i,test1,test2))
                    log_file.close()
            
                
            end_time = datetime.now()
            print('total training time: {}'.format(end_time-start_time))
            coord.request_stop()
            coord.join(threads)
            


if __name__ == '__main__':
    tf.app.run()
