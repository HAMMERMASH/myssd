import tensorflow as tf
import layer

def inference(image,norm = True,phase_train = True):
    """
        build ssd architecture, based on vgg16
        Args:
            image: a [300,300,3] tensor, the input image.
            phase_train: a boolean indicating training or testing stage.
        Returns:
            cls_logit: a 2-D [-1,20] tensor, the concatenated class predictions
            loc_logit: a 2-D [-1,4] tensor, the concatenated location predictions
    """
    batch_size = image.shape[0]
    r,g,b = tf.split(axis = 3,num_or_size_splits = 3,value = image)
    p_image = tf.concat([r - 123.68,
                        g - 116.78,
                        b - 103.94],axis = 3)
    with tf.variable_scope('vgg_16'):
        with tf.variable_scope('conv1'):
            conv1_1 = layer.conv_layer('conv1_1',p_image,[3,3,3,64])
            conv1_2 = layer.conv_layer('conv1_2',conv1_1,[3,3,64,64])
            pool1 = layer.pool_layer('pool1',conv1_2)
        with tf.variable_scope('conv2'):
            conv2_1 = layer.conv_layer('conv2_1',pool1,[3,3,64,128])
            conv2_2 = layer.conv_layer('conv2_2',conv2_1,[3,3,128,128])
            pool2 = layer.pool_layer('pool2',conv2_2)
        with tf.variable_scope('conv3'):
            conv3_1 = layer.conv_layer('conv3_1',pool2,[3,3,128,256])
            conv3_2 = layer.conv_layer('conv3_2',conv3_1,[3,3,256,256])
            conv3_3 = layer.conv_layer('conv3_3',conv3_2,[3,3,256,256])
            pool3 = layer.pool_layer('pool3',conv3_3)
        with tf.variable_scope('conv4'):
            conv4_1 = layer.conv_layer('conv4_1',pool3,[3,3,256,512])
            conv4_2 = layer.conv_layer('conv4_2',conv4_1,[3,3,512,512])
            conv4_3 = layer.conv_layer('conv4_3',conv4_2,[3,3,512,512])
            pool4 = layer.pool_layer('pool4',conv4_3)
        with tf.variable_scope('conv5'):
            conv5_1 = layer.conv_layer('conv5_1',pool4,[3,3,512,512])
            conv5_2 = layer.conv_layer('conv5_2',conv5_1,[3,3,512,512])
            conv5_3 = layer.conv_layer('conv5_3',conv5_2,[3,3,512,512])
            pool5 = layer.pool_layer('pool5',conv5_3,ksize = [1,3,3,1],strides = [1,1,1,1])
    with tf.variable_scope('ssd'):
        conv6 = layer.atrous_conv('conv6',pool5,[3,3,512,1024],rate = 6,
            batch_normalization = norm,phase_train = phase_train)
        conv7 = layer.conv_layer('conv7',conv6,[1,1,1024,1024],
            batch_normalization = norm,phase_train = phase_train)
        with tf.variable_scope('conv8'):
            conv8_1 = layer.conv_layer('conv8_1',conv7,[1,1,1024,256],
                batch_normalization = norm,phase_train = phase_train)
            conv8_2 = layer.conv_layer('conv8_2',conv8_1,[3,3,256,512],
                stride = [1,2,2,1],batch_normalization = norm,phase_train = phase_train)
        with tf.variable_scope('conv9'):
            conv9_1 = layer.conv_layer('conv9_1',conv8_2,[1,1,512,128],
                batch_normalization = norm,phase_train = phase_train)
            conv9_2 = layer.conv_layer('conv9_2',conv9_1,[3,3,128,256],
                stride = [1,2,2,1],batch_normalization = norm,phase_train = phase_train)
        with tf.variable_scope('conv10'):
            conv10_1 = layer.conv_layer('conv10_1',conv9_2,[1,1,256,128],
                batch_normalization = norm,phase_train = phase_train)
            conv10_2 = layer.conv_layer('conv10_2',conv10_1,[3,3,128,256],
                padding = 'VALID',batch_normalization = norm,phase_train = phase_train)
        with tf.variable_scope('conv11'):
            conv11_1 = layer.conv_layer('conv11_1',conv10_2,[1,1,256,128],
                batch_normalization = norm,phase_train = phase_train)
            conv11_2 = layer.conv_layer('conv11_2',conv11_1,[3,3,128,256],
                padding = 'VALID',batch_normalization = norm,phase_train = phase_train)#vgg300
        with tf.variable_scope('multibox'):

            l2_conv4_3 = layer.l2_normalization('l2_normalization',conv4_3,scaling = True)
            cls4 = layer.conv_layer('cls4',l2_conv4_3,[3,3,512,84],activation = None)#conv4_3,conv10_2,conv11_2 has 4 default boxes per cell
            loc4 = layer.conv_layer('loc4',l2_conv4_3,[3,3,512,16],activation = None)

            cls4_reshape = tf.reshape(cls4,[batch_size,-1,21])
            loc4_reshape = tf.reshape(loc4,[batch_size,-1,4])


            cls7 = layer.conv_layer('cls7',conv7,[3,3,1024,126],activation = None)
            loc7 = layer.conv_layer('loc7',conv7,[3,3,1024,24],activation = None)

            cls7_reshape = tf.reshape(cls7,[batch_size,-1,21])
            loc7_reshape = tf.reshape(loc7,[batch_size,-1,4])

            cls8 = layer.conv_layer('cls8',conv8_2,[3,3,512,126],activation = None)
            loc8 = layer.conv_layer('loc8',conv8_2,[3,3,512,24],activation = None)

            cls8_reshape = tf.reshape(cls8,[batch_size,-1,21])
            loc8_reshape = tf.reshape(loc8,[batch_size,-1,4])

            cls9 = layer.conv_layer('cls9',conv9_2,[3,3,256,126],activation = None)
            loc9 = layer.conv_layer('loc9',conv9_2,[3,3,256,24],activation = None)

            cls9_reshape = tf.reshape(cls9,[batch_size,-1,21])
            loc9_reshape = tf.reshape(loc9,[batch_size,-1,4])

            cls10 = layer.conv_layer('cls10',conv10_2,[3,3,256,84],activation = None)
            loc10 = layer.conv_layer('loc10',conv10_2,[3,3,256,16],activation = None)

            cls10_reshape = tf.reshape(cls10,[batch_size,-1,21])
            loc10_reshape = tf.reshape(loc10,[batch_size,-1,4])

            cls11 = layer.conv_layer('cls11',conv11_2,[1,1,256,84],activation = None)
            loc11 = layer.conv_layer('loc11',conv11_2,[1,1,256,16],activation = None)

            cls11_reshape = tf.reshape(cls11,[batch_size,-1,21])
            loc11_reshape = tf.reshape(loc11,[batch_size,-1,4])

            cls_logit = tf.concat([
                cls4_reshape,
                cls7_reshape,
                cls8_reshape,
                cls9_reshape,
                cls10_reshape,
                cls11_reshape
                ],1)
            loc_logit = tf.concat([
                loc4_reshape,
                loc7_reshape,
                loc8_reshape,
                loc9_reshape,
                loc10_reshape,
                loc11_reshape
                ],1)
            
    return cls_logit,loc_logit
    
