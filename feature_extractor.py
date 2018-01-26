import layer

def vgg16(image):

    conv1_1 = layer.conv_layer('conv1_1',image,[3,3,3,64])
    conv1_2 = layer.conv_layer('conv1_2',conv1_1,[3,3,64,64])
    pool1 = layer.pool_layer('pool1',conv1_2)
    conv2_1 = layer.conv_layer('conv2_1',pool1,[3,3,64,128])
    conv2_2 = layer.conv_layer('conv2_2',conv2_1,[3,3,128,128])
    pool2 = layer.pool_layer('pool2',conv2_2)
    conv3_1 = layer.conv_layer('conv3_1',pool2,[3,3,128,256])
    conv3_2 = layer.conv_layer('conv3_2',conv3_1,[3,3,256,256])
    conv3_3 = layer.conv_layer('conv3_3',conv3_2,[3,3,256,256])
    pool3 = layer.pool_layer('pool3',conv3_3)
    conv4_1 = layer.conv_layer('conv4_1',pool3,[3,3,256,512])
    conv4_2 = layer.conv_layer('conv4_2',conv4_1,[3,3,512,512])
    conv4_3 = layer.conv_layer('conv4_3',conv4_2,[3,3,512,512])
    pool4 = layer.pool_layer('pool4',conv4_3)
    conv5_1 = layer.conv_layer('conv5_1',pool4,[3,3,512,512])
    conv5_2 = layer.conv_layer('conv5_2',conv5_1,[3,3,512,512])
    conv5_3 = layer.conv_layer('conv5_3',conv5_2,[3,3,512,512])

    return conv5_3

def extract(base_net,image):

    """
        choose a base net to build
        Args:
            base_net: a string indicating which base net to choose
            image: a [batch_size,height,width,3] tensor
        Returns:
            convolution results
    """

    if base_net == 'vgg16':
        return vgg16(image)
    else:
        raise ValueError('Unknown extractor.')
