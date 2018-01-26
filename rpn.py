import tensorflow as tf
import layer

def inference(feature,k):
    """
        rpn part net work

        Args:
            feature: a 4-D tensor [batch_size,height,width,n_channel] tensor, the output of last conv layer of feature extractor
            k: a scalar indicating number of predictions per cell

        Returns:
            rpn_obj: a 4-D tensor [batch_size,height,width,2*k], the prediction of objectness score
            rpn_box: a 4-D tensor [batch_size,height,width,4*k], the prediction of box coordinates
    """
    shape = feature.get_shape().as_list()

    rpn_conv = layer.conv_layer('rpn_conv',feature,[3,3,shape[3],256])
    rpn_obj = layer.conv_layer('rpn_obj_conv',rpn_conv,[1,1,256,2*k])
    rpn_box = layer.conv_layer('rpn_box_conv',rpn_conv,[1,1,256,4*k])

    return rpn_obj,rpn_box
