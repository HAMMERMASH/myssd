import tensorflow as tf

def sub_sample(indicator,
            num_samples,
            indices_value = 1,
            default_value = 0,
            dtype = tf.float32):
    """
        input a 1-D tensor, sub-sample indices and fill it to original dimension
        Args:
            indicator: A 1-D boolean tensor [num_anchors] indicating which elements are allowed to be sampled
            indices_value: A scalar indicating what value for the sampled elements
            default_value: A scalar indicating what value for the rest elements
        Returns:
            filld_indices: A 1-D tensor [num_anchors] where elements equals one indicating sampled result
    """
    total_samples = tf.size(indicator)
    indices = tf.where(indicator)
    indices = tf.random_shuffle(indices)
    indices = tf.reshape(indices,[-1])

    selected_indices = tf.slice(indices,[0],tf.reshape(num_samples,[1]))
    
    size = tf.to_int32(total_samples)
    zeros = tf.ones([size],dtype = dtype) * default_value
    values = tf.ones_like(selected_indices,dtype = dtype) * indices_value
    filled_indices = tf.dynamic_stitch([tf.range(size),tf.to_int32(selected_indices)],[zeros,values])

    return filled_indices

def pad_with_indices(indices,
                    num_samples,
                    size,
                    indices_value = 1,
                    default_value = 0,
                    dtype = tf.float32):

    selected_indices = tf.slice(indices,[0],tf.reshape(num_samples,[1]))
    size = tf.to_int32(size)
    zeros = tf.ones([size],dtype = dtype) * default_value
    values = tf.ones_like(selected_indices,dtype = dtype) * indices_value
    padded_tensor = tf.dynamic_stitch([tf.range(size),tf.to_int32(selected_indices)],[zeros,values])

    return padded_tensor

