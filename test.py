import numpy as np
import tensorflow as tf

np.random.seed(2016)
data = np.random.randn(100000).astype(np.float32)

vec = tf.placeholder(tf.float32, data.shape)
avg = tf.reduce_mean(vec)

avgs = []
with tf.Session() as sess:
    for _ in range(100):
        avgs.append(sess.run(avg, feed_dict={vec: data}))
            
print(min(avgs) == max(avgs))
print(max(avgs) - min(avgs))
