import tensorflow as tf
import input_reader
import sub_sampler

flags = tf.app.flags
flags.DEFINE_integer('batch_size',1,'batch size for input queue')
flags.DEFINE_integer('dataset_size',11540,'dataset size')
flags.DEFINE_integer('image_size',300,'image height and width')
flags.DEFINE_boolean('is_train',True,'train if true')
flags.DEFINE_string('record_path','../pascal.record','record path')

FLAGS = flags.FLAGS

def main(_):
    with tf.Graph().as_default():
        image,label,target,xmin,ymin,xmax,ymax,classes = input_reader.input(record_path = FLAGS.record_path,dataset_size = FLAGS.dataset_size,batch_size = FLAGS.batch_size,image_size = FLAGS.image_size)
        label_indices = tf.equal(label[0],1)
        sampled = sub_sampler.sub_sample(label_indices,2)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            
            for i in range(3):
                a = sess.run(sampled)
                print(a)
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
