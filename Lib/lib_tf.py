

def reduce_mean_nan(img):
    ## This code is for tensorflow2
    import tensorflow as tf
    # tf.__version__

    x = tf.constant(img)

    # mymean = tf.reduce_mean(tf.boolean_mask(x, tf.is_finite(x)))
    mymean = tf.reduce_mean(tf.boolean_mask(x, ~tf.is_nan(x)))

    sess = tf.Session()
    reduced_array=sess.run(mymean)
    return reduced_array

reduced_array=reduce_mean_nan(img)



def max_pool_nan(img):
    ## This code is for tensorflow
    import tensorflow as tf
    # tf.__version__

    L2=tf.nn.avg_pool2d(img, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    sess = tf.Session()
    reduced_array=sess.run(mymean)
    return reduced_array
