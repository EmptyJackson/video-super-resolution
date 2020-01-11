import tensorflow as tf
from tensorflow.python.keras.models import Model

def fsrcnn(in_shape,
           fsrcnn_args,
           batch_size,
           scale=2):
    """
    Returns FSRCNN model.

    fsrcnn_args = (d, s, m)

    Architecture from original paper: https://arxiv.org/pdf/1608.00367.pdf
    """

    # Initialize filter and bias tensors
    d, s, m = fsrcnn_args

    filters = [
        tf.Variable(tf.random.normal([5, 5, in_shape[2], d], stddev=0.01, name="fil1")),
        tf.Variable(tf.random.normal([1, 1, d, s], stddev=0.01, name="fil2"))
    ]
    for i in range(m):
        filters.append(tf.Variable(
            tf.random.normal([3, 3, s, s], stddev=0.01, name="fil"+str(i+3))))
    filters.append(tf.Variable(tf.random.normal([1, 1, s, d], stddev=0.01, name="fil"+str(m+3))))
    filters.append(tf.Variable(tf.random.normal([9, 9, 1, d], stddev=0.01, name="fil"+str(m+4))))

    bias = [
        tf.Variable(tf.zeros([d]), name="bias1"),
        tf.Variable(tf.zeros([s]), name="bias2")
    ]
    for i in range(m):
        bias.append(tf.Variable(tf.zeros([s]), name="bias"+str(i+3)))
    bias.append(tf.Variable(tf.zeros([d]), name="bias"+str(m+3)))
    bias.append(tf.Variable(tf.zeros([in_shape[2]]), name="bias"+str(m+4)))

    # Define model architecture
    x_in = tf.keras.Input(shape=in_shape)

    # TODO: Add normalization layer, remember to remove x_in from first conv

    with tf.name_scope("feature_extraction"):
        x = tf.nn.conv2d(
            x_in,
            filters=filters[0],
            strides=[1,1,1,1],
            padding="SAME",
            name="conv")
        x = tf.nn.bias_add(x, bias[0])
        #x = tf.nn.relu(x, name="relu")
    
    with tf.name_scope("shrinking"):
        x = tf.nn.conv2d(
            x,
            filters=filters[1],
            strides=[1,1,1,1],
            padding="SAME",
            name="conv")
        x = tf.nn.bias_add(x, bias[1])
        #x = tf.nn.relu(x, name="relu")

    with tf.name_scope("non-linear_mapping"):
        for i in range(m):
            x = tf.nn.conv2d(
                x,
                filters=filters[i+2],
                strides=[1,1,1,1],
                padding="SAME",
                name="conv"+str(i+1))
            x = tf.nn.bias_add(x, bias[i+2])
            #x = tf.nn.relu(x, name="relu")

    with tf.name_scope("expanding"):
        x = tf.nn.conv2d(
            x,
            filters=filters[m+2],
            strides=[1,1,1,1],
            padding="SAME",
            name="conv")
        x = tf.nn.bias_add(x, bias[m+2])
        #x = tf.nn.relu(x, name="relu")

    with tf.name_scope("deconvolution"):
        x = tf.nn.conv2d_transpose(
            x,
            filters=filters[m+3],
            output_shape=[batch_size, in_shape[0]*scale, in_shape[0]*scale, in_shape[2]],
            strides=[1, scale, scale, 1],
            padding="SAME",
            name="transpose_conv"
        )
        #x = tf.nn.bias_add(x, bias[m+3])

    # TODO: Include bias add here?

    # TODO: Add denormalization layer

    return tf.keras.Model(x_in, x, name='fsrcnn')
