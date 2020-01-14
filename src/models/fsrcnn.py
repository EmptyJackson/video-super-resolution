import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.activations import relu

def fsrcnn(in_shape,
           fsrcnn_args,
           scale=2):
    """
    Returns FSRCNN model.

    fsrcnn_args = (d, s, m)

    Architecture from original paper: https://arxiv.org/pdf/1608.00367.pdf
    """

    # Initialize filter and bias tensors
    d, s, m = fsrcnn_args

    """
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
    """

    # Define model architecture
    x_in = tf.keras.Input(shape=in_shape)

    # TODO: Add normalization layer, remember to remove x_in from first conv

    with tf.name_scope("feature_extraction"):
        x = Conv2D(
            filters=d,
            kernel_size=5,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv1"
        )(x_in)
        x = relu(x)
        #x = tf.nn.bias_add(x, bias[0])
        #x = tf.nn.relu(x, name="relu")
    
    with tf.name_scope("shrinking"):
        x = Conv2D(
            filters=s,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv2"
        )(x)
        x = relu(x)
        #x = tf.nn.bias_add(x, bias[1])
        #x = tf.nn.relu(x, name="relu")

    with tf.name_scope("non-linear_mapping"):
        for i in range(m):
            x = Conv2D(
                filters=s,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                name="conv"+str(i+3)
            )(x)
            x = relu(x)
            #x = tf.nn.bias_add(x, bias[i+2])
            #x = tf.nn.relu(x, name="relu")

    with tf.name_scope("expanding"):
        x = Conv2D(
            filters=d,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv01"
        )(x)
        x = relu(x)
        #x = tf.nn.bias_add(x, bias[m+2])
        #x = tf.nn.relu(x, name="relu")

    with tf.name_scope("deconvolution"):
        x = Conv2DTranspose(
            filters=in_shape[2],
            kernel_size=9,
            strides=scale,
            padding="same",
            name="transpose_conv"
        )(x)
        x = relu(x)
        """
        tf.nn.conv2d_transpose(
            x,
            filters=filters[m+3],
            output_shape=[batch_size, in_shape[0]*scale, in_shape[0]*scale, in_shape[2]],
            strides=[1, scale, scale, 1],
            padding="SAME",
            name="transpose_conv"
        )
        """
        #x = tf.nn.bias_add(x, bias[m+3])

    # TODO: Include bias add here?

    # TODO: Add denormalization layer

    return tf.keras.Model(x_in, x, name='fsrcnn')
