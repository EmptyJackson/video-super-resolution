import tensorflow as tf
from tensorflow.python.keras.models import Model

def fsrcnn(in_shape,
           (d, s, m),
           scale=2):
    """
    Architecture from original paper: https://arxiv.org/pdf/1608.00367.pdf
    """

    # Initialize filter and bias tensors
    filters = [
        tf.Variable(tf.random_normal([5, 5, 1, d], stddev=0.1), name="fil1"),
        tf.Variable(tf.random_normal([1, 1, d, s], stddev=0.1), name="fil2")
    ]
    for i in range(m):
        filters.append(tf.random_normal([3, 3, s, s], stddev=0.1), name="fil"+str(fid+3))
    filters.append(tf.random_normal([1, 1, s, d], stddev=0.1), name="fil"+str(m+3))
    filters.append(tf.random_normal([9, 9, 1, d], stddev=0.1), name="fil"+str(m+4))
    
    bias = 
        tf.get_variable(shape=[d], initializer=bias_initializer, name="bias1"),
        tf.get_variable(shape=[s], initializer=bias_initializer, name="bias2")
    ]
    for i in range(m):
        bias.append(tf.get_variable(shape=[s], initializer=bias_initializer, name="bias"+str(i+3)))
    bias.append(tf.get_variable(shape=[d], initializer=bias_initializer, name="bias"+str(m+3)))
    bias.append(tf.get_variable(shape=[1], initializer=bias_initializer, name="bias"+str(m+4)))

    # Define model architecture
    x_in = tf.layers.Input(shape=in_shape)

    # TODO: Add normalization layer, remember to remove x_in from first conv

    with tf.name_scope("feature_extraction"):
        x = tf.layers.conv2d(
            x_in,
            filters=filters[0],
            padding="same",
            name="conv")
        x = x + bias[0]
        x = tf.nn.relu(x, name="relu")

    with tf.name_scope("shrinking"):
        x = tf.layers.conv2d(
            x,
            filters=filters[1],
            padding="same",
            name="conv")
        x = x + bias[1]
        x = tf.nn.relu(x, name="relu")

    with tf.name_scope("non-linear_mapping"):
        for i in range(m):
                x = tf.layers.conv2d(
                x,
                filters=filters[i+2],
                padding="same",
                name="conv"+str(i+1))
            x = x + bias[i+2]    
        x = tf.nn.relu(x, name="relu")

    with tf.name_scope("expanding"):
        x = tf.layers.conv2d(
            x,
            filters=filters[m+2],
            padding="same",
            name="conv")
        x = tf.nn.bias_add(x, bias[m+2]
        x = tf.nn.relu(x, name="relu")

    with tf.name_scope("deconvolution"):
        x = tf.layers.conv2d_transpose(
            x,
            filters=filters[m+3],
            output_shape=[1, in_shape*scale, in_shape*scale, in_shape[2]],
            strides=[1, scale, scale, 1]
            padding="same",
            name="transpose_conv"
        )

    # TODO: Include bias add here?

    # TODO: Add denormalization layer

    return tf.models.Model(x_in, x, name='fsrcnn')
