import tensorflow as tf
from tensorflow.python.keras.models import Model

def simple_model(in_shape,
                 batch_size,
                 scale=2):
    """
    Simple two layer CNN for testing purposes
    """

    filters = [
        tf.Variable(tf.random.normal([5, 5, in_shape[2], 4], stddev=0.01, name="fil1")),
        tf.Variable(tf.random.normal([9, 9, 1, 4], stddev=0.01, name="fil2"))
    ]

    bias = tf.Variable(tf.zeros([4]), name="bias")

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
        x = tf.nn.bias_add(x, bias)
        #x = tf.nn.relu(x, name="relu")

    with tf.name_scope("deconvolution"):
        x = tf.nn.conv2d_transpose(
            x,
            filters=filters[1],
            output_shape=[batch_size, in_shape[0]*scale, in_shape[0]*scale, in_shape[2]],
            strides=[1, scale, scale, 1],
            padding="SAME",
            name="transpose_conv"
        )

    # TODO: Include bias add here?

    # TODO: Add denormalization layer

    return tf.keras.Model(x_in, x, name='fsrcnn')
