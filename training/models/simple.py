import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

def simple_model(in_shape,
                 scale=2):
    """
    Simple two layer CNN for testing purposes
    """

    # Define model architecture
    x_in = tf.keras.Input(shape=in_shape)

    # TODO: Add normalization layer, remember to remove x_in from first conv

    with tf.name_scope("feature_extraction"):
        x = Conv2D(
            filters=4,
            kernel_size=5,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv"
        )(x_in)
        #x = tf.keras.backend.bias_add(x, bias)
        #x = tf.nn.relu(x, name="relu")

    with tf.name_scope("deconvolution"):
        x = Conv2DTranspose(
            filters=3,
            kernel_size=9,
            strides=scale,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="transpose_conv"
        )(x)

    # TODO: Include bias add here?

    # TODO: Add denormalization layer
    
    model = tf.keras.Model(x_in, x, name='simple')

    return model
