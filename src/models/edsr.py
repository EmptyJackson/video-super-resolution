import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Add, Conv2D, Lambda
from tensorflow.keras.activations import relu

def edsr(in_shape, scale=2, num_filters=64, num_res_blocks=16):
    x_in = tf.keras.Input(shape=in_shape)

    x = Conv2D(
        filters=num_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        name="conv1"
    )(x_in)
    x_st = x
    for i in range(num_res_blocks):
        with tf.name_scope("res_block_" + str(i)):
            x_res = x
            x = Conv2D(
                filters=num_filters,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                name="conv" + str(2 + 2*i)
            )(x)
            x = relu(x)
            x = Conv2D(
                filters=num_filters,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                name="conv" + str(3 + 2*i)
            )(x)
            x = Add()([x_res, x])
            x = relu(x)

    x = Conv2D(
        filters=num_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        name="long_res_conv"
    )(x)
    x = Add()([x_st, x])

    # Sub-pixel convolution
    up_factor = 2
    subpixel_layer = Lambda(lambda x: tf.nn.depth_to_space(x, up_factor))
    x = Conv2D(
        filters=num_filters * (up_factor ** 2),
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        name="subpix_conv"
    )(x)
    x = subpixel_layer(inputs=x)
    if scale == 4:
        # Second sub-pixel convolution for x4 upscaling
        x = Conv2D(
            filters=num_filters * (up_factor ** 2),
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="subpix_conv2"
        )(x)
        x = subpixel_layer(inputs=x)
    elif scale != 2:
        raise ValueError('EDSR scale must be in [2, 4]')

    x = Conv2D(
        filters=3,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        name="up_conv"
    )(x)
    x = relu(x)
    return Model(x_in, x, name="edsr")
