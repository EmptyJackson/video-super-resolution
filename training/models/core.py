import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Concatenate, Conv2D, Conv2dTranspose, ConvLSTM2D, Lambda, PReLU, ReLU

from enum import Enum

class Size(Enum):
    SMALL = 1
    MED = 2
    LARGE = 3

class Upscale(Enum):
    DECONV = 1
    SUB_PIXEL = 2

class Residual(Enum):
    NONE = 1
    LOCAL = 2
    GLOBAL = 3

class Activation(Enum):
    RELU = 1
    PRELU = 2


def core_model(
    scale,
    size,
    upscale=Upscale.SUB_PIXEL,
    residual=Residual.NONE,
    activation=Activation.RELU,
    activation_removal=False,
    recurrent=False):

    if size == Size.LARGE:
        depth = 16
        num_filters = 64
    elif size == Size.MED:
        depth = 8
        num_filters = 32
    else:
        depth = 2
        num_filters = 16

    #x_in = tf.keras.Input(shape=in_shape)
    if not recurrent:
        # single-image
        x_in = tf.keras.Input(shape=(None, None, 3))
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv1"
        )(x_in)
    else:
        # video-based
        x_in = tf.keras.Input(shape=(None, None, None, 3))
        x = ConvLSTM2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="start_lstm_conv"
        )(x)

    x_res = x
    for i in range(depth):
        if residual == Residual.LOCAL:
            x_res = x
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv" + str(2 + 2*i)
        )(x)
        if activation == Activation.PRELU:
            x = PReLU()(x)
        elif activation == Activation.RELU:
            x = ReLU()(x)
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv" + str(3 + 2*i)
        )(x)
        if not activation_removal:
            if activation == Activation.PRELU:
                x = PReLU()(x)
            elif activation == Activation.RELU:
                x = ReLU()(x)
        if residual == Residual.GLOBAL:
            x_res = Concatenate([x_res, x])
            x = x_res
        if residual == Residual.LOCAL:
            x = Add()([x_res, x])
    
    if not recurrent:
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="final_conv"
        )(x)
    else:
        x = ConvLSTM2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="final_lstm_conv"
        )(x)

    if not scale in [2, 4]:
        raise ValueError('Scale must be in [2, 4]')

    if upscale == Upscale.SUB_PIXEL:
        # Sub-pixel convolution
        up_factor = 2
        subpixel_layer = Lambda(lambda x: tf.nn.depth_to_space(x, up_factor))
        x = Conv2D(
            filters=3*(up_factor**2),
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
                filters=3*(up_factor**2),
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                name="subpix_conv2"
            )(x)
            x = subpixel_layer(inputs=x)
    elif upscale == Upscale.DECONV:
        # Deconvolution
        x = Conv2DTranspose(
            filters=3,
            kernel_size=9,
            strides=scale,
            padding="same",
            name="transpose_conv"
        )(x)
    #x = ReLU()(x)

    # Return model and learning rate multiplier
    return Model(x_in, x, name="core"), None
