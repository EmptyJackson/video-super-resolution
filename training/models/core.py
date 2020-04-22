import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Concatenate, Conv2D, Conv2DTranspose, ConvLSTM2D, Lambda, PReLU, ReLU
import tensorflow.keras.backend as K

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

class CoreArgs:
    def __init__(self, scale, size, upscale, residual, activation, activation_removal, recurrent):
        self.scale = scale
        size_dict = {'s':Size.SMALL, 'm':Size.MED, 'l':Size.LARGE}
        upscale_dict = {'de':Upscale.DECONV, 'sp':Upscale.SUB_PIXEL}
        residual_dict = {'n':Residual.NONE, 'l':Residual.LOCAL, 'g':Residual.GLOBAL}
        activation_dict = {'r':Activation.RELU, 'p':Activation.PRELU}
        self.size = size_dict[size]
        self.upscale = upscale_dict[upscale]
        self.residual = residual_dict[residual]
        self.activation = activation_dict[activation]
        self.activation_removal = activation_removal
        self.recurrent = recurrent

def core_model(args):
    if not args.scale in [2, 4]:
        raise ValueError('Scale must be in [2, 4]')

    if args.size == Size.LARGE:
        depth = 16
        num_filters = 64
    elif args.size == Size.MED:
        depth = 4
        num_filters = 32
    else:
        depth = 1
        num_filters = 16

    x_in = tf.keras.Input(shape=(None, None, 3))
    if not args.recurrent:
        # single-image
        x = Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv1"
        )(x_in)
    else:
        # video-based
        x = K.expand_dims(x_in, 0)
        x = ConvLSTM2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            return_sequences=True,
            name="start_lstm_conv"
        )(x)
        x = K.squeeze(x, 0)

    if args.residual == Residual.LOCAL:
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv1"
        )(x)
    elif args.residual == Residual.LOCAL:
        x_res = x

    for i in range(depth):
        if args.residual == Residual.LOCAL:
            x_res = x
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv" + str(2 + 2*i)
        )(x)
        if args.activation == Activation.PRELU:
            x = PReLU(shared_axes=[1,2])(x)
        elif args.activation == Activation.RELU:
            x = ReLU()(x)
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="conv" + str(3 + 2*i)
        )(x)
        if not args.activation_removal:
            if args.activation == Activation.PRELU:
                x = PReLU(shared_axes=[1,2])(x)
            elif args.activation == Activation.RELU:
                x = ReLU()(x)
        if args.residual == Residual.GLOBAL:
            x_res = Concatenate()([x_res, x])
            x = x_res
        if args.residual == Residual.LOCAL:
            x = Add()([x_res, x])
    
    if not args.recurrent:
        x = Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            name="final_conv"
        )(x)
    else:
        x = K.expand_dims(x, 0)
        x = ConvLSTM2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            return_sequences=True,
            name="final_lstm_conv"
        )(x)
        x = K.squeeze(x, 0)

    if args.upscale == Upscale.SUB_PIXEL:
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
        if args.scale == 4:
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
    elif args.upscale == Upscale.DECONV:
        # Deconvolution
        x = Conv2DTranspose(
            filters=3,
            kernel_size=9,
            strides=args.scale,
            padding="same",
            name="transpose_conv"
        )(x)
    #x = ReLU()(x)

    # Return model and learning rate multiplier
    return Model(x_in, x, name="core"), None
