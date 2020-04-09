import os
import sys
import argparse
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras_lr_multiplier import LRMultiplier

from utils import get_resolution
from data_loader import ImageLoader, VideoLoader
from model_loader import load_model, save_model_arch, save_model_weights
from models.core import CoreArgs

"""
def halt_training(model, criterion):
    [Something based on testing learning rate, epoch etc.]
    return true/false
"""

def train(model, loader, stopping_criterion, learn_rate, ckpt_args, train_batches, rebuild_freq, lr_mul=None):
    tf.print("Building datasets")
    train_dataset = loader.build_dataset('train')
    if train_dataset is None:
        raise ValueError("Training dataset missing")
    val_dataset = loader.build_dataset('valid')
    
    if ckpt_args.epochs and not ckpt_args.completed:
        save_model_arch(model, ckpt_args)

    opt = tf.optimizers.Adam(learn_rate)
    #if not lr_mul is None:
    #    opt = LRMultiplier(opt, lr_mul)

    model.compile(optimizer=opt, loss=tf.losses.mean_squared_error)

    for epoch in range(1, stopping_criterion['epochs']+1):
        """
        train_loss = 0.
        progbar = tf.keras.utils.Progbar(train_batches, unit_name='batch')
        for lr_batch, hr_batch in train_dataset:
            #train_loss += train_step(model, opt, tf.losses.mean_squared_error, lr_batch, hr_batch)
            progbar.add(1)
        """
        history = model.fit(train_dataset, epochs=1, steps_per_epoch=train_batches, verbose=1)

        if val_dataset is None:
            tf.print("Train loss:", history.history['loss'][0])
        else:
            tf.print("Training epoch complete, calculating validation loss...")
            val_loss = eval_model(model, val_dataset, tf.losses.mean_squared_error)
            tf.print("Train loss:", history.history['loss'][0], "   Validation loss: ", val_loss)

        if ckpt_args.epochs and (epoch % ckpt_args.epochs) == 0:
            save_model_weights(model, epoch, ckpt_args)
            tf.print()

        if (epoch % rebuild_freq) == 0:
            tf.print("Rebuilding train dataset")
            train_dataset = loader.build_dataset('train')

@tf.function
def train_step(model, opt, loss_fn, lr_batch, hr_batch):
    """
    DEPRECATED
    """
    #lr_batch = tf.cast(lr_batch, tf.float32) / 255.0
    #hr_batch = tf.cast(hr_batch, tf.float32) / 255.0

    #tf.debugging.check_numerics(lr_batch, 'low')
    #tf.debugging.check_numerics(hr_batch, 'high')

    with tf.GradientTape() as tape:
        hr_pred = model(lr_batch)
        #tf.debugging.check_numerics(hr_pred, 'pred')
        loss = loss_fn(hr_batch, hr_pred)
        #tf.print(hr_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    #tf.print(hr_pred)
    #tf.print(loss)
    #tf.print(tf.reduce_mean(tf.image.psnr(hr_image, hr_pred, max_val=1.0)))
    #tf.reduce_mean(tf.reduce_sum(tf.square(self.labels - self.pred), reduction_indices=0))
    #tf.print(tf.reduce_mean(loss))
    return tf.reduce_mean(loss)

@tf.function
def eval_model(model, dataset, loss_fn):
    i = 0.
    loss = 0.
    for lr_image, hr_image in dataset:
        hr_pred = model(lr_image)
        loss += tf.reduce_mean(loss_fn(hr_image, hr_pred))
        i += 1.
    loss /= i
    return loss

class CheckpointArgs:
    """
    Stores training checkpoint information.
    Set epochs = 0 to disable checkpointing.
    """
    def __init__(self, epochs, completed=0, model="", scale=2, res=240, core_args=None):
        if epochs < 0:
            raise ValueError('Epoch checkpoint frequency must be non-negative.')
        self.epochs = epochs
        self.completed = completed
        self.model = model
        self.scale = scale
        self.res = res
        self.core_args = core_args


def main():
    parser=argparse.ArgumentParser(
        description="Model training script\n" +\
                    "Usage: train.py [options]"
    )
    parser.add_argument('--model', default="fsrcnn", help="name of model")
    parser.add_argument('--scale', default=2, type=int, help="factor by which to upscale the given model")
    parser.add_argument('--resolution', default=240, type=int, help="height of low resolution image")
    parser.add_argument('--learn_rate', default=1e-3, type=float, help="initial learning rate of optimizer")
    parser.add_argument('--rebuild_freq', default=20, type=int, help="frequency to rebuild dataset (epochs)")
    parser.add_argument('--epochs', default=100, type=int, help="training epochs")
    parser.add_argument('--pre_epochs', default=0, type=int,  help="restores model weights from checkpoint with given epochs of pretraining; set to 0 to train from scratch")
    parser.add_argument('--ckpt_epochs', default=0, type=int, help="number of training epochs in between checkpoints; set to 0 to not save checkpoints")
    parser.add_argument('--size', default='s', type=str, choices=['s', 'm', 'l'], help="size of core model")
    parser.add_argument('--upscale', default='sp', type=str, choices=['de', 'sp'], help="upscale method of core model")
    parser.add_argument('--residual', default='n', type=str, choices=['n', 'l', 'g'], help="residual method of core model")
    parser.add_argument('--activation', default='r', type=str, choices=['r', 'p'], help="activation of core model")
    parser.add_argument('--activation_removal', action='store_true', help="activation removal in core model")
    parser.add_argument('--recurrent', action='store_true', help="recurrent core model")

    args = parser.parse_args()

    core_args = None
    if args.model == 'core':
        core_args = CoreArgs(args.scale, args.size, args.upscale, args.residual, args.activation, args.activation_removal, args.recurrent)

    ckpt_args = CheckpointArgs(
        epochs=args.ckpt_epochs,
        completed=args.pre_epochs,
        model=args.model,
        scale=args.scale,
        res=args.resolution,
        core_args=core_args
    )

    stopping_criterion = {
        'epochs': args.epochs
    }

    if args.model == 'core' and args.recurrent:
        loader = VideoLoader(
            'reds',
            scale=args.scale,
            prefetch_buffer_size=4
        )
    else:
        lr_shape = get_resolution(args.resolution)
        lr_shape.append(3) # Colour channels
        batch_size = 16
        loader = ImageLoader(
            'div2k',
            lr_shape=lr_shape,
            scale=args.scale,
            batch_size=batch_size,
            prefetch_buffer_size=4
        )

    model, lr_mul = load_model(ckpt_args)
    train(model, loader, stopping_criterion, args.learn_rate, ckpt_args,
          loader.get_num_train_batches(), args.rebuild_freq, lr_mul)


if __name__=='__main__':
    main()
