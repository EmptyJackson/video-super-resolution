import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import get_resolution
from dataset import Dataset
from models.model_io import load_model, save_model_arch, save_model_weights

"""
def halt_training(model, criterion):
    [Something based on testing learning rate, epoch etc.]
    return true/false
"""

def train(model, train_dataset, val_dataset, stopping_criterion, learn_rate, ckpt_args, train_batches):
    if ckpt_args.epochs and not ckpt_args.completed:
        save_model_arch(model, ckpt_args)

    opt = tf.optimizers.Adam(learn_rate)
    for epoch in range(1, stopping_criterion['epochs']+1):
        train_loss = 0.
        progbar = tf.keras.utils.Progbar(train_batches, unit_name='batch')
        for lr_batch, hr_batch in train_dataset:
            train_loss += train_step(model, opt, tf.losses.mean_squared_error, lr_batch, hr_batch)
            progbar.add(1)

        tf.print("Training epoch complete, calculating validation loss...")
        train_loss /= train_batches
        val_loss = eval_model(model, val_dataset, tf.losses.mean_squared_error)
        tf.print("Train loss:", train_loss, "   Validation loss: ", val_loss)

        if ckpt_args.epochs and (epoch % ckpt_args.epochs) == 0:
            save_model_weights(model, epoch, ckpt_args)
            tf.print()

@tf.function
def train_step(model, opt, loss_fn, lr_batch, hr_batch):
    #lr_batch = tf.cast(lr_batch, tf.float32) / 255.0
    #hr_batch = tf.cast(hr_batch, tf.float32) / 255.0

    tf.debugging.check_numerics(lr_batch, 'low')
    tf.debugging.check_numerics(hr_batch, 'high')

    with tf.GradientTape() as tape:
        hr_pred = model(lr_batch)
        tf.debugging.check_numerics(hr_pred, 'pred')
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
    def __init__(self, epochs, completed=0, model="", scale=2, res=240):
        if epochs < 0:
            raise ValueError('Epoch checkpoint frequency must be non-negative.')
        self.epochs = epochs
        self.completed = completed
        self.model = model
        self.scale = scale
        self.res = res

def main():
    parser=argparse.ArgumentParser(
        description="Model training script\n" +\
                    "Usage: train.py [options]"
    )
    parser.add_argument('--model', default="fsrcnn", help="name of model")
    parser.add_argument('--scale', default=2, type=int, help="factor by which to upscale the given model")
    parser.add_argument('--resolution', default=240, type=int, help="height of low resolution image")
    parser.add_argument('--epochs', default=100, type=int, help="training epochs")
    parser.add_argument('--pre_epochs', default=0, type=int,  help="restores model weights from checkpoint with given epochs of pretraining; set to 0 to train from scratch")
    parser.add_argument('--ckpt_epochs', default=0, type=int, help="number of training epochs in between checkpoints; set to 0 to not save checkpoints")

    args = parser.parse_args()

    lr_shape = get_resolution(args.resolution)
    lr_shape.append(3) # Colour channels
    batch_size = 16

    ckpt_args = CheckpointArgs(
        epochs=args.ckpt_epochs,
        completed=args.pre_epochs,
        model=args.model,
        scale=args.scale,
        res=args.resolution
    )

    stopping_criterion = {
        'epochs': args.epochs
    }

    print("Building datasets")
    div2k = Dataset(
        'div2k',
        lr_shape=lr_shape,
        scale=args.scale,
        downscale='bicubic',
        batch_size=batch_size,
        prefetch_buffer_size=4
    )
    train_dataset = div2k.build_dataset('train')
    val_dataset = div2k.build_dataset('valid')

    model = load_model(ckpt_args)
    train(model, train_dataset, val_dataset, stopping_criterion, 1e-3, ckpt_args, div2k.get_num_train_batches())


if __name__=='__main__':
    main()
