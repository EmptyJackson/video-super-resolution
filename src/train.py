import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataset import Dataset
from models.fsrcnn import fsrcnn
from models.simple import simple_model

CHECKPOINT_DIR = './checkpoints/'
MODEL_WEIGHTS_PATH = lambda model, x, r, i: CHECKPOINT_DIR + model + '_x' + str(x) + '_' + r + '_' + str(i) + '.h5'

"""
def halt_training(model, criterion):
    [Something based on testing learning rate, epoch etc.]
    return true/false
"""

@tf.function
def train(model, tf_dataset, stopping_criterion, learn_rate, ckpt_args, dataset_size):
    if ckpt_args.completed:
        weights_path = MODEL_WEIGHTS_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res, ckpt_args.completed)
        model.load_weights(weights_path)
        tf.print('Restored model weights from ' + weights_path)
    else:
        tf.print('Training model from scratch')

    opt = tf.optimizers.Adam(learn_rate)
    for epoch in range(1, stopping_criterion['epochs']+1):
        progbar = tf.keras.utils.Progbar(dataset_size, unit_name='batch')
        for lr_image, hr_image in tf_dataset:
            train_step(model, opt, tf.losses.mean_squared_error, lr_image, hr_image)
            progbar.add(1)

        tf.print("Epoch complete.")
        if ckpt_args.epochs and (epoch % ckpt_args.epochs) == 0:
            weights_path = MODEL_WEIGHTS_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res, epoch+ckpt_args.completed)
            model.save_weights(weights_path)
            tf.print(weights_path + " saved.")

@tf.function
def train_step(model, opt, loss_fn, lr_image, hr_image):
    lr_image = tf.cast(lr_image, tf.float32) / 255.0
    hr_image = tf.cast(hr_image, tf.float32) / 255.0

    tf.debugging.check_numerics(lr_image, 'low')
    tf.debugging.check_numerics(hr_image, 'high')

    with tf.GradientTape() as tape:
        hr_pred = model(lr_image)
        tf.debugging.check_numerics(hr_pred, 'pred')
        loss = loss_fn(hr_image, hr_pred)
        #tf.print(hr_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    #tf.print(hr_pred)
    #tf.print(loss)
    #tf.print(tf.reduce_mean(tf.image.psnr(hr_image, hr_pred, max_val=1.0)))
    #tf.reduce_mean(tf.reduce_sum(tf.square(self.labels - self.pred), reduction_indices=0))
    #tf.print(tf.reduce_mean(loss))

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
        self.res = str(res) + 'p'

def main():
    parser=argparse.ArgumentParser(
        description="Model training script\n" +\
                    "Usage: train.py [options]"
    )
    parser.add_argument('--model', default="fsrcnn", help="name of model")
    parser.add_argument('--scale', default=2, help="factor by which to upscale the given model")
    parser.add_argument('--epochs', default=100, help="training epochs")
    parser.add_argument('--pre_epochs', default=0, help="restores model weights from checkpoint with given epochs of pretraining; set to 0 to train from scratch")
    parser.add_argument('--ckpt_epochs', default=0, help="number of training epochs in between checkpoints; set to 0 to not save checkpoints")

    args = parser.parse_args()

    lr_shape = [240,352,3]
    batch_size = 4

    ckpt_args = CheckpointArgs(
        epochs=args.ckpt_epochs,
        completed=args.pre_epochs,
        model=args.model,
        scale=args.scale,
        res=lr_shape[0]
    )
    
    if args.model == 'fsrcnn':
        model = fsrcnn(
            in_shape=lr_shape,
            fsrcnn_args=(48, 12, 2),#(48,12,3),  # (d, s, m)
            scale=2
        )
    """
    model = simple_model(
        in_shape=lr_shape,
        scale=2
    )
    """

    stopping_criterion = {
        'epochs': args.epochs
    }

    div2k = Dataset(
        'div2k',
        lr_shape=lr_shape,
        scale=2,
        subset='valid',
        downscale='bicubic',
        batch_size=batch_size,
        prefetch_buffer_size=4
    )
    tf_dataset = div2k.build_tf_dataset()
    
    # CHANGE LR BACK TO 1E-3!!!!!!!!!! (WITH LARGER BATCH SIZE)
    train(model, tf_dataset, stopping_criterion, 1e-4, ckpt_args, div2k.get_num_batches())


if __name__=='__main__':
    main()
