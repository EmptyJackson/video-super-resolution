import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataset import Dataset
from models.fsrcnn import fsrcnn
from models.simple import simple_model

"""
def halt_training(model, criterion):
    [Something based on testing learning rate, epoch etc.]
    return true/false
"""

def train(model, stopping_criterion, tf_dataset):
    # for epoch in range(1, stopping_criterion.epochs+1):
    for lr_image, hr_image in tf_dataset:#.take(10):
        train_step(model, tf.losses.mean_absolute_error, lr_image, hr_image, 1e-4)


@tf.function()
def train_step(model, loss_fn, lr_image, hr_image, learn_rate):
    lr_image = tf.cast(lr_image, tf.float32) / 255.0
    hr_image = tf.cast(hr_image, tf.float32) / 255.0

    tf.debugging.check_numerics(lr_image, 'low')
    tf.debugging.check_numerics(hr_image, 'high')

    with tf.GradientTape() as tape:
        hr_pred = model(lr_image)
        tf.debugging.check_numerics(hr_pred, 'pred')
        loss = loss_fn(hr_image, hr_pred)
        tf.print(hr_pred)

    hr_pred = model(lr_image)
    tf.debugging.check_numerics(hr_pred, 'pred')

    grads = tape.gradient(loss, model.trainable_variables)
    opt = tf.optimizers.Adam(learn_rate)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    tf.print(hr_pred)
    #tf.print(loss)
    #tf.print(tf.reduce_mean(tf.image.psnr(hr_image, hr_pred, max_val=1.0)))
    return opt


def main():
    lr_shape = [100,100,3]
    batch_size = 4
    
    """
    model = fsrcnn(
        in_shape=lr_shape,
        fsrcnn_args=(1, 1, 0),#(48,12,3),  # (d, s, m)
        batch_size=batch_size,
        scale=2
    )
    """
    model = simple_model(
        in_shape=lr_shape,
        batch_size=batch_size,
        scale=2
    )
    stopping_criterion = {
        'epochs': 10
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

    train(model, stopping_criterion, tf_dataset)


if __name__=='__main__':
    main()

    """


    init model variables

    psnr, ssim = evaluate(model, val_dir)
    print psnr ssim
    log on tensorboard (psnr ssim)

    while not halt_training(model, stopping_criterion):
        batch = next_batch(train_files)
        model.train(batch)

        if epoch finished (training step * batch num >= num images):
            model.epochs++
            psnr, ssim = evaluate(model, val_dir)
            print psnr ssim
            log on tensorboard (psnr ssim)
            save model?

    save model

    for dir_ in test_dirs:
        psnr ssim = evaluate(model, dir_)
        print psnr ssim?

    return psnr ssim model ....."""