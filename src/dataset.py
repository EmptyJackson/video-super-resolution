import sys
import utils
import datasets
import tensorflow as tf
from datasets.div2k import Div2k

class Dataset:
    def __init__(self,
                 dataset="div2k",
                 lr_shape=[480,640,3],
                 scale=2,
                 subset="train",
                 downscale="bicubic",
                 batch_size=16,
                 prefetch_buffer_size=4):
        
        if dataset == 'div2k':
            self.dataset = Div2k(scale, subset, downscale)
        elif dataset == 'set5':
            raise NotImplementedError("Set5 support not yet implemented")
        else:
            raise ValueError("Dataset must be Div2k or Set5")
        
        self.scale = scale
        self.lr_shape = lr_shape
        self.batch_size = batch_size
        self.prefetch_buffer_size = prefetch_buffer_size

    def build_tf_dataset(self):
        tf_dataset = tf.data.Dataset.from_generator(self.dataset.get_image_pair,
                                                    output_types=(tf.string, tf.string))
        tf_dataset = tf_dataset.map(self._load_image)
        tf_dataset = tf_dataset.map(self._preprocess_image)
        tf_dataset = tf_dataset.batch(self.batch_size)
        tf_dataset = tf_dataset.prefetch(self.prefetch_buffer_size)
        tf_dataset = tf_dataset.cache()
        return tf_dataset

    def _load_image(self, lr_file, hr_file):
        lr_image = tf.image.decode_png(tf.io.read_file(lr_file), channels=self.lr_shape[2])
        hr_image = tf.image.decode_png(tf.io.read_file(hr_file), channels=self.lr_shape[2])
        return lr_image, hr_image

    def _preprocess_image(self, lr_image, hr_image):
        """ Randomly crops low- and high-resolution images to correspond with lr_shape """
        full_lr_shape = tf.shape(lr_image)
        tf.print(lr_image)#, output_stream=sys.stdout)
        lr_up = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=full_lr_shape[0] - self.lr_shape[0],
            dtype=tf.int32)
        lr_left = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=full_lr_shape[1] - self.lr_shape[1],
            dtype=tf.int32)
        scaled_lr_image = tf.slice(lr_image, [lr_up, lr_left, 0],
                                   [self.lr_shape[0], self.lr_shape[1], -1])
        hr_up = lr_up * self.scale
        hr_left = lr_left * self.scale
        hr_shape = self.lr_shape * self.scale
        hr_shape[2] = self.lr_shape[2]        # Don't scale channel dimension
        scaled_hr_image = tf.slice(hr_image, [hr_up, hr_left, 0],
                                   [hr_shape[0], hr_shape[1], -1])
        return scaled_lr_image, scaled_hr_image
