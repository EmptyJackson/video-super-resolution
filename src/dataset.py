import os
import sys
import math
import datasets
import tensorflow as tf

from datasets.div2k import Div2k

class Dataset:
    def __init__(self,
                 dataset="div2k",
                 lr_shape=[480,640,3],
                 scale=2,
                 downscale="bicubic",
                 batch_size=16,
                 prefetch_buffer_size=4):
        
        if dataset == 'div2k':
            self.train_dataset = Div2k(lr_shape, scale, "train", downscale)
            self.val_dataset = Div2k(lr_shape, scale, "valid", downscale)
        elif dataset == 'set5':
            raise NotImplementedError("Set5 support not yet implemented")
        else:
            raise ValueError("Dataset must be Div2k or Set5")
        
        self.scale = scale
        self.lr_shape = lr_shape
        self.batch_size = batch_size
        self.prefetch_buffer_size = prefetch_buffer_size

    def get_num_train_batches(self):
        return math.ceil(self.train_dataset.get_size() / self.batch_size)

    def build_dataset(self, mode='train'):
        if mode=='train':
            dataset = self.train_dataset
        elif mode=='valid':
            dataset = self.val_dataset
        else:
            raise ValueError("Dataset mode must be in ['train', 'valid']")

        tf_dataset = tf.data.Dataset.from_generator(dataset.get_image_pair,
                                                    output_types=(tf.string, tf.string))
        tf_dataset = tf_dataset.shuffle(dataset.get_size())
        tf_dataset = tf_dataset.map(self._load_image)
        tf_dataset = tf_dataset.map(self._preprocess_image)
        tf_dataset = tf_dataset.batch(self.batch_size)
        tf_dataset = tf_dataset.prefetch(self.prefetch_buffer_size)
        #tf_dataset = tf_dataset.cache()
        return tf_dataset

    def _load_image(self, lr_file, hr_file):
        lr_image = tf.image.decode_png(tf.io.read_file(lr_file), channels=self.lr_shape[2])
        hr_image = tf.image.decode_png(tf.io.read_file(hr_file), channels=self.lr_shape[2])
        return lr_image, hr_image

    def _preprocess_image(self, lr_image, hr_image):
        """ Randomly crops low- and high-resolution images to correspond with lr_shape """
        full_lr_shape = tf.shape(lr_image)
        h_diff = full_lr_shape[0] - self.lr_shape[0]
        w_diff = full_lr_shape[1] - self.lr_shape[1]

        if h_diff > 0:  
            lr_up = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=h_diff,
                dtype=tf.int32)
        else:
            lr_up = 0

        if w_diff > 0:    
            lr_left = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=w_diff,
                dtype=tf.int32)
        else:
            lr_left = 0

        scaled_lr_image = tf.slice(lr_image, [lr_up, lr_left, 0],
                                   [self.lr_shape[0], self.lr_shape[1], -1])
        hr_up = lr_up * self.scale
        hr_left = lr_left * self.scale
        hr_shape = [x * self.scale for x in self.lr_shape[:2]]
        scaled_hr_image = tf.slice(hr_image, [hr_up, hr_left, 0],
                                   [hr_shape[0], hr_shape[1], -1])
        
        #scaled_lr_image = tf.cast(scaled_lr_image, tf.float32) / 255.0
        #scaled_hr_image = tf.cast(scaled_hr_image, tf.float32) / 255.0
        scaled_lr_image = tf.image.convert_image_dtype(scaled_lr_image, dtype=tf.float32)
        scaled_hr_image = tf.image.convert_image_dtype(scaled_hr_image, dtype=tf.float32)
        return scaled_lr_image, scaled_hr_image
