import os
import sys
import math
import datasets
import tensorflow as tf

from datasets.div2k import Div2k
from datasets.vid4 import Vid4
from datasets.vimeo90k import Vimeo90k

class DataLoader:
    def __init__(self,
                 dataset="div2k",
                 scale=2,
                 batch_size=16,
                 prefetch_buffer_size=4):
        
        self.name = dataset
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.scale = scale
        self.batch_size = batch_size
        self.prefetch_buffer_size = prefetch_buffer_size

    def get_num_train_batches(self):
        return math.ceil(self.train_dataset.get_size() / self.batch_size)

    def _get_partition(self, mode):
        mode_to_set = {
            'train': self.train_dataset,
            'valid': self.val_dataset,
            'test': self.test_dataset
        }
        if mode in mode_to_set:
            dataset = mode_to_set[mode]
            if dataset is None:
                raise ValueError(
                    self.name + " does not contain a " + mode + " partition")
            return dataset
        else:
            raise ValueError(
                "Dataset mode must be in " + str(list(mode_to_set.keys())))


class ImageLoader(DataLoader):
    def __init__(self,
                 dataset="div2k",
                 lr_shape=[480,640,3],
                 scale=2,
                 batch_size=16,
                 prefetch_buffer_size=4):
        
        if dataset == 'div2k':
            self.train_dataset = Div2k(lr_shape, scale, "train")
            self.val_dataset = Div2k(lr_shape, scale, "valid")
        elif dataset == 'set5':
            raise NotImplementedError("Set5 object net yet implemented")
        else:
            raise ValueError("Video dataset must be in [div2k, set5]")
        self.lr_shape = lr_shape
        DataLoader.__init__(self, dataset, lr_shape, scale, batch_size, prefetch_buffer_size)

    def build_dataset(self, mode='train'):
        dataset = self._get_partition(mode)
        tf_dataset = tf.data.Dataset.from_generator(dataset.get_image_pair,
                                                    output_types=(tf.string, tf.string))
        if mode == 'train':
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


class VideoLoader(DataLoader):
    def __init__(self,
                 dataset="vimeo90k",
                 lr_shape=[480,640,3],
                 scale=2,
                 batch_size=16,
                 prefetch_buffer_size=4):
        
        if dataset == 'vimeo90k':
            self.train_dataset = Vimeo90k(lr_shape, scale, "train")
            self.val_dataset = Vimeo90k(lr_shape, scale, "valid")
        elif dataset == 'vid4':
            self.test_dataset = Vid4(lr_shape)
        else:
            raise ValueError("Video dataset must be in [vimeo90k, vid4]")
        DataLoader.__init__(self, dataset, lr_shape, scale, batch_size, prefetch_buffer_size)

    def build_dataset(self, mode='train'):
        dataset = self._get_partition(mode)
        tf_dataset = tf.data.Dataset.from_generator(dataset.get_video,
                                                    output_types=(tf.string, tf.string),
                                                    output_shapes=(tf.TensorShape([None]), tf.TensorShape([None])))
        tf_dataset = tf_dataset.map(self._load_video)
        #tf_dataset = tf_dataset.map(self._preprocess_image)
        #tf_dataset = tf_dataset.batch(self.batch_size)
        #tf_dataset = tf_dataset.prefetch(self.prefetch_buffer_size)
        #tf_dataset = tf_dataset.cache()
        return tf_dataset

    def _load_video(self, lr_paths, hr_paths):
        lr_vid = []
        hr_vid = []
        for lr_path, hr_path in zip(lr_paths, hr_paths):
            lr_vid.append(tf.image.decode_png(tf.io.read_file(lr_path), channels=3))
            hr_vid.append(tf.image.decode_png(tf.io.read_file(hr_path), channels=3))
        return tf.stack(lr_vid), tf.stack(hr_vid)



class Dataset:
    pass #TBCTBCTBCTBCTBCTBCTBCTBCTBCTBCTBCTBC
