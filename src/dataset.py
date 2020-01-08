import datasets
import tensorflow as tf

class Dataset:
    def __init__(self, 
                 dataset="div2k",
                 scale=2,
                 subset="train",
                 downscale="bicubic",
                 channels=3,
                 batch_size=16,
                 prefetch_buffer_size=4):
        
        if dataset == 'div2k':
            self.dataset_gen = Div2k(scale, subset, downscale).get_image_pair()
        elif dataset == 'set5':
            raise NotImplementedError("Set5 support not yet implemented")
        else:
            raise ValueError("Dataset must be Div2k or Set5")
        
        self.batch_size = batch_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.channels = channels

    def extract_image_dataset(self, target_size):
        tf_dataset = tf.data.Dataset.from_generator(self.dataset_gen.get_image_pair,
                                                    output_types=(tf.string, tf.string))
        tf_dataset = tf_dataset.map(self._load_image)
        tf_dataset = tf_dataset.batch(self.batch_size)
        tf_dataset = tf_dataset.prefetch(self.prefetch_buffer_size)
        # tf_dataset = tf_dataset.cache()
        return tf_dataset

    def _load_image(self, lr_file, hr_file):
        lr_image = tf.image.decode_png(tf.read_file(lr_file), channels=self.channels)
        hr_image = tf.image.decode_png(tf.read_file(hr_file), channels=self.channels)
        return lr_image, hr_image
