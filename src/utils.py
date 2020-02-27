import tensorflow as tf

RESOLUTIONS = { 240:352, 360:480, 480:858, 720:1280, 1080:1920 }

"""
Returns image resolution from a standard height.

i.e. 240p -> 352x240
"""
def get_resolution(height):
    if height in RESOLUTIONS:
        return [height, RESOLUTIONS[height]]
    raise ValueError("Resolution must be standard (240, 360, 480, 720, 1080)")

def load_image(image_file):
    lr_image = tf.image.decode_png(tf.io.read_file(image_file), channels=3)
    return tf.image.convert_image_dtype(lr_image, dtype=tf.float32)

@tf.function
def save_image_from_tensor(tensor, path):
    int_tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8)
    int_tensor = tf.squeeze(int_tensor)
    image = tf.image.encode_png(int_tensor)
    tf.io.write_file(path, image)
