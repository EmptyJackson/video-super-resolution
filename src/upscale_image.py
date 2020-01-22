import sys
import tensorflow as tf

from utils import load_image, save_image_from_tensor
from models.model_io import load_model_from_dir

@tf.function
def get_prediction(model, lr_image):
    lr_image = tf.expand_dims(lr_image, 0)
    return model(lr_image)

def main():
    if len(sys.argv) != 5:
        print("Usage: upscale_image.py <model dir> <checkpoint epoch> <input path> <output path>\n")
        exit()

    model_dir = sys.argv[1]
    epoch = sys.argv[2]
    image_path_in = sys.argv[3]
    image_path_out = sys.argv[4]

    model = load_model_from_dir(model_dir, epoch)
    lr_image = load_image(image_path_in)
    hr_image = get_prediction(model, lr_image)

    save_image_from_tensor(hr_image, image_path_out)

if __name__=='__main__':
    main()
