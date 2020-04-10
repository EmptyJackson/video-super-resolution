import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from PIL import Image

from utils import load_image, save_image_from_tensor
from model_loader import load_model_from_dir

@tf.function
def get_prediction(model, lr_image):
    lr_image = tf.expand_dims(lr_image, 0)
    hr_pred = model(lr_image)
    return tf.clip_by_value(hr_pred, 0., 1.)

def main():
    nargs = len(sys.argv)
    if nargs < 5 or (sys.argv[3] == 'bicubic' and nargs != 5) or (sys.argv[3] == 'network' and nargs != 6):
        print("Usage: upscale_image.py <input path> <output path> <method> [<scale> | <model dir> <checkpoint epoch>] \n")
        exit()

    image_path_in = sys.argv[1]
    image_path_out = sys.argv[2]
    method = sys.argv[3]

    if method == 'bicubic':
        scale = int(sys.argv[4])
        im = Image.open(image_path_in)
        hr_image = im.resize((im.width*scale, im.height*scale), Image.BICUBIC)
        hr_image.save(image_path_out)
    elif method == 'network':
        lr_image = load_image(image_path_in)
        model_dir = sys.argv[4]
        epoch = sys.argv[5]
        model, lr_mul = load_model_from_dir(model_dir, epoch)
        hr_image = get_prediction(model, lr_image)
        save_image_from_tensor(hr_image, image_path_out)
    else:
        raise ValueError('Upscale method must be in [network, bicubic]')

if __name__=='__main__':
    main()
