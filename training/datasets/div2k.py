import os
import random
from PIL import Image

DIV2K_DIR = "data/div2k/"
LR_DIR = lambda s, d, scl: os.path.join(DIV2K_DIR, "DIV2K_" + s + "_LR_" + d, "X" + scl)
HR_DIR = lambda s: os.path.join(DIV2K_DIR, "DIV2K_" + s + "_HR")

class Div2k:
    def __init__(self, lr_shape, scale=2, subset="train", downscale="bicubic"):
        if not subset in ['train', 'valid', 'test']:
            raise ValueError('Div2k subset must be train, valid or test')
        if not downscale in ['bicubic', 'unknown']:
            raise ValueError('Div2k downscale method must be bicubic or unknown')

        self.scale = str(scale)
        self.subset = subset
        self.hr_dir = HR_DIR(subset)
        self.lr_dir = LR_DIR(subset, downscale, self.scale)

        # Extract sufficiently sized Div2k image ids for selection later
        self.image_ids = [filename[:4] for filename in os.listdir(self.hr_dir)]
        self.image_ids = [
            image_id for image_id in self.image_ids if self._is_minimum_size(image_id, lr_shape)]

    def _is_minimum_size(self, image_id, lr_shape):
        path = os.path.join(self.lr_dir, image_id + "x" + self.scale + ".png")
        if not os.path.exists(path):
            print("Missing Div2k file:", path)
            return False
        im = Image.open(path)
        width, height = im.size
        return height >= lr_shape[0] and width >= lr_shape[1]

    def get_image_pair(self):
        random.shuffle(self.image_ids)
        for image_id in self.image_ids:
            hr_path = os.path.join(self.hr_dir, image_id + ".png")
            lr_path = os.path.join(self.lr_dir, image_id + "x" + self.scale + ".png")
            if not os.path.exists(hr_path):
                print("Missing Div2k file: ", hr_path)
                continue
            if not os.path.exists(lr_path):
                print("Missing Div2k file: ", lr_path)
                continue
            yield lr_path, hr_path

    def get_size(self):
        return len(self.image_ids)
