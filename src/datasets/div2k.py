import os
import random

DIV2K_DIR = "data/div2k/"
LR_DIR = lambda s, d, scl: DIV2K_DIR + "/DIV2K_" + s + "_LR_" + d + "/X" + str(scl) + "/"
HR_DIR = lambda s: DIV2K_DIR + "DIV2K_" + s + "_HR/"

class Div2k:
    def __init__(self, scale=2, subset="train", downscale="bicubic"):
        if not subset in ['train', 'valid', 'test']:
            raise ValueError('Div2k subset must be train, valid or test')
        if not downscale in ['bicubic', 'unknown']:
            raise ValueError('Div2k downscale method must be bicubic or unknown')

        self.scale = scale
        self.subset = subset
        self.hr_dir = HR_DIR(subset)
        self.lr_dir = LR_DIR(subset, downscale, scale)

        # Extract Div2k image ids for selection later
        self.image_ids = [filename[:4] for filename in os.listdir(self.hr_dir)]
    
    def get_image_pair(self):
        while True:
            image_id = random.choice(self.image_ids)
            hr_path = self.hr_dir + image_id + ".png"
            lr_path = self.lr_dir + image_id + "x" + str(self.scale) + ".png"
            yield lr_path, hr_path
