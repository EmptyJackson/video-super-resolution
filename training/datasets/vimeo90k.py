import os
import random
from PIL import Image

DIV2K_DIR = "data/vimeo-90k-t/"
LR_DIR = os.path.join(DIV2K_DIR, "low_resolution")
HR_DIR = os.path.join(DIV2K_DIR, "target")

class Vimeo90k:
    def __init__(self, scale=4, subset="train"):
        if not subset=='train':
            raise ValueError('Vimeo90k subset must be train')

        if not scale==4:
            raise ValueError('Vimeo90k scale must be 4')

        self.scale = scale
        self.subset = subset
        self.hr_dir = HR_DIR(subset)
        self.lr_dir = LR_DIR(subset, scale)
        self.seq_length = 7

    def get_video_pair(self):
        for container in os.listdir(self.lr_dir):
            for sequence in os.listdir(os.path.join(self.lr_dir, container)):
                hr_path = os.path.join(self.hr_dir, container, sequence)
                lr_path = os.path.join(self.lr_dir, container, sequence)
                lrs = []
                hrs = []
                for i in range(1, self.seq_length+1):
                    lrs.append(os.path.join(lr_path, "im"+str(i)+".png"))
                    hrs.append(os.path.join(hr_path, "im"+str(i)+".png"))
                yield lrs, hrs

    def get_size(self):
        return len(self.image_ids)
