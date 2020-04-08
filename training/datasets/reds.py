import os
import random
from PIL import Image

DIV2K_DIR = "data/reds/"
LR_DIR = lambda s: os.path.join(DIV2K_DIR, s + "_sharp_bicubic")
HR_DIR = lambda s: os.path.join(DIV2K_DIR, s + "_sharp")

class Reds:
    def __init__(self, scale=4, subset="train"):
        if not subset=='train':
            raise ValueError('Reds subset must be in [train, val]')

        if not scale==4:
            raise ValueError('Reds scale must be 4')

        self.subset = subset
        self.hr_dir = HR_DIR(subset)
        self.lr_dir = LR_DIR(subset)
        self.vid_length = 100
        self.seq_length = 10

    def get_video_pair(self):
        for sequence in os.listdir(self.lr_dir):
            hr_path = os.path.join(self.hr_dir, sequence)
            lr_path = os.path.join(self.lr_dir, sequence)
            lrs = []
            hrs = []
            # Return 10-frame videos
            for i in range(self.vid_length/10):
                for _id in range(i*self.seq_length, (i+1)*self.seq_length):
                    prefix = "000000"
                    if i < 10:
                        prefix += "0"
                    lrs.append(os.path.join(lr_path, prefix+str(i)+".png"))
                    hrs.append(os.path.join(hr_path, prefix+str(i)+".png"))
            yield lrs, hrs

    def get_size(self):
        """
        Returns number of clips (total_frames/clip_length)
        """
        vids = len(os.listdir(self.lr_dir))
        return vids * (self.vid_length / self.seq_length)
