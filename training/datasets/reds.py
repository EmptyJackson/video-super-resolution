import os
import random
from PIL import Image

REDS_DIR = "data/reds/"
LR_DIR = lambda s: os.path.join(REDS_DIR, s + "_sharp_bicubic")
HR_DIR = lambda s: os.path.join(REDS_DIR, s + "_sharp")

class Reds:
    def __init__(self, scale=4, subset="train"):
        if not subset in ['train', 'val']:
            raise ValueError('Reds subset must be in [train, val]')

        if not scale==4:
            raise ValueError('Reds scale must be 4')

        self.subset = subset
        self.hr_dir = HR_DIR(subset)
        self.lr_dir = LR_DIR(subset)
        # Must be evenly divisible
        self.vid_length = 100
        self._seq_length = 10

    def get_frame_pair(self):
        for sequence in os.listdir(self.lr_dir):
            if sequence[0] == '.':
                continue
            hr_dir = os.path.join(self.hr_dir, sequence)
            lr_dir = os.path.join(self.lr_dir, sequence)
            # Return 10-frame clips
            for i in range(self.vid_length):
                prefix = "000000"
                if i < 10:
                    prefix += "0"
                lr_path = os.path.join(lr_dir, prefix+str(i)+".png")
                hr_path = os.path.join(hr_dir, prefix+str(i)+".png")
                yield lr_path, hr_path

    def get_size(self):
        """
        Returns number of clips (total_frames/clip_length)
        """
        vids = len([f for f in os.listdir(self.lr_dir) if f[0] != '.'])
        return vids * (self.vid_length / self._seq_length)

    @property
    def seq_length(self):
        return self._seq_length
