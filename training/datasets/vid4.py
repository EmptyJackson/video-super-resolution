import os
import random
from PIL import Image

VID4_DIR = "data/vid4/"
LR_DIR = os.path.join(VID4_DIR, "Blx4")
HR_DIR = os.path.join(VID4_DIR, "GT")

class Reds:
    def __init__(self, scale=4, subset="test"):
        if not subset in ['test']:
            raise ValueError('Vid4 subset must be in test')
        if not scale==4:
            raise ValueError('Vid4 scale must be 4')
        self._seq_length = 41

    def get_frame_pair(self):
        # City must be last due to variable length sequences
        for sequence in ['foliage', 'walk', 'calendar', 'city']:
            if sequence[0] == '.':
                continue
            hr_seq_dir = os.path.join(self.hr_dir, sequence)
            lr_seq_dir = os.path.join(self.lr_dir, sequence)
            for i in range(self._seq_length):
                prefix = "000000"
                if i < 10:
                    prefix += "0"
                lr_path = os.path.join(lr_seq_dir, prefix+str(i)+".png")
                hr_path = os.path.join(hr_seq_dir, prefix+str(i)+".png")
                if os.path.exists(lr_path):
                    yield lr_path, hr_path

    def get_size(self):
        """
        Returns number of videos
        """
        return 4

    @property
    def seq_length(self):
        return self._seq_length
