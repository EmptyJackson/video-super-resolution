import sys
import numpy as np

from PIL import Image

RESOLUTIONS = { 240: 352, 360: 480, 480: 858, 720: 1280, 1080: 1920 }

def main():
    if len(sys.argv) != 4:
        print("Usage: crop_image.py <input path> <output path> <vertical resolution> [shape]\n")
        exit()

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    res = int(sys.argv[3])
    shape = "standard"
    if len(sys.argv) == 5:
        shape = sys.argv[4]

    if shape == 'standard' and not res in RESOLUTIONS:
        raise ValueError("Resolution must be standard (240, 360, 480, 720, 1080)")

    im = Image.open(in_path)
    w, h = im.size
    hs = (h - res) / 2
    if shape == 'standard':
        ws = (w - RESOLUTIONS[res]) / 2
    elif shape == 'square':
        ws = (w - res) / 2
    else:
        raise ValueError("Shape " + shape + " not recognised.")

    if shape == 'standard':
        cropped_im = im.crop((ws, hs, ws+RESOLUTIONS[res], hs+res))
    elif shape == 'square':
        cropped_im = im.crop((ws, hs, ws+res, hs+res))
    cropped_im.save(out_path)


if __name__ == '__main__':
    main()
