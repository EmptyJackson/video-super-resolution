import sys
import numpy as np

from PIL import Image

RESOLUTIONS = { 240: 352, 360: 480, 480: 858, 720: 1280, 1080: 1920 }

def main():
    if len(sys.argv) != 4:
        print("Usage: crop_image.py <input path> <output path> <vertical resolution>\n")
        exit()

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    res = int(sys.argv[3])

    if not res in RESOLUTIONS:
        raise ValueError("Resolution must be standard (240, 360, 480, 720, 1080)")

    im = Image.open(in_path)
    w, h = im.size
    hs = (h - res) / 2
    ws = (w - RESOLUTIONS[res]) / 2

    cropped_im = im.crop((ws, hs, ws+RESOLUTIONS[res], hs+res))
    cropped_im.save(out_path)


if __name__ == '__main__':
    main()
