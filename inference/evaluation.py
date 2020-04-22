import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image

from upsampler import Upsampler, ModelArgs

VID4_DIR = "data/vid4"
VID4_LENGTH = 167
SAMPLE_ARGS = ModelArgs(
        epoch=5,
        scale=4,
        size='m',
        upscale='sp',
        residual='n',
        activation='r',
        activation_removal=False,
        recurrent=False
    )

class Performance:
    def __init__(self, psnr=None, ssim=None, fps=None):
        self._psnr = psnr
        self._ssim = ssim
        self._fps = fps
    
    @property
    def psnr(self):
        return self._psnr

    @property
    def ssim(self):
        return self._ssim

    @property
    def fps(self):
        return self._fps


def evaluate_model(args, eval_runs, warm_runs, metrics=['psnr', 'ssim', 'fps']):
    """
    Evaluate model on Vid4.
    """
    upsampler = Upsampler(args)
    if warm_runs > 0:
        print("Warming up for evaluation")
        for i in range(warm_runs):
            print("Performing warm-up run", str(i+1))
            for sequence in ['foliage', 'walk', 'calendar', 'city']:
                bix_dir = os.path.join(VID4_DIR, 'BIx4', sequence)
                upsampler.run_dir(bix_dir, reset=False)
    
    time = 0.
    psnrs = []
    ssims = []
    for i in range(eval_runs):
        run_psnrs = []
        run_ssims = []
        print("Performing evaluation run", str(i+1))
        for sequence in ['foliage', 'walk', 'calendar', 'city']:
            bix_dir = os.path.join(VID4_DIR, 'BIx4', sequence)
            gt_dir = os.path.join(VID4_DIR, 'GT', sequence)
            print("Evaluating on", bix_dir)
            time += upsampler.run_dir(bix_dir, reset=False)
            vid_psnrs, vid_ssims = _eval_sr_perf(os.path.join(bix_dir, 'up'), gt_dir)
            run_psnrs += vid_psnrs
            run_ssims += vid_ssims
        if i == eval_runs-1:
            with open(os.path.join(upsampler.get_model_dir(), "psnr.txt"), "w") as f:
                f.writelines(str(psnr) + '\n' for psnr in run_psnrs)
            with open(os.path.join(upsampler.get_model_dir(), "ssim.txt"), "w") as f:
                f.writelines(str(ssim) + '\n' for ssim in run_ssims)
        psnrs += run_psnrs
        ssims += run_ssims

    fps = VID4_LENGTH/ (time/eval_runs)
    return Performance(psnr=psnrs, ssim=ssims, fps=fps)

def _eval_sr_perf(sr_dir, hr_dir):
    psnrs = []
    ssims = []
    for f in os.listdir(sr_dir):
        if f[0] != ".":
            psnr, ssim = _get_sr_metrics(os.path.join(sr_dir, f), os.path.join(hr_dir, f))
            psnrs.append(psnr)
            ssims.append(ssim)
    return psnrs, ssims

def _get_sr_metrics(im1_path, im2_path):
    im1 = tf.image.decode_png(tf.io.read_file(im1_path), channels=3)
    im1 = tf.image.convert_image_dtype(im1, dtype=tf.float32)
    im2 = tf.image.decode_png(tf.io.read_file(im2_path), channels=3)
    im2 = tf.image.convert_image_dtype(im2, dtype=tf.float32)
    return _get_psnr(im1, im2), _get_ssim(im1, im2)

def _get_psnr(im1_src, im2_src):
    im1 = tf.image.rgb_to_yuv(im1_src)
    im1 = im1[:, :, 0]
    im1 = tf.expand_dims(im1, 0)
    im2 = tf.image.rgb_to_yuv(im2_src)
    im2 = im2[:, :, 0]
    im2 = tf.expand_dims(im2, 0)
    return tf.image.psnr(im1, im2, max_val=1.).numpy()

def _get_ssim(im1_src, im2_src):
    im1 = tf.expand_dims(im1_src, 0)
    im2 = tf.expand_dims(im2_src, 0)
    return tf.image.ssim(im1, im2, max_val=1.).numpy()

def evaluate_bicubic(metrics=['psnr', 'ssim', 'fps']):
    """
    Evaluates performance of baseline VSR model (Bicubic interpolation) on Vid4.

    All videos in lr_dir must contain an upsampled video in hr_dir with a matching filename.
    """
    time = 0.
    psnrs = []
    ssims = []
    upsampler = Upsampler(SAMPLE_ARGS)
    for sequence in ['foliage', 'walk', 'calendar', 'city']:
        bix_dir = os.path.join(VID4_DIR, 'BIx4', sequence)
        gt_dir = os.path.join(VID4_DIR, 'GT', sequence)
        print("Evaluating bicubic on", bix_dir)
        time += upsampler.run_dir(bix_dir, bicubic=True, reset=False)
        vid_psnrs, vid_ssims = _eval_sr_perf(os.path.join(bix_dir, 'bicub'), gt_dir)
        psnrs += vid_psnrs
        ssims += vid_ssims
    fps = VID4_LENGTH/time
    return Performance(psnr=psnrs, ssim=ssims, fps=fps)


if __name__=='__main__':
    parser=argparse.ArgumentParser(
        description="Model evaluation\n" +\
                    "Usage: evaluation.py [options]"
    )
    parser.add_argument('--bicubic', action='store_true', help="evaluate bicubic interpolation")
    parser.add_argument('--warm_runs', default=0, type=int, help="number of warm up runs")
    parser.add_argument('--eval_runs', default=1, type=int, help="number of evaluation runs")
    parser.add_argument('--scale', default=2, type=int, help="factor by which to upscale the given model")
    parser.add_argument('--epoch', default=100, type=int, help="checkpoint epoch")
    parser.add_argument('--size', default='s', type=str, choices=['s', 'm', 'l'], help="size of core model")
    parser.add_argument('--upscale', default='sp', type=str, choices=['de', 'sp'], help="upscale method of core model")
    parser.add_argument('--residual', default='n', type=str, choices=['n', 'l', 'g'], help="residual method of core model")
    parser.add_argument('--activation', default='r', type=str, choices=['r', 'p'], help="activation of core model")
    parser.add_argument('--activation_removal', action='store_true', help="activation removal in core model")
    parser.add_argument('--recurrent', action='store_true', help="recurrent core model")

    args = parser.parse_args()

    if args.bicubic:
        perf = evaluate_bicubic()
    else:
        model_args = ModelArgs(
            epoch=args.epoch,
            scale=args.scale,
            size=args.size,
            upscale=args.upscale,
            residual=args.residual,
            activation=args.activation,
            activation_removal=args.activation_removal,
            recurrent=args.recurrent
        )
        perf = evaluate_model(model_args, args.eval_runs, args.warm_runs)

    print('psnr:', np.mean(perf.psnr), "+-", np.std(perf.psnr))
    print('ssim:', np.mean(perf.ssim), "+-", np.std(perf.ssim))
    print("fps:", perf.fps)
