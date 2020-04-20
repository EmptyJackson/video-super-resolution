import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from PIL import Image

from upsampler import Upsampler, ModelArgs

VID4_DIR = "data/vid4"
VID4_LENGTH = 167
SAMPLE_ARGS = ModelArgs(
        epoch=13,
        scale=4,
        size='s',
        upscale='sp',
        residual='n',
        activation='r',
        activation_removal=False,
        recurrent=True
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


def evaluate_model(args, metrics=['psnr', 'ssim', 'fps']):
    """
    Evaluate model on Vid4.
    """
    warm_runs = 0
    perf_runs = 1
    upsampler = Upsampler(args)
    if warm_runs > 0:
        print("Warming up for evaluation")
        for i in range(warm_runs):
            print("Performing warm-up run", str(i+1))
            for sequence in ['foliage', 'walk', 'calendar', 'city']:
                bix_dir = os.path.join(VID4_DIR, 'BIx4', sequence)
                upsampler.run_dir(bix_dir, reset=False)
    
    time = 0.
    psnr = 0.
    for i in range(perf_runs):
        print("Performing evaluation run", str(i+1))
        for sequence in ['foliage', 'walk', 'calendar', 'city']:
            bix_dir = os.path.join(VID4_DIR, 'BIx4', sequence)
            gt_dir = os.path.join(VID4_DIR, 'GT', sequence)
            print("Evaluating on", bix_dir)
            time += upsampler.run_dir(bix_dir, reset=False)
            psnr += _eval_sr_perf(os.path.join(bix_dir, 'up'), gt_dir)
    fps = VID4_LENGTH/ (time/perf_runs)
    psnr /= (4 * perf_runs)
    return Performance(psnr=psnr, fps=fps)

def _eval_sr_perf(sr_dir, hr_dir, metrics=['psnr', 'ssim']):
    psnr = 0.
    i = 0
    for f in os.listdir(sr_dir):
        if f[0] != ".":
            psnr += _get_psnr(os.path.join(sr_dir, f), os.path.join(hr_dir, f))
            i += 1
    psnr /= i
    return psnr

def _get_psnr(im1_path, im2_path):
    im1 = tf.image.decode_png(tf.io.read_file(im1_path), channels=3)
    im1 = tf.image.convert_image_dtype(im1, dtype=tf.float32)
    im1 = tf.image.rgb_to_yuv(im1)
    im1 = im1[:, :, 0]
    im1 = tf.expand_dims(im1, 0)
    im2 = tf.image.decode_png(tf.io.read_file(im2_path), channels=3)
    im2 = tf.image.convert_image_dtype(im2, dtype=tf.float32)
    im2 = tf.image.rgb_to_yuv(im2)
    im2 = im2[:, :, 0]
    im2 = tf.expand_dims(im2, 0)
    return tf.image.psnr(im1, im2, max_val=1.).numpy()

def evaluate_bicubic(metrics=['psnr', 'ssim', 'fps']):
    """
    Evaluates performance of baseline VSR model (Bicubic interpolation) on Vid4.

    All videos in lr_dir must contain an upsampled video in hr_dir with a matching filename.
    """
    time = 0.
    psnr = 0.
    upsampler = Upsampler(SAMPLE_ARGS)
    for sequence in ['foliage', 'walk', 'calendar', 'city']:
        bix_dir = os.path.join(VID4_DIR, 'BIx4', sequence)
        gt_dir = os.path.join(VID4_DIR, 'GT', sequence)
        print("Evaluating bicubic on", bix_dir)
        time += upsampler.run_dir(bix_dir, bicubic=True, reset=False)
        psnr += _eval_sr_perf(os.path.join(bix_dir, 'bicub'), gt_dir)
    fps = VID4_LENGTH/time
    psnr /= 4
    return Performance(psnr=psnr, fps=fps)


if __name__=='__main__':
    args = ModelArgs(
        epoch=315,
        scale=4,
        size='m',
        upscale='sp',
        residual='g',
        activation='r',
        activation_removal=False,
        recurrent=False
    )
    #perf = evaluate_bicubic()
    perf = evaluate_model(args)
    print('psnr:', perf.psnr)
    print('fps:', perf.fps)
