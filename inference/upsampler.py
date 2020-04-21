import os, shutil
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from PIL import Image
from enum import Enum

CHECKPOINT_DIR = './checkpoints/'
CORE_DIR = lambda ca: CHECKPOINT_DIR + 'core_' + "_".join((ca.size.name, ca.upscale.name, ca.residual.name, ca.activation.name)) + ("_REM" if ca.activation_removal else "") + ("_REC" if ca.recurrent else "") + '/'
CORE_ARCH_PATH = lambda ca: CORE_DIR(ca) + "arch.json"
CORE_ARCH_INFER_PATH = lambda ca: CORE_DIR(ca) + "arch_infer.json"
CORE_CKPT_PATH = lambda ca, i: CORE_DIR(ca) + str(i) + ".h5"

"""
Core model arguments
"""
class Size(Enum):
    SMALL = 1
    MED = 2
    LARGE = 3

class Upscale(Enum):
    DECONV = 1
    SUB_PIXEL = 2

class Residual(Enum):
    NONE = 1
    LOCAL = 2
    GLOBAL = 3

class Activation(Enum):
    RELU = 1
    PRELU = 2

class ModelArgs:
    def __init__(self, epoch, scale, size, upscale, residual, activation, activation_removal, recurrent):
        self.epoch = epoch
        self.scale = scale
        size_dict = {'s':Size.SMALL, 'm':Size.MED, 'l':Size.LARGE}
        upscale_dict = {'de':Upscale.DECONV, 'sp':Upscale.SUB_PIXEL}
        residual_dict = {'n':Residual.NONE, 'l':Residual.LOCAL, 'g':Residual.GLOBAL}
        activation_dict = {'r':Activation.RELU, 'p':Activation.PRELU}
        self.size = size_dict[size]
        self.upscale = upscale_dict[upscale]
        self.residual = residual_dict[residual]
        self.activation = activation_dict[activation]
        self.activation_removal = activation_removal
        self.recurrent = recurrent

"""
Upsampler object
"""
class Upsampler:
    def __init__(self, args):
        self._init_model(args)

    def _init_model(self, args):
        self._args = args
        self._load_model()
        
    def run_dir(self, source_dir, bicubic=False, reset=False):
        """
        Performs inference on directory of frames

        Returns:
            Inference time (int)
        """
        if reset:
            print("Resetting model")
            self._load_model()

        self.source_dir = source_dir
        self.frame_files = [f for f in sorted(os.listdir(source_dir)) if f[-4:] == '.png']
        tf_dataset = tf.data.Dataset.from_generator(self._frame_generator, output_types=(tf.string))#, output_shapes=(tf.TensorShape([])))
        tf_dataset = tf_dataset.map(self._load_frame)
        tf_dataset = tf_dataset.batch(len(self.frame_files))

        if bicubic:
            pred, time = self._execute_bicubic(tf_dataset)
            dest_dir = os.path.join(source_dir, 'bicub')
        else:
            pred, time = self._execute(tf_dataset)
            dest_dir = os.path.join(source_dir, 'up')
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.mkdir(dest_dir)
        for pred_frame, pred_file in zip(pred, self.frame_files):
            im = Image.fromarray(pred_frame)
            im.save(os.path.join(dest_dir, pred_file))
        return time

    def get_model_dir(self):
        return CORE_DIR(self._args)

    def _execute_bicubic(self, dataset):
        for batch in dataset:
            start = tf.timestamp()
            up_size = tf.math.multiply(tf.slice(tf.shape(batch), [1], [2]), 4)
            pred = tf.image.resize(batch, size=up_size, method=tf.image.ResizeMethod.BICUBIC)
            end = tf.timestamp()
            time = tf.math.subtract(end, start)
            pred = tf.image.convert_image_dtype(pred, dtype=tf.uint8, saturate=True)
            return pred.numpy(), time.numpy()


    def _execute(self, dataset):
        start = tf.timestamp()
        pred = self._model.predict(dataset)
        end = tf.timestamp()
        time = tf.math.subtract(end, start)
        pred = tf.image.convert_image_dtype(pred, dtype=tf.uint8, saturate=True)
        return pred.numpy(), time.numpy()

    def _frame_generator(self):
        for frame in self.frame_files:
            yield os.path.join(self.source_dir, frame)

    def _load_frame(self, path):
        frame = tf.image.decode_png(tf.io.read_file(path), channels=3)
        frame = tf.image.convert_image_dtype(frame, dtype=tf.float32)
        return frame

    def _load_model(self):
        arch_path = CORE_ARCH_PATH(self._args)
        weights_path = CORE_CKPT_PATH(self._args, self._args.epoch)
        if not os.path.exists(arch_path):
            raise ValueError(
                "Unable to load model architecture: " + arch_path + "not found")
        elif not os.path.exists(weights_path):
            raise ValueError("Unable to load model weights: " + weights_path + "not found")

        print("Loading model from " + arch_path)
        with open(arch_path, 'r') as f:
            json_str = json.load(f)
            model = tf.keras.models.model_from_json(json_str, custom_objects={'tf': tf})
        print("Restoring model weights from " + weights_path)
        model.load_weights(weights_path)
        self._model = model
