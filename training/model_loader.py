import os
import json
import tensorflow as tf

from models.fsrcnn import fsrcnn
from models.edsr import edsr
from models.core import core_model, CoreArgs
from utils import get_resolution

CHECKPOINT_DIR = './checkpoints/'
MODEL_DIR = lambda model, x, r: CHECKPOINT_DIR + model + '_x' + str(x) + '_' + str(r) + 'p/'
MODEL_ARCH_PATH = lambda model, x, r: MODEL_DIR(model, x, r) + "arch.json"
MODEL_CKPT_PATH = lambda model, x, r, i: MODEL_DIR(model, x, r) + str(i) + ".h5"
CORE_DIR = lambda ca: CHECKPOINT_DIR + 'core_' + "_".join((ca.size.name, ca.upscale.name, ca.residual.name, ca.activation.name)) + ("_REM" if ca.activation_removal else "") + ("_REC" if ca.recurrent else "") + '/'
CORE_ARCH_PATH = lambda ca: CORE_DIR(ca) + "arch.json"
CORE_CKPT_PATH = lambda ca, i: CORE_DIR(ca) + str(i) + ".h5"

def _load_existing_model(arch_path, weights_path):
    if not os.path.exists(arch_path):
        raise ValueError(
            "Unable to load model architecture: " + arch_path + "not found")
    elif not os.path.exists(weights_path):
        raise ValueError("Unable to load model weights: " + weights_path + "not found")

    print("Loading model from " + arch_path)
    with open(arch_path, 'r') as f:
        json_str = json.load(f)
        model = tf.keras.models.model_from_json(json_str)
    print("Restoring model weights from " + weights_path)
    model.load_weights(weights_path)
    return model

def load_model(ckpt_args):
    if ckpt_args.completed:
        if ckpt_args.core_args is None:
            arch_path = MODEL_ARCH_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res)
            weights_path = MODEL_CKPT_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res, ckpt_args.completed)
        else:
            arch_path = CORE_ARCH_PATH(ckpt_args.core_args)
            weights_path = CORE_CKPT_PATH(ckpt_args.core_args, ckpt_args.completed)
        model = _load_existing_model(arch_path, weights_path)
    else:
        print("Initializing new model")
        lr_shape = get_resolution(ckpt_args.res)
        lr_shape.append(3)
        if ckpt_args.model == 'fsrcnn':
            model, lr_mul = fsrcnn(
                in_shape=lr_shape,
                fsrcnn_args=(32, 5, 1),#(48,12,3),  # (d, s, m) #(32, 5, 1) for -s
                scale=ckpt_args.scale
            )
        elif ckpt_args.model == 'edsr':
            model, lr_mul = edsr(
                in_shape=lr_shape,
                scale=ckpt_args.scale,
                num_filters=32, #64
                num_res_blocks=4 #8
            )
        elif ckpt_args.model == 'core':
            if ckpt_args.core_args is None:
                raise ValueError("Must supply core_args to core model")
            model, lr_mul = core_model(
                ckpt_args.core_args
            )
        else:
            raise ValueError("Model '" + ckpt_args.model + "' not supported")
    return model, lr_mul

def load_model_from_dir(dir_path, epoch):
    arch_path = os.path.join(dir_path, 'arch.json')
    weights_path = os.path.join(dir_path, str(epoch)+'.h5')
    return _load_existing_model(arch_path, weights_path)

def save_model_arch(model, ckpt_args):
    if ckpt_args.core_args is None:
        model_dir = MODEL_DIR(ckpt_args.model, ckpt_args.scale, ckpt_args.res)
    else:
        model_dir = CORE_DIR(ckpt_args.core_args)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if ckpt_args.core_args is None:
        arch_path = MODEL_ARCH_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res)
    else:
        arch_path = CORE_ARCH_PATH(ckpt_args.core_args)
    config = model.to_json()
    with open(arch_path, 'w') as f:
        json.dump(config, f)
    print("Model arch saved to " + arch_path)

def save_model_weights(model, epoch, ckpt_args):
    if ckpt_args.core_args is None:
        weights_path = MODEL_CKPT_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res, epoch+ckpt_args.completed)
    else:
        weights_path = CORE_CKPT_PATH(ckpt_args.core_args, epoch+ckpt_args.completed)
    model.save_weights(weights_path)
    print("Model weights saved to " + weights_path)
