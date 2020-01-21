import os
import json
import tensorflow as tf

from models.fsrcnn import fsrcnn
from utils import get_resolution

CHECKPOINT_DIR = './checkpoints/'
MODEL_DIR = lambda model, x, r: CHECKPOINT_DIR + model + '_x' + str(x) + '_' + str(r) + 'p/'
MODEL_ARCH_PATH = lambda model, x, r: MODEL_DIR(model, x, r) + "arch.json"
MODEL_CKPT_PATH = lambda model, x, r, i: MODEL_DIR(model, x, r) + str(i) + ".h5"

def load_model(ckpt_args):
    if ckpt_args.completed:
        arch_path = MODEL_ARCH_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res)
        print("Loading model from " + arch_path)
        with open(arch_path, 'r') as f:
            json_str = json.load(f)
            model = tf.keras.models.model_from_json(json_str)
        weights_path = MODEL_CKPT_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res, ckpt_args.completed)
        print("Restoring model weights from " + weights_path)
        model.load_weights(weights_path)
    else:
        print("Initializing new model")
        lr_shape = get_resolution(ckpt_args.res)
        lr_shape.append(3)
        if ckpt_args.model == 'fsrcnn':
            model = fsrcnn(
                in_shape=lr_shape,
                fsrcnn_args=(48, 12, 2),#(48,12,3),  # (d, s, m)
                scale=ckpt_args.scale
            )
        else:
            raise ValueError("Model '" + ckpt_args.model + "' not supported")

    return model

def save_model_arch(model, ckpt_args):
    model_dir = MODEL_DIR(ckpt_args.model, ckpt_args.scale, ckpt_args.res)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    arch_path = MODEL_ARCH_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res)
    config = model.to_json()
    with open(arch_path, 'w') as f:
        json.dump(config, f)
    print("Model arch saved to " + arch_path)

def save_model_weights(model, epoch, ckpt_args):
    weights_path = MODEL_CKPT_PATH(ckpt_args.model, ckpt_args.scale, ckpt_args.res, epoch+ckpt_args.completed)
    model.save_weights(weights_path)
    print("Model weights saved to " + weights_path)
