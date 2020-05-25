from upsampler import ModelArgs, Upsampler
from evaluation import evaluate_model

class NotConfiguredError(Exception):
    def __init__(self, message):
        self.message = message


"""
Core model ranking
"""
MODEL_RANKING = [
    ModelArgs(
        epoch=300,
        scale=4,
        size='m',
        upscale='sp',
        residual='l',
        activation='r',
        activation_removal=True,
        recurrent=True
    ),
    ModelArgs(
        epoch=300,
        scale=4,
        size='m',
        upscale='sp',
        residual='l',
        activation='r',
        activation_removal=True,
        recurrent=False
    ),
    ModelArgs(
        epoch=200,
        scale=4,
        size='s',
        upscale='sp',
        residual='l',
        activation='r',
        activation_removal=True,
        recurrent=True
    )
]


"""
Vsr object
"""
class Vsr:
    def __init__(self):
        self.configured = False

    def configure(self, resolution, min_fps):
        """
        Evaluate model performance and select model with desired fps

        Parameters:
            fps (int): Minimum frames per second for executing model
        """

        model_found = False
        for args in MODEL_RANKING:
            perf = evaluate_model(args, resolution)
            if perf.fps < min_fps:
                self.upsampler = Upsampler(args)
                model_found = True
                break

        self.configured = False
        return model_found

    def execute_video(self, source_dir, clear_processed=False):
        """
        Performs inference on a given directory of frames

        Parameters:
            source_dir: Source directory of LR frames, in style '00000001.png, ...'

        Optional parameters:
            clear_processed (bool): Delete LR frames after processing
        """

        if not self.configured:
            raise NotConfiguredError("VSR object unconfigured or model not found")

        self.upsampler.run_dir(source_dir)
