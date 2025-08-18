import json
import os
from imaginaire.utils import log
from gradio_deployment.video2world_worker import Video2World_Worker
from gradio_deployment.text2image_worker import Text2Image_Worker
from megatron.core import parallel_state
import torch
import gc
from imaginaire.utils.distributed import barrier, get_rank
from gradio_deployment.model_config import Config

_DEFAULT_POSITIVE_PROMPT = "An autonomous welding robot arm operating inside a modern automotive factory, sparks flying as it welds a car frame with precision under bright overhead lights."
_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


class Text2World_Validator:
    """same as text2image plus guidance"""

    def validate_params(
        self,
        prompt: str = _DEFAULT_POSITIVE_PROMPT,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        aspect_ratio: str = "16:9",
        seed: int = 0,
        guidance: float = 7.0,
        output_dir: str = "outputs/",
    ):
        """replicates cmd line interface examples/video2world.py with same optional parameters and defaults
        Note that we need to set the same defaults here as the underlying pipeline might have it's own different defaults
        """

        args = {}
        args["prompt"] = prompt
        args["negative_prompt"] = negative_prompt
        choices = ["1:1", "4:3", "3:4", "16:9", "9:16"]
        if aspect_ratio not in choices:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}. Must be one of {choices}")
        args["aspect_ratio"] = aspect_ratio
        args["seed"] = seed
        args["guidance"] = guidance
        args["output_dir"] = output_dir
        return args

    def parse_and_validate(self, controlnet_specs):
        args_dict = {}
        for key, value in controlnet_specs.items():
            args_dict[key] = value
        args_dict = self.validate_params(
            **args_dict,
        )

        log.info(json.dumps(args_dict, indent=4))

        return args_dict


class Text2World_Worker:
    def __init__(
        self,
        num_gpus=1,
        checkpoint_dir="checkpoints/",
        model_size="2B",
        resolution="720",
        fps=16,
        load_ema=False,
        disable_prompt_refiner=True,
        offload_prompt_refiner=False,
    ):
        self.text2image_worker = Text2Image_Worker(
            num_gpus=num_gpus, checkpoint_dir=checkpoint_dir, model_size=model_size, load_ema=load_ema
        )

        self.video2world_worker = Video2World_Worker(
            num_gpus=num_gpus,
            checkpoint_dir=checkpoint_dir,
            model_size=model_size,
            resolution=resolution,
            fps=fps,
            load_ema=load_ema,
            disable_prompt_refiner=disable_prompt_refiner,
            offload_prompt_refiner=offload_prompt_refiner,
        )

    def infer(self, args: dict):
        output_path = self.text2image_worker.infer(args)

        is_distributed = parallel_state.is_initialized() and torch.distributed.is_initialized()
        if is_distributed:
            barrier()
        args["input_path"] = output_path
        self.video2world_worker.infer(args)


def create_worker(create_model=True):

    log.info("Creating predict pipeline and validator")
    cfg = Config()
    pipeline = None
    if create_model:
        pipeline = Text2World_Worker(
            num_gpus=int(os.environ.get("WORLD_SIZE", 1)),
            checkpoint_dir=cfg.checkpoint_dir,
            model_size=cfg.model_size,
            load_ema=cfg.load_ema,
        )
        gc.collect()
        torch.cuda.empty_cache()

    validator = Text2World_Validator()
    return pipeline, validator
