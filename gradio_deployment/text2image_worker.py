import json
import os
import gc

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state
from cosmos_predict2.configs.base.config_text2image import (
    PREDICT2_TEXT2IMAGE_PIPELINE_0P6B,
    PREDICT2_TEXT2IMAGE_PIPELINE_2B,
    PREDICT2_TEXT2IMAGE_PIPELINE_14B,
)
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video, save_text_prompts
from gradio_deployment.model_config import Config

_DEFAULT_POSITIVE_PROMPT = "A well-worn broom sweeps across a dusty wooden floor, its bristles gathering crumbs and flecks of debris in swift, rhythmic strokes. Dust motes dance in the sunbeams filtering through the window, glowing momentarily before settling. The quiet swish of straw brushing wood is interrupted only by the occasional creak of old floorboards. With each pass, the floor grows cleaner, restoring a sense of quiet order to the humble room."


class Text2Image_Validator:

    def validate_params(
        self,
        prompt: str = _DEFAULT_POSITIVE_PROMPT,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        seed: int = 0,
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


class Text2Image_Worker:
    def __init__(
        self,
        num_gpus=1,
        checkpoint_dir="checkpoints/",
        model_size="2B",
        load_ema=False,
    ):
        log.info(f"Using model size: {model_size}")

        if model_size == "0.6B":
            config = PREDICT2_TEXT2IMAGE_PIPELINE_0P6B
            dit_path = f"{checkpoint_dir}/nvidia/Cosmos-Predict2-0.6B-Text2Image/model.pt"
        elif model_size == "2B":
            config = PREDICT2_TEXT2IMAGE_PIPELINE_2B
            dit_path = f"{checkpoint_dir}/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"
        elif model_size == "14B":
            config = PREDICT2_TEXT2IMAGE_PIPELINE_14B
            dit_path = f"{checkpoint_dir}/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt"
        else:
            raise ValueError("Invalid model size. Choose either '0.6B', '2B' or '14B'.")

        log.info(f"Using dit_path: {dit_path}")

        # Only set up text encoder path if no encoder is provided
        text_encoder_path = f"{checkpoint_dir}/google-t5/t5-11b"
        log.info(f"Using text encoder from: {text_encoder_path}")

        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # Floating-point precision settings.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        config.guardrail_config.enabled = False
        config.tokenizer.vae_pth = config.tokenizer.vae_pth.replace("checkpoints/", "")
        config.tokenizer.vae_pth = os.path.join(checkpoint_dir, config.tokenizer.vae_pth)

        # Initialize distributed environment for multi-GPU inference
        if num_gpus > 1:
            log.info(f"Initializing distributed environment with {num_gpus} GPUs for context parallelism")

            # Check if distributed environment is already initialized
            if not parallel_state.is_initialized():
                distributed.init()
                parallel_state.initialize_model_parallel(context_parallel_size=num_gpus)
                log.info(f"Context parallel group initialized with {num_gpus} GPUs")
            else:
                log.info("Distributed environment already initialized, skipping initialization")
                # Check if we need to reinitialize with different context parallel size
                current_cp_size = parallel_state.get_context_parallel_world_size()
                if current_cp_size != num_gpus:
                    log.warning(f"Context parallel size mismatch: current={current_cp_size}, requested={num_gpus}")
                    log.warning("Using existing context parallel configuration")
                else:
                    log.info(f"Using existing context parallel group with {current_cp_size} GPUs")

        # Load models for standalone execution
        log.info(f"Initializing Text2ImagePipeline with model size: {model_size}")
        self.pipe = Text2ImagePipeline.from_config(
            config=config,
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=load_ema,
        )

    def _infer(
        self,
        prompt: str,
        output_dir: str = "outputs/",
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        seed: int = 0,
        use_cuda_graphs: bool = False,
    ):

        log.info(f"Running Text2ImagePipeline\nprompt: {prompt}")

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
        if image is not None:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "output.jpg")

            log.info(f"Saving generated image to: {output_path}")
            save_image_or_video(image, output_path)
            log.success(f"Successfully saved image to: {output_path}")
            # save the prompts used to generate the video
            output_prompt_path = os.path.splitext(output_path)[0] + ".txt"
            prompts_to_save = {"prompt": prompt, "negative_prompt": negative_prompt}
            save_text_prompts(prompts_to_save, output_prompt_path)
            log.success(f"Successfully saved prompt file to: {output_prompt_path}")

    def infer(self, args: dict):
        self._infer(**args)


def create_worker(create_model=True):

    log.info("Creating predict pipeline and validator")
    cfg = Config()
    pipeline = None
    if create_model:
        pipeline = Text2Image_Worker(
            num_gpus=int(os.environ.get("WORLD_SIZE", 1)),
            checkpoint_dir=cfg.checkpoint_dir,
            model_size=cfg.model_size,
            load_ema=cfg.load_ema,
        )
        gc.collect()
        torch.cuda.empty_cache()

    validator = Text2Image_Validator()
    return pipeline, validator
