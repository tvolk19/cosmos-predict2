# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import gc

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state

from cosmos_predict2.configs.base.config_video2world import (
    _PREDICT2_VIDEO2WORLD_PIPELINE_2B,
    _PREDICT2_VIDEO2WORLD_PIPELINE_14B,
)
from cosmos_predict2.pipelines.video2world import _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS, Video2WorldPipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video, save_text_prompts
from gradio_deployment.model_config import Config

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def validate_input_file(input_path: str, num_conditional_frames: int) -> bool:
    if not os.path.exists(input_path):
        log.warning(f"Input file does not exist, skipping: {input_path}")
        return False

    ext = os.path.splitext(input_path)[1].lower()

    if num_conditional_frames == 1:
        # Single frame conditioning: accept both images and videos
        if ext not in _IMAGE_EXTENSIONS and ext not in _VIDEO_EXTENSIONS:
            log.warning(
                f"Skipping file with unsupported extension for single frame conditioning: {input_path} "
                f"(expected: {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS})"
            )
            return False
    elif num_conditional_frames == 5:
        # Multi-frame conditioning: only accept videos
        if ext not in _VIDEO_EXTENSIONS:
            log.warning(
                f"Skipping file for multi-frame conditioning (requires video): {input_path} "
                f"(expected: {_VIDEO_EXTENSIONS}, got: {ext})"
            )
            return False
    else:
        log.error(f"Invalid num_conditional_frames: {num_conditional_frames} (must be 1 or 5)")
        return False

    return True


class Video2World_Validator:
    def validate_params(
        self,
        input_path: str = "assets/video2world/input0.jpg",
        prompt: str = "",
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        aspect_ratio: str = "16:9",
        num_conditional_frames: int = 1,
        guidance: float = 7.0,
        seed: int = 0,
        output_dir: str = "outputs/",
    ):
        """replicates cmd line interface examples/video2world.py with same optional parameters and defaults
        Note that we need to set the same defaults here as the underlying pipeline might have it's own different defaults
        """
        if not validate_input_file(input_path, num_conditional_frames):
            log.warning(f"Input file validation failed: {input_path}")
            raise ValueError(f"Invalid input file: {input_path}")

        args = {}
        args["input_path"] = input_path
        args["prompt"] = prompt
        args["negative_prompt"] = negative_prompt
        choices = ["1:1", "4:3", "3:4", "16:9", "9:16"]
        if aspect_ratio not in choices:
            raise ValueError(f"Invalid aspect ratio: {aspect_ratio}. Must be one of {choices}")
        args["aspect_ratio"] = aspect_ratio
        choices = [1, 5]
        if num_conditional_frames not in choices:
            raise ValueError(f"Invalid num_conditional_frames: {num_conditional_frames}. Must be one of {choices}")
        args["num_conditional_frames"] = num_conditional_frames
        args["guidance"] = guidance
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


class Video2World_Worker:
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
        log.info(f"Using model size: {model_size}")

        if model_size == "2B":
            config = _PREDICT2_VIDEO2WORLD_PIPELINE_2B

            config.resolution = resolution
            if fps == 10:  # default is 16 so no need to change config
                config.state_t = 16

            dit_path = f"{checkpoint_dir}/nvidia/Cosmos-Predict2-2B-Video2World/model-{resolution}p-{fps}fps.pt"
        elif model_size == "14B":
            config = _PREDICT2_VIDEO2WORLD_PIPELINE_14B

            config.resolution = resolution
            if fps == 10:  # default is 16 so no need to change config
                config.state_t = 16

            dit_path = f"{checkpoint_dir}/nvidia/Cosmos-Predict2-14B-Video2World/model-{resolution}p-{fps}fps.pt"
        else:
            raise ValueError("Invalid model size. Choose either '2B' or '14B'.")

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

        # Disable prompt refiner if requested
        if disable_prompt_refiner:
            log.warning("Prompt refiner is disabled")
            config.prompt_refiner_config.enabled = False
        config.prompt_refiner_config.offload_model_to_cpu = offload_prompt_refiner
        config.guardrail_config.enabled = False
        config.tokenizer.vae_pth = config.tokenizer.vae_pth.replace("checkpoints/", "")
        config.tokenizer.vae_pth = os.path.join(checkpoint_dir, config.tokenizer.vae_pth)

        # Load models
        log.info(f"Initializing Video2WorldPipeline with model size: {model_size}")
        self.pipe = Video2WorldPipeline.from_config(
            config=config,
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=load_ema,
            load_prompt_refiner=config.prompt_refiner_config.enabled,
        )

    def _infer(
        self,
        input_path: str,
        prompt: str,
        output_dir: str = "outputs/",
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        num_conditional_frames: int = 1,
        guidance: float = 7.0,
        seed: int = 0,
        use_cuda_graphs: bool = False,
    ):

        log.info(f"Running Video2WorldPipeline\ninput: {input_path}\nprompt: {prompt}")

        # misc.set_random_seed(seed=seed, by_rank=True)
        video, prompt_used = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            input_path=input_path,
            num_conditional_frames=num_conditional_frames,
            guidance=guidance,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
            return_prompt=True,
        )

        if video is not None:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, "output.mp4")
            log.info(f"Saving the generated video to: {output_path}")
            if self.pipe.config.state_t == 16:
                fps = 10
            else:
                fps = 16
            save_image_or_video(video, output_path, fps=fps)
            log.success(f"Successfully saved video to: {output_path}")
            # save the prompts used to generate the video
            output_prompt_path = os.path.join(output_dir, "prompts.txt")
            prompts_to_save = {"prompt": prompt, "negative_prompt": negative_prompt}
            if (
                self.pipe.prompt_refiner is not None
                and getattr(self.pipe.config, "prompt_refiner_config", None) is not None
                and getattr(self.pipe.config.prompt_refiner_config, "enabled", False)
            ):
                prompts_to_save["refined_prompt"] = prompt_used
            save_text_prompts(prompts_to_save, output_prompt_path)
            log.success(f"Successfully saved prompt file to: {output_prompt_path}")

    def infer(self, args: dict):
        self._infer(**args)


def create_worker(create_model=True):

    log.info("Creating predict pipeline and validator")
    cfg = Config()
    pipeline = None
    if create_model:
        pipeline = Video2World_Worker(
            num_gpus=int(os.environ.get("WORLD_SIZE", 1)),
            checkpoint_dir=cfg.checkpoint_dir,
            model_size=cfg.model_size,
            resolution=cfg.resolution,
            fps=cfg.fps,
            load_ema=cfg.load_ema,
            disable_prompt_refiner=cfg.disable_prompt_refiner,
            offload_prompt_refiner=cfg.offload_prompt_refiner,
        )
        gc.collect()
        torch.cuda.empty_cache()

    validator = Video2World_Validator()
    return pipeline, validator
