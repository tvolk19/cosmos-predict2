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

import os
import json
from dataclasses import dataclass


default_request_v2w = json.dumps(
    {
        "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
        "input_path": "assets/video2world/input0.jpg",
        "num_conditional_frames": 1,
        "aspect_ratio": "16:9",
        "guidance": 7.0,
        "seed": 0,
    },
    indent=2,
)

default_request_t2i = json.dumps(
    {
        "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
        "aspect_ratio": "16:9",
        "seed": 0,
    },
    indent=2,
)


help_text_v2w = """
                    ### Generation Parameters:
                    - `input_path` (string): Path to input image/video file (default: "assets/video2world/input0.jpg")
                    - `prompt` (string): Text description of desired output (default: empty string)
                    - `negative_prompt` (string): What to avoid in generation (default: predefined negative prompt)
                    - `aspect_ratio` (string): Output aspect ratio (default: "16:9")
                    - `num_conditional_frames` (int): Number of conditioning frames (default: 1)
                    - `guidance` (float): Classifier-free guidance scale (default: 7.0)
                    - `seed` (int): Random seed for reproducibility (default: 0)
                    """
help_text_t2i = """
                    ### Generation Parameters:
                    - `prompt` (string): Detailed text description of the desired video content and scene (default: predefined positive prompt)
                    - `negative_prompt` (string): Text describing elements to exclude from generation (default: empty string)
                    - `aspect_ratio` (string): Output video aspect ratio in width:height format (default: "16:9")
                    - `seed` (int): Random seed value for reproducible generation results (default: 0)
             """


@dataclass
class Config:
    # model_name: str = os.getenv("MODEL_NAME", "video2world")
    # checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    # num_gpus: int = int(os.environ.get("NUM_GPU", 1))
    model_size: str = os.getenv("MODEL_SIZE", "2B")  # 2B or 14B
    resolution: str = os.getenv("RESOLUTION", "720")  # 480 or 720
    fps: int = int(os.getenv("FPS", 16))  # 10 or 16
    load_ema: bool = os.getenv("LOAD_EMA", "False").lower() in ("true", "1", "yes")
    disable_prompt_refiner: bool = os.getenv("DISABLE_PROMPT_REFINER", "True").lower() in ("true", "1", "yes")
    offload_prompt_refiner: bool = os.getenv("OFFLOAD_PROMPT_REFINER", "False").lower() in ("true", "1", "yes")
    header = {"video2world": "Cosmos-Predict2 Video2World", "text2image": "Cosmos-Predict2 Text2Image"}

    help_text = {
        "video2world": help_text_v2w,
        "text2image": help_text_t2i,
    }

    default_request = {"video2world": default_request_v2w, "text2image": default_request_t2i}
