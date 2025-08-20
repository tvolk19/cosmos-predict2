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
from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = os.getenv("MODEL_NAME", "video2world")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    num_gpus: int = int(os.environ.get("NUM_GPU", 1))
    model_size: str = os.getenv("MODEL_SIZE", "2B")  # 2B or 14B
    resolution: str = os.getenv("RESOLUTION", "720")  # 480 or 720
    fps: int = int(os.getenv("FPS", 16))  # 10 or 16
    load_ema: bool = os.getenv("LOAD_EMA", "False").lower() in ("true", "1", "yes")
    disable_prompt_refiner: bool = os.getenv("DISABLE_PROMPT_REFINER", "True").lower() in ("true", "1", "yes")
    offload_prompt_refiner: bool = os.getenv("OFFLOAD_PROMPT_REFINER", "False").lower() in ("true", "1", "yes")
