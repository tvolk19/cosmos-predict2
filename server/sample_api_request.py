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

"""
Sample gradio client interactions for the Cosmos Transfer1 Gradio app.

Example usage:

    # Synchronous inference with a file that already exists on the server
    python sample_client.py \
        --url http://localhost:8080/ \
        --example sync \
        --input_video_path path/to/video/on/server.mp4

    # Asynchronous upload + inference
    python sample_client.py \
        --url https://cosmos-transfer1.inference.dgxcloud.ai/ \
        --example async_with_upload \
        --input_video_path path/to/video/on/local/machine.mp4
"""

import argparse
import json
import time
import typing

import gradio_client.client as gradio_client
import gradio_client.utils as gradio_utils
from loguru import logger


sample_request = {
    "input_path": "assets/video2world/input0.jpg",
    "num_conditional_frames": 1,
    "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
}

if __name__ == "__main__":
    client = gradio_client.Client("http://localhost:8080/")

    request_text = json.dumps(sample_request)

    logger.info(f"generate_video_request: {request_text=}")
    result = client.predict(request_text, api_name="/generate_video")
    logger.info(f"generate_video_result: {result=}")

    local_video_path = result[0]["video"]
    logger.info(f"Local video path (downloaded to local machine): {local_video_path=}")
