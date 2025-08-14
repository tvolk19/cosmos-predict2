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
import gradio as gr
from cosmos_gradio.server_config import Config

from cosmos_gradio import gradio_file_server, gradio_log_file_viewer


def create_gradio_interface(infer_func, cfg):
    with gr.Blocks(title="Cosmos-Transfer2 Video Generation", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Cosmos-Transfer2: World Generation with Adaptive Multimodal Control")
        gr.Markdown("Upload a video and configure controls to generate a new video with the Cosmos-Transfer1 model.")

        with gr.Row():
            gradio_file_server.file_server_components(cfg.uploads_dir, open=False)

        gr.Markdown("---")
        gr.Markdown(f"**Output Directory**: {cfg.output_dir}")

        with gr.Row():
            with gr.Column(scale=1):
                # Single request input field
                request_input = gr.Textbox(
                    label="Request (JSON)",
                    value=json.dumps(
                        {
                            "input_path": "assets/video2world/input0.jpg",
                            "num_conditional_frames": 1,
                            "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
                        },
                        indent=2,
                    ),
                    lines=20,
                    interactive=True,
                )

                # Help section
                with gr.Accordion("Request Format Help", open=False):
                    gr.Markdown(
                        """
                    ### Required Fields:
                    At least one of the following controlnet specifications must be provided:
                    - `vis` (object): Vis controlnet (default: {"control_weight": 0.0})
                    - `seg` (object): Segmentation controlnet (default: {"control_weight": 0.0})
                    - `edge` (object): Edge controlnet (default: {"control_weight": 0.0})
                    - `depth` (object): Depth controlnet (default: {"control_weight": 0.0})
                    - `keypoint` (object): Keypoint controlnet (default: {"control_weight": 0.0})
                    - `upscale` (object): Upscale controlnet (default: {"control_weight": 0.0})
                    - `hdmap` (object): HDMap controlnet (default: {"control_weight": 0.0})
                    - `lidar` (object): Lidar controlnet (default: {"control_weight": 0.0})

                    ### Optional Fields:
                    - `input_video_path` (string): Path to the input video file
                    - `prompt` (string): Text prompt describing the desired output
                    - `negative_prompt` (string): What to avoid in the output
                    - `guidance` (float): Guidance scale (1-15, default: 7.0)
                    - `num_steps` (int): Number of inference steps (10-50, default: 35)
                    - `seed` (int): Random seed (default: 1)
                    - `sigma_max` (float): Maximum noise level (0-80, default: 70.0)
                    - `blur_strength` (string): One of ["very_low", "low", "medium", "high", "very_high"] (default: "medium")
                    - `canny_threshold` (string): One of ["very_low", "low", "medium", "high", "very_high"] (default: "medium")
                    ```
                    """
                    )
                with gr.Accordion("Tips", open=False):
                    gr.Markdown(
                        """
                    - **Use the file browser above** to upload your video and copy its path for the `input_video` field
                    - **Describe a single, captivating scene**: Focus on one scene to prevent unnecessary shot changes
                    - **Use detailed prompts**: Rich descriptions lead to better quality outputs
                    - **Experiment with control weights**: Different combinations can yield different artistic effects
                    - **Adjust sigma_max**: Lower values preserve more of the input video structure
                    """
                    )

            with gr.Column(scale=1):
                # Output
                output_video = gr.Video(label="Generated Video", height=400)
                status_text = gr.Textbox(label="Status", lines=5, interactive=False)
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

        gradio_log_file_viewer.log_file_viewer(log_file=cfg.log_file, num_lines=100, update_interval=1)

        generate_btn.click(
            fn=infer_func,
            inputs=[request_input],
            outputs=[output_video, status_text],
            api_name="generate_video",
        )

    return interface
