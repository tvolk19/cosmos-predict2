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


def create_gradio_interface(infer_func, default_request, help_text, cfg):
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
                    value=default_request,
                    lines=20,
                    interactive=True,
                )

                # Help section
                with gr.Accordion("Request Format Help", open=False):
                    gr.Markdown(help_text)
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
