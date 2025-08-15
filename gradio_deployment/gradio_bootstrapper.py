import json
import os
from cosmos_gradio.server_config import Config
from cosmos_gradio.gradio_app_cli import GradioCLIApp
from cosmos_gradio.gradio_app import GradioApp
from cosmos_gradio.gradio_interface import create_gradio_interface
from imaginaire.utils import log
from gradio_deployment.model_config import Config as ModelConfig

header = {"video2world": "Cosmos-Predict2 Video2World", "text2image": "Cosmos-Predict2 Text2Image"}

default_request_v2w = json.dumps(
    {
        "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
        "input_path": "assets/video2world/input0.jpg",
        "num_conditional_frames": 1,
        "aspect_ratio": "16:9",
        "num_conditional_frames": 1,
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

default_request = {"video2world": default_request_v2w, "text2image": default_request_t2i}

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

help_text = {
    "video2world": help_text_v2w,
    "text2image": help_text_t2i,
}

if __name__ == "__main__":

    cfg = Config()
    model_cfg = ModelConfig()

    log.info(f"Starting Gradio app with model config: {str(model_cfg)}")
    log.info(f"server config: {str(cfg)}")

    if cfg.use_cli:  # todo use cfg.cli_app instead?
        app = GradioCLIApp(num_workers=cfg.num_gpus, checkpoint_dir=cfg.checkpoint_dir)
    else:
        app = GradioApp(cfg)

    interface = create_gradio_interface(
        app.infer,
        header=header[model_cfg.model_name],
        default_request=default_request[model_cfg.model_name],
        help_text=help_text[model_cfg.model_name],
        cfg=cfg,
    )

    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        max_file_size="500MB",
        allowed_paths=[cfg.output_dir, cfg.uploads_dir],
    )
