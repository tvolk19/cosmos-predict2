import json
import os
from cosmos_gradio.server_config import Config
from cosmos_gradio.gradio_app_cli import GradioCLIApp
from cosmos_gradio.gradio_app import GradioApp
from cosmos_gradio.gradio_interface import create_gradio_interface
from imaginaire.utils import log
from server.model_config import Config as ModelConfig

default_request = json.dumps(
    {
        "input_path": "assets/video2world/input0.jpg",
        "num_conditional_frames": 1,
        "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
    },
    indent=2,
)

help_text = """
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

if __name__ == "__main__":

    cfg = Config()
    model_cfg = ModelConfig()

    log.info(f"Starting Gradio app with config: {str(cfg)}")

    if cfg.use_cli:  # todo use cfg.cli_app instead?
        app = GradioCLIApp(num_workers=cfg.num_gpus, checkpoint_dir=cfg.checkpoint_dir)
    else:
        app = GradioApp(cfg)

    # todo pass in model specific text to UI

    interface = create_gradio_interface(app.infer, default_request=default_request, help_text=help_text, cfg=cfg)

    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        max_file_size="500MB",
        allowed_paths=[cfg.output_dir, cfg.uploads_dir],
    )
