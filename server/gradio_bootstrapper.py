import os
from cosmos_gradio.server_config import Config
from cosmos_gradio.gradio_app_cli import GradioCLIApp
from cosmos_gradio.gradio_app import GradioApp
from cosmos_gradio.gradio_interface import create_gradio_interface
from imaginaire.utils import log


if __name__ == "__main__":

    cfg = Config()
    log.info(f"Starting Gradio app with config: {str(cfg)}")

    if cfg.use_cli:  # todo use cfg.cli_app instead?
        app = GradioCLIApp(num_workers=cfg.num_gpus, checkpoint_dir=cfg.checkpoint_dir)
    else:
        app = GradioApp(cfg)

    # todo pass in model specific text to UI

    interface = create_gradio_interface(app.infer, cfg)

    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        max_file_size="500MB",
        allowed_paths=[cfg.output_dir, cfg.uploads_dir],
    )
