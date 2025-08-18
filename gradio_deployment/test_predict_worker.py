from gradio_deployment.video2world_worker import Video2World_Validator, Video2World_Worker
from gradio_deployment.text2image_worker import Text2Image_Validator, Text2Image_Worker
from gradio_deployment.text2world_worker import Text2World_Worker
from imaginaire.utils import log
from gradio_deployment.model_config import Config

prompt = "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
sample = {"input_path": "assets/video2world/input0.jpg", "num_conditional_frames": 1, "prompt": prompt}

cfg = Config()


def test_text2image():
    validator = Text2Image_Validator()
    model_params = validator.parse_and_validate({})
    pipeline = Text2Image_Worker(num_gpus=1, checkpoint_dir=cfg.checkpoint_dir)

    log.info("Inference start****************************************")
    model_params["output_dir"] = "outputs/"
    pipeline.infer(
        model_params,
    )
    log.info("Inference complete****************************************")


def test_video2world():
    validator = Video2World_Validator()
    model_params = validator.parse_and_validate(sample)
    pipeline = Video2World_Worker(num_gpus=1, checkpoint_dir=cfg.checkpoint_dir)

    log.info("Inference start****************************************")

    model_params["output_dir"] = "outputs/"
    pipeline.infer(
        model_params,
    )
    log.info("Inference complete****************************************")


def test_text2world():
    validator = Text2Image_Validator()
    model_params = validator.parse_and_validate({})
    pipeline = Text2World_Worker(num_gpus=1, checkpoint_dir=cfg.checkpoint_dir)

    log.info("Inference start****************************************")
    model_params["output_dir"] = "outputs/"
    pipeline.infer(
        model_params,
    )
    log.info("Inference complete****************************************")


if __name__ == "__main__":
    log.info(cfg)
    # test_video2world()
    # test_text2image()
    test_text2world()
