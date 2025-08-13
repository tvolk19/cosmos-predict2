import os
from deployment.server.generic.model_server import ModelServer
from deployment.server.model_specific.video2world_worker import Video2World_Validator
from deployment.server.model_specific.text2image_worker import Text2Image_Validator
from imaginaire.utils import log
from deployment.server.generic.server_config import Config


prompt = "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
sample = {"input_path": "assets/video2world/input0.jpg", "num_conditional_frames": 1, "prompt": prompt}


def test_text2image():
    os.environ["FACTORY_MODULE"] = "deployment.server.model_specific.text2image_worker"

    folder = "outputs/"
    with ModelServer(Config()) as pipeline:
        validator = Text2Image_Validator()

        log.info("Inference start****************************************")

        model_params = validator.parse_and_validate({})
        model_params["output_dir"] = f"{folder}/"
        pipeline.infer(model_params)
        log.info("Inference complete****************************************")


def test_video2world():
    os.environ["FACTORY_MODULE"] = "deployment.server.model_specific.video2world_worker"

    folder = "outputs/"
    with ModelServer(Config()) as pipeline:
        validator = Video2World_Validator()

        log.info("Inference start****************************************")

        model_params = validator.parse_and_validate(sample)
        model_params["output_dir"] = f"{folder}/"
        pipeline.infer(model_params)
        log.info("Inference complete****************************************")


if __name__ == "__main__":
    test_text2image()
    test_video2world()
