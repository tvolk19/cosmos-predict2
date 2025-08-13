from server.predict_worker import (
    PredictWorker,
    PredictValidator,
)
from imaginaire.utils import log
from server.deploy_config import Config

prompt = "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
sample = {"input_path": "assets/video2world/input0.jpg", "num_conditional_frames": 1, "prompt": prompt}


def test_predict():
    validator = PredictValidator()
    model_params = validator.parse_and_validate(sample)
    pipeline = PredictWorker(num_gpus=Config.num_gpus, checkpoint_dir=Config.checkpoint_dir)

    log.info("Inference start****************************************")

    model_params["output_dir"] = "outputs/"
    pipeline.infer(
        model_params,
    )
    log.info("Inference complete****************************************")


if __name__ == "__main__":
    test_predict()
