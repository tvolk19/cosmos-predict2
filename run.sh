export NUM_GPUS=8

PROMPT_="A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
# Run video2world generation
# python -m examples.video2world \
#     --model_size 2B \
#     --input_path assets/video2world/input0.jpg \
#     --num_conditional_frames 1 \
#     --prompt "${PROMPT_}" \
#     --save_path output/video2world_2b.mp4
    

# Run video2world generation with context parallelism using torchrun
PYTHONPATH=. torchrun --nproc_per_node=${NUM_GPUS} examples/video2world.py \
    --model_size 2B \
    --input_path assets/video2world/input0.jpg \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_2b_${NUM_GPUS}gpu.mp4 \
    --num_gpus ${NUM_GPUS}