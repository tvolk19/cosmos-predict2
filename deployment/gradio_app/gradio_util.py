import os
from datetime import datetime
from imaginaire.utils import log


def get_output_folder(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_dir, f"generation_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def get_outputs(output_folder):
    # Check if any mp4 file was generated
    mp4_files = [f for f in os.listdir(output_folder) if f.endswith(".mp4")]
    img_files = [f for f in os.listdir(output_folder) if f.endswith(".jpg")]
    txt_files = [f for f in os.listdir(output_folder) if f.endswith(".txt")]

    # Read the generated prompt
    final_prompt = ""
    if txt_files:
        with open(os.path.join(output_folder, txt_files[0]), "r", encoding="utf-8") as f:
            final_prompt = f.read().strip()

    output_path = None
    if mp4_files:
        # Use the first mp4 file found
        output_path = os.path.join(output_folder, mp4_files[0])

    if img_files:
        # Use the first image file found
        output_path = os.path.join(output_folder, img_files[0])

    if output_path:
        return (
            output_path,
            f"Output saved to: {output_path}\nFinal prompt: {final_prompt}",
        )
    else:
        return None, f"Generation failed - no output was created\nCheck folder: {output_folder}"
