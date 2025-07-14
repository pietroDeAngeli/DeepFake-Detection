import os
import subprocess
from tqdm import tqdm

def convert_single_video_to_h264_with_mv(input_path, output_path):
    """
    Re-encodes a single .mp4 video to H.264 with motion vectors.

    Parameters:
        input_path (str): Full path to the input .mp4 file.
        output_path (str): Full path to the output .mp4 file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
    "ffmpeg",
    "-i", input_path,
    "-c:v", "libx264",           # H.264 codec
    "-preset", "veryfast",       # velocità di encoding decente
    "-g", "30",                  # GOP: keyframe ogni 30 frame
    "-bf", "2",                  # abilita B-frames
    "-flags2", "+export_mvs",   # rende i motion vectors visibili
    "-pix_fmt", "yuv420p",      # formato standard per compatibilità
    "-an",                      # disabilita l'audio
    "-movflags", "+faststart",  # utile per streaming
    "-y",                       # sovrascrive senza chiedere
    output_path
]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def convert_all_videos_in_dir(input_dir, output_dir):
    """
    Converts all .mp4 videos in input_dir to H.264 with motion vectors,
    saving them to output_dir.

    Parameters:
        input_dir (str): Input directory path.
        output_dir (str): Output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    for filename in tqdm(video_files, desc=f"Converting videos in {input_dir}"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        convert_single_video_to_h264_with_mv(input_path, output_path)

    print("Conversion completed for all files.")


if __name__ == "__main__":
    # ✅ Converte un solo file (puoi cambiare qui il file che vuoi)
    input_file = "../FF++/fake/01_02__outside_talking_still_laughing__YVGY8LOK.mp4"
    output_file = "../FF++/fake_h264/01_02__outside_talking_still_laughing__YVGY8LOK.mp4"
    convert_single_video_to_h264_with_mv(input_file, output_file)

    # Se vuoi attivare la conversione batch in futuro:
    # convert_all_videos_in_dir("../FF++/fake", "../FF++/fake_h264")
    # convert_all_videos_in_dir("../FF++/real", "../FF++/real_h264")
