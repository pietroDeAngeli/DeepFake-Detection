import os

# Get all the videos
def get_dir_videos(path):
    """
    Retrieve all .mp4 video file paths from a given directory.

    Parameters:
        path (str): Path to the directory containing video files.

    Returns:
        list of str: List of full paths to .mp4 video files.
    """
    videos = []
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            videos.append(os.path.join(path, file))
    return videos

