# Face Recognition and Detection using MTCNN
from mtcnn import MTCNN

# Tools
import os
import tdqm as tqdm

# Custom Libraries
import face_detection_tools as tools


# Get all the videos
def get_videos(path):
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

# Function to perform face detection on a list of videos
def faces_detection(videos):
    """
    Applies face extraction on a list of videos.

    For each video in the list:
    - Processes all frames to detect and crop the largest face per frame
    - Uses a face detector (e.g., MTCNN) to find faces
    - Resizes each cropped face to 224x224 pixels
    - Stores the resulting face sequences for each video

    Parameters:
        videos (list of str): List of paths to video files.

    Returns:
        list of list of np.ndarray: A nested list where each sublist contains 
        the extracted face frames (as NumPy arrays) for one video.
    """
    faces = []
    for video in tqdm(videos, desc="Extracting faces"):

        video_faces = tools.face_video_extractor(video, detector)
        faces.append(video_faces)

    return faces


# Create a detector instance
detector = MTCNN(device="cpu")

videos_path = "../FF++/"

fake_videos = get_videos(os.path.join(videos_path, "fake"))
real_videos = get_videos(os.path.join(videos_path, "real"))

fake_faces = faces_detection(fake_videos)
real_faces = faces_detection(real_videos)

fake_faces, real_faces

