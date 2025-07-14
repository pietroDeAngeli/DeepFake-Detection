# Tools
from tqdm import tqdm

# Custom Libraries
import face_detection.face_detection_tools as tools

# Function to perform face detection on a list of videos

def faces_detection(videos):
    """
    For each video in the provided list, this function extracts faces
    and returns a list of lists containing the detected faces.

    Parameters:
        videos (list of str): List of paths to video files.

    Returns:
        list of list of Face: A nested list where each sublist contains 
        the extracted face frames (as Face objects) for one video.
    """
    faces = []
    detector = tools.initialize_detector()

    for video in tqdm(videos, desc="Extracting faces"):

        video_faces = tools.face_video_extractor(video, detector)
        faces.append(video_faces)

    return faces


#fake_videos = get_dir_videos(videos_path)
#real_videos = get_dir_videos(videos_path)

#fake_faces = faces_detection(fake_videos)
#real_faces = faces_detection(real_videos)
