# Tools
import tdqm as tqdm

# Custom Libraries
import face_detection_tools as tools

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
    detector = tools.initialize_detector()

    for video in tqdm(videos, desc="Extracting faces"):

        video_faces = tools.face_video_extractor(video, detector)
        faces.append(video_faces)

    return faces


#fake_videos = get_dir_videos(videos_path)
#real_videos = get_dir_videos(videos_path)

#fake_faces = faces_detection(fake_videos)
#real_faces = faces_detection(real_videos)
