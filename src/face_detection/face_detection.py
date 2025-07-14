# Custom Libraries
import face_detection.face_detection_tools as tools

# Function to perform face detection on a list of videos

def faces_detection(videos):
    """
    Applies face extraction on a list of videos.

    For each video in the list:
    - Processes all frames to detect and crop the largest face per frame
    - Uses a face detector to find faces
    - Resizes each cropped face to 224x224 pixels
    - Stores the resulting face sequences for each video

    Parameters:
        videos (list of str): List of paths to video files.

    Returns:
        list of list of Face: A nested list where each sublist contains 
        the extracted face frames (as Face objects) for one video.
    """
    faces = []
    detector = tools.initialize_detector()

    for video in videos:

        video_faces = tools.face_video_extractor_2(video, detector)
        faces.append(video_faces)

    return faces