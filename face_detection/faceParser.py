# Face Recognition and Detection using MTCNN
from mtcnn import MTCNN
from mtcnn.utils.images import load_image

# OpenCV
import cv2

# Tools
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Function to get the biggest face
def get_bigger_face(results):
    """
    Convert a rectangular bounding box to a square box centered on the original.

    Ensures the new square box stays within the image boundaries.

    Parameters:
        x (int): X coordinate of the top-left corner of the original box.
        y (int): Y coordinate of the top-left corner of the original box.
        w (int): Width of the original box.
        h (int): Height of the original box.
        img_w (int): Width of the full image.
        img_h (int): Height of the full image.

    Returns:
        tuple: Coordinates (new_x, new_y, side, side) of the square box.
    """
    bigger_face = None
    for face in results:
        # Get the bounding box coordinates
        x, y, width, height = face['box']
        
        # take the bigger one
        if bigger_face is None or (width * height) > (bigger_face['box'][2] * bigger_face['box'][3]):
            bigger_face = face

    return bigger_face

# Reshape the box to a square
def make_square_box(x, y, w, h, img_w, img_h):
    """
    Retrieve all .mp4 video file paths from a given directory.

    Parameters:
        path (str): Path to the directory containing video files.

    Returns:
        list of str: List of full paths to .mp4 video files.
    """
    cx = x + w // 2
    cy = y + h // 2
    side = max(w, h)

    new_x = max(0, cx - side // 2)
    new_y = max(0, cy - side // 2)

    # Adjust if it goes out of bounds
    if new_x + side > img_w:
        new_x = img_w - side
    if new_y + side > img_h:
        new_y = img_h - side

    return int(new_x), int(new_y), int(side), int(side)

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
    Perform face detection on a list of video files.

    For each video, the function:
    - Extracts all frames
    - Detects faces using MTCNN
    - Selects the largest face per frame
    - Applies padding to make the bounding box square
    - Resizes the face crop to 224x224 pixels
    - Stores the processed face images

    Parameters:
        videos (list of str): List of paths to video files.

    Returns:
        list of list of np.ndarray: A nested list where each sublist contains 
        the detected and resized face images (as NumPy arrays) for a single video.
    """

    faces = []
    for i, video in enumerate(videos):
        cap = cv2.VideoCapture(video)

        video_faces = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the frame
            results = detector.detect_faces(frame_rgb)
            
            # Get bigger face
            face = get_bigger_face(results)
            
            # If a face is detected, draw a bounding box
            if face is not None:
                x, y, width, height = face['box']
                img_h, img_w, _ = frame_rgb.shape
                
                # Make the bounding box square
                squared_image = make_square_box(x, y, width, height, img_w, img_h)
                x, y, width, height = squared_image

                face_crop = frame_rgb[y:y+height, x:x+width]
                face_resized = cv2.resize(face_crop, (224, 224))

                video_faces.append(face_resized)
        
        cap.release()
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

