# Face Recognition and Detection using MTCNN
from mtcnn import MTCNN
from mtcnn.utils.images import load_image

# OpenCV
import cv2

# Tools
from PIL import Image
import numpy as np
import os

def initialize_detector():
    """
    Initializes the MTCNN face detector.

    Returns:
        MTCNN: An instance of the MTCNN face detector.
    """
    detector = MTCNN(device="cpu")
    return detector

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

def face_video_extractor(video, detector=None, return_faces=False):
    """
    Extracts and crops the largest face from each frame of a video.

    For each frame in the video:
    - Converts the frame to RGB
    - Detects faces using the provided detector (MTCNN)
    - If at least one face is detected, selects the largest one
    - Applies padding to make the bounding box square
    - Crops and resizes the face to 224x224 pixels
    - Appends the processed face frame to the final list

    Parameters:
        video (str): Path to the video file (.mp4).
        detector (object): A face detector instance with a `detect_faces(img)` method 
                           (MTCNN). Must be initialized before use.
        return_faces (bool): If True, returns the cropped face images; 
                             if False, returns the bounding box coordinates.

    Returns:
        list of np.ndarray: A list of cropped and resized face frames 
                            (one per frame where a face is detected).
    """
    video_faces = []

    cap = cv2.VideoCapture(video)
    
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

            if not return_faces:
                # Return just the shape
                video_faces.append({"x": x, "y": y, "width": width, "height": height})
            else:
                # Return the cropped face
                img_h, img_w, _ = frame_rgb.shape
                
                # Make the bounding box square
                squared_image = make_square_box(x, y, width, height, img_w, img_h)
                x, y, width, height = squared_image

                face_crop = frame_rgb[y:y+height, x:x+width]
                face_resized = cv2.resize(face_crop, (224, 224))

                video_faces.append(face_resized)
    
    cap.release()

    return video_faces
