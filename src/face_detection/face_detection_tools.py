# Face Recognition and Detection using MTCNN
#from mtcnn import MTCNN

# OpenCV
import cv2

# Tools
import numpy as np
#from PIL import Image
#import os

class FaceBox:
    def __init__(self, x,y,side):
        self.x = x
        self.y = y
        self.side = side

class Face:
    def __init__(self, image: cv2.typing.MatLike, box: FaceBox):
        self.image = image
        self.box = box

def initialize_detector():
    """
    Initializes the MTCNN face detector.

    Returns:
        MTCNN: An instance of the MTCNN face detector.
    """
    #detector = MTCNN(device="cpu") # MTCNN model initialization

    model_path = "src/face_detection/face_detection_yunet_2023mar.onnx" # yuNet model path
    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(1920, 1080),
    )

    return detector

# Reshape the box to a square
def make_square_box(x, y, w, h, img_w, img_h):
    """
    Reshape a rectangular bounding box to a square box centered on the original.
    Ensures the new square box stays within the image boundaries.
    Parameters:
        x (int): X coordinate of the top-left corner of the original box.
        y (int): Y coordinate of the top-left corner of the original box.
        w (int): Width of the original box.
        h (int): Height of the original box.
        img_w (int): Width of the full image.
        img_h (int): Height of the full image.
    Returns:
        tuple: Coordinates (new_x, new_y, side) of the square box.
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

    return int(new_x), int(new_y), int(side)

def face_video_extractor(video, detector=None):
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

    Returns:
        list of Face: A list of cropped and resized face frames 
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
        faces = detector.detect_faces(frame_rgb)
        
        # Get bigger face
        face = get_bigger_face(faces)
        
        # If a face is detected, draw a bounding box
        if face is not None:
            x, y, width, height = face['box']

            # Return the cropped face
            img_h, img_w, _ = frame_rgb.shape
            
            # Make the bounding box square
            x, y, side = make_square_box(x, y, width, height, img_w, img_h)
            face_box = FaceBox(x, y, side)

            face_crop = frame_rgb[y:y+side, x:x+side]
            face_image = cv2.resize(face_crop, (224, 224))
            
            video_faces.append(Face(face_image, face_box))
        else:
            video_faces.append(None)
    
    cap.release()
    

    return video_faces

def face_video_extractor_2(video: str, detector) -> list[Face | None]:
    """
    Extracts and crops the largest face from each frame of a video
    using OpenCV’s FaceDetectorYN (YuNet).

    For each frame:
    1. Resize the detector input to the current frame size.
    2. Convert the frame to RGB.
    3. Run face detection with a preallocated output buffer.
    4. Filter detections by confidence threshold.
    5. Select the largest face by area.
    6. Make its bounding box square and crop the face.
    7. Resize the face crop to 224×224 and wrap it in a Face object.

    Args:
        video (str): Path to the video file.
        detector: An initialized cv2.FaceDetectorYN instance.

    Returns:
        List[Face|None]: One Face per frame if detected, otherwise None.
    """
    video_faces = []
    cap = cv2.VideoCapture(video)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Configure detector for this frame resolution
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))

        # Perform face detection
        num, faces_mat = detector.detect(frame)

        # If faces are detected, find the largest one
        biggest_face = None
        if faces_mat is not None:
            # Find the largest face
            for det in faces_mat[:int(num)]:
                x, y, w_box, h_box = map(int, det[:4])
                
                if biggest_face is None or w_box * h_box > biggest_face['box'][2] * biggest_face['box'][3]:
                    biggest_face = {
                        'box': (x, y, w_box, h_box),
                    }

        if biggest_face is not None:
            x, y, width, height = biggest_face['box']

            # Make bounding box square
            new_x, new_y, side = make_square_box(x, y, width, height, w, h)
            face_box = FaceBox(new_x, new_y, side)

            # Crop and resize
            crop = frame[new_y:new_y+side, new_x:new_x+side]
            face_img = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)

            video_faces.append(Face(face_img, face_box))
        else:
            video_faces.append(None)

    cap.release()
    return video_faces