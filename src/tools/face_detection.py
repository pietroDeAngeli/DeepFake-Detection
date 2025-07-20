# OpenCV
import cv2
import av
import random
import os

class FaceBox:
    def __init__(self, x,y,side):
        self.x = x
        self.y = y
        self.side = side

class Face:
    def __init__(self, image: cv2.typing.MatLike, box: FaceBox):
        self.image = image
        self.box = box

def initialize_detector(model_path:str) -> cv2.FaceDetectorYN:
    """
    Initializes the YuNet face detector.

    Returns:
        MTCNN: An instance of the MTCNN face detector.
    """
    #detector = MTCNN(device="cpu") # MTCNN model initialization

    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(1920, 1080),
    )

    return detector

def make_square_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int]:
    """
    Reshape a rectangular bounding box to a square box

    Parameters:
        x (int):   X coordinate of the top-left corner of the original box.
        y (int):   Y coordinate of the top-left corner of the original box.
        w (int):   Width of the original box.
        h (int):   Height of the original box.
        img_w (int): Width of the full image.
        img_h (int): Height of the full image.

    Returns:
        (new_x, new_y, side): top-left corner and side length of the square box.
    """
    # 1) choose the square side so it covers the original box,
    #    but never larger than the bigger image dimension
    side = max(w, h)
    max_img_side = max(img_w, img_h)
    side = min(side, max_img_side)

    # 2) compute the original box center
    cx = x + w // 2
    cy = y + h // 2

    # 3) center the square on the box center
    new_x = cx - side // 2
    new_y = cy - side // 2

    # 4) clamp into valid image coords
    new_x = max(0, min(new_x, img_w - side))
    new_y = max(0, min(new_y, img_h - side))

    return int(new_x), int(new_y), int(side)

def face_frame_extractor(
    detector: cv2.FaceDetectorYN,
    frame: av.video.frame.VideoFrame,
    conf_threshold: float = 0.5
) -> Face | None:
    """
    Extracts and crops the largest face from a frame
    using the provided face detector.

    Args:
        frame (VideoFrame): VideoFrame object.
        detector: An initialized cv2.FaceDetectorYN instance.
        conf_threshold (float): minimum confidence to keep a detection.

    Returns:
        Face|None: one Face if detected, otherwise None.
    """

    # Configure detector for this frame resolution
    frame = frame.to_ndarray(format='bgr24')
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    # Perform face detection
    num, faces_mat = detector.detect(frame)

    # No faces detected
    if num == 0 or faces_mat is None:
        return None

    # Filter and pick the largest face
    biggest = None
    for det in faces_mat[:int(num)]:
        x, y, w_box, h_box, score = det[:5]
        if score < conf_threshold:
            continue
        x, y, w_box, h_box = map(int, (x, y, w_box, h_box))
        area = w_box * h_box
        if biggest is None or area > biggest["box"][2] * biggest["box"][3]:
            biggest = {
                "box": (x, y, w_box, h_box),
            }

    # There are no faces above the confidence threshold
    if biggest is None:
        return None

    # 5. Crop, resize, and pack result
    x, y, width, height = biggest["box"]

    # Make bounding box square
    new_x, new_y, side = make_square_box(x, y, width, height, w, h)
    face_box = FaceBox(new_x, new_y, side)

    # Crop and resize to 224×224
    crop = frame[new_y:new_y + side, new_x:new_x + side]
    face_img = cv2.resize(crop, (224, 224))
    
    # Create Face object
    face = Face(image=face_img, box=face_box)

    return face

def extract_frames_with_faces(detector: cv2.FaceDetectorYN,
                              video_path: str,
                              n_frames: int = 100
                              ) -> list[tuple[av.video.frame.VideoFrame, Face]] | None:
    assert os.path.exists(video_path), f"Video file {video_path} does not exist"

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.options = {"flags2": "+export_mvs"}

    reservoir: list[tuple[av.video.frame.VideoFrame, Face]] = []
    count = 0

    for frame in container.decode(video=0):
        count += 1
        j = random.randrange(count)  # estrai indice casuale in [0..count-1]
        if j < n_frames:
            # solo ora chiamo il face detector
            face = face_frame_extractor(detector=detector, frame=frame)
            if face is None:
                continue

            item = (frame, face)
            if len(reservoir) < n_frames:
                reservoir.append(item)
            else:
                reservoir[j] = item

    if not reservoir:
        print(f"Warning: no faces found in {video_path}")
        return None

    return reservoir