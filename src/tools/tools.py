import os
import torch
from dataclasses import dataclass
import tools.face_detection as faceDetection
import tools.motion_vectors as motionVectors
import tools.feature_computation as featureComputation
import classifier.network as nw
import cv2
import time
import json

@dataclass
class Result:
    total_time_s: float
    face_time_s: float
    mv_time_s: float
    cls_time_s: float
    prob_real: float  # probability for REAL (0..1)
    label_str: str    # "REAL" or "FAKE"
    


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

def pipeline(models_dir = "../../models", temp_dir = "../../temp", video_path = None) -> Result:

    total_start_time = time.time()

    json_path = os.path.join(temp_dir, "result.json")
    
    # Models paths
    face_detection_model_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    classifier_model_path = os.path.join(models_dir, "classifier.pth")

    # Temp dir
    os.makedirs(temp_dir, exist_ok=True)

    faces_dir = os.path.join(temp_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)

    report_file = os.path.join(temp_dir, "report.json")

    # Detector inizialization
    detector = faceDetection.initialize_detector(face_detection_model_path)

    # Extract faces
    time_start = time.time()
    results = faceDetection.extract_frames_with_faces(detector, video_path, unique_frames=True)
    end_time = time.time()
    face_time = end_time - time_start
    
    if results is None or len(results) == 0:
        raise RuntimeError("No faces detected in the video.")
    
    frames, faces = zip(*results)
    frames = list(frames)
    video_faces  = list(faces)

    # Extract data
    face_boxes = [ 
        face.box if face is not None else None
        for face in video_faces
    ]

    # Motion Vector extraction
    time_start = time.time()
    results = motionVectors.extract_motion_vectors_and_im(
        frames, face_boxes
    )
    end_time = time.time()
    mv_time = end_time - time_start

    # I chose to not save the motion vectors
    #torch.save({"features": feature_matrix}, os.path.join(temp_dir, "tensors.pt"))

    # Extract data
    mv_x, mv_y, ims = zip(*results)
    mv_x  = list(mv_x)
    mv_y  = list(mv_y)
    ims   = list(ims)

    # Compute the features 
    feature_matrix = featureComputation.compute_features_video_tensor(
        mv_x, mv_y, ims
    )
    
    # Save the images
    for idx, face in enumerate(video_faces):
        if face is not None:
            img_path = os.path.join(faces_dir, f"frame_{idx:02d}.jpg")
            cv2.imwrite(img_path, face.image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nw.MLP(in_channels=3)

    # carica i pesi salvati nel training
    state = torch.load(classifier_model_path, map_location=device)
    model.load_state_dict(state)

    # 3. Porta su GPU se disponibile
    model = model.to(device)
    model.eval()
    
    # 4. Prepara i tensori
    imgs_tensor = torch.stack([
        torch.from_numpy(face.image).permute(2,0,1).float().div(255.0)
        for face in video_faces if face is not None
    ], dim=0)  # [N,3,H,W]

    mv_tensor = feature_matrix.permute(0,3,1,2).float()  # [N,3,H,W]

    imgs_tensor = imgs_tensor.to(device)
    mv_tensor   = mv_tensor.to(device)

    # 5. Inference
    start_time = time.time()
    with torch.no_grad():
        logits = model((imgs_tensor, mv_tensor)) 
        prob_real = torch.sigmoid(logits).item()
        label_str = "REAL" if prob_real >= 0.5 else "FAKE"
    end_time = time.time()
    cls_time = end_time - start_time

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    json.dump({
        "video_path": video_path,
        "prob_real": prob_real,
        "label_str": label_str,
        "times": {
            "total_time_s": total_time,
            "face_time_s": face_time,
            "mv_time_s": mv_time,
            "cls_time_s": cls_time,
        }
    }, open(report_file, "w"))

    return Result(
        total_time_s = total_time,
        face_time_s = face_time,
        mv_time_s   = mv_time,
        cls_time_s  = cls_time,
        prob_real   = prob_real,
        label_str   = label_str,
    )


def load_video_data(video_dir: str, label: int, augment_fn=None):
    """
    Load one video sample: MV+IM tensors + face images + label.

    Args:
        video_dir (str): path to the directory containing tensors.pt and faces/
        label (int): 0 = Fake, 1 = Real
        augment_fn (callable or None): optional function (img, mv) -> (img_aug, mv_aug)

    Returns:
        ((imgs_tensor, mv_tensor), label_tensor)
        - imgs_tensor: [N,3,H,W] float32 in [0,1]
        - mv_tensor:  [N,3,H,W] float32
        - label_tensor: torch.tensor([label], dtype=torch.float32) shape [1]
    """
    # 1) Load MV+IM features
    tensor_path = os.path.join(video_dir, "tensors.pt")
    features = torch.load(tensor_path)["features"]        # [N,H,W,3]
    mv_tensor = features.permute(0, 3, 1, 2).float()      # [N,3,H,W]

    # 2) Load cropped face images (RGB -> float32 [0,1])
    faces_dir = os.path.join(video_dir, "faces")
    img_files = sorted(os.listdir(faces_dir))
    imgs = []
    for fname in img_files:
        img = cv2.imread(os.path.join(faces_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        imgs.append(img)
    imgs_tensor = torch.stack(imgs, dim=0)                # [N,3,H,W]

    # 3) Apply augmentation if provided
    if augment_fn is not None:
        aug_imgs, aug_mvs = [], []
        for i in range(imgs_tensor.size(0)):
            img_i, mv_i = augment_fn(imgs_tensor[i], mv_tensor[i])
            aug_imgs.append(img_i)
            aug_mvs.append(mv_i)
        imgs_tensor = torch.stack(aug_imgs, dim=0)
        mv_tensor   = torch.stack(aug_mvs,  dim=0)

    # 4) Label sempre shape [1]
    label_tensor = torch.tensor([label], dtype=torch.float32)

    return (imgs_tensor, mv_tensor), label_tensor
    


