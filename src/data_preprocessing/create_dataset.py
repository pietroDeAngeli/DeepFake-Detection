import os
import json
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
from tqdm import tqdm
import torch

import tools.face_detection as faceDetection
import tools.tools as tools
import tools.motion_vectors as motionVectos
import tools.feature_computation as featureComputation


if __name__ == "__main__":

    fake_dir = "../FF++/fake"
    real_dir = "../FF++/real"
    model_path = "../models/face_detection_yunet_2023mar.onnx"
    out_dir = "../dataset"

    label = False

    videos_path = tools.get_dir_videos(fake_dir)

    # Import the detector
    detector = faceDetection.initialize_detector(model_path)

    for video_path in tqdm(videos_path, desc="Processing videos"):

        # Detect faces
        results = faceDetection.extract_frames_with_faces(detector, video_path, unique_frames=True)

        frames, faces = zip(*results)
        frames = list(frames)
        video_faces  = list(faces)        

        # Extract data
        face_boxes = [ 
            face.box if face is not None else None
            for face in video_faces
        ]

        # Motion Vector extraction
        results = motionVectos.extract_motion_vectors_and_im(
            frames, face_boxes
        )

        # Extract data
        mv_x, mv_y, ims = zip(*results)
        mv_x  = list(mv_x)
        mv_y  = list(mv_y)
        ims   = list(ims)

        # Compute the features 
        feature_matrix = featureComputation.compute_features_video_tensor(
            mv_x, mv_y, ims
        )
        
        # Create dir for the video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_dir = os.path.join(out_dir, video_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the tensor
        torch.save({"features": feature_matrix}, os.path.join(save_dir, "tensors.pt"))
        
        # Save the images
        faces_dir = os.path.join(save_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)
        for idx, face in enumerate(video_faces):
            if face is not None:
                img_path = os.path.join(faces_dir, f"frame_{idx:04d}.jpg")
                cv2.imwrite(img_path, face.image)
        
        # save the data on the video
        meta = {
            "video_path": video_path,
            "label": label,
            "n_frames": feature_matrix.shape[0]
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)