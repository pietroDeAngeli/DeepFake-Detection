import os
import json
import cv2
from tqdm import tqdm
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'

import face_detection.face_detection_tools as faceDetectionTools
import motion_vectors.motion_vectors as motionVectors
import feature_extraction.feature_computation as featureExtraction
import tools.tools as tools

if __name__ == "__main__":
    videos = tools.get_dir_videos("FF++/real")
    detector = faceDetectionTools.initialize_detector()

    for video_path in tqdm(videos, desc="Processing videos"):

        print("Extracting frames from video...")
        frames = tools.extract_n_frames(video_path)  # lista di av.VideoFrame

        print("Initializing face detection...")
        video_faces = faceDetectionTools.face_frames_extractor(frames, detector)

        # Prepariamo la lista di FaceBox|None
        face_boxes = [
            face.box if face is not None else None
            for face in video_faces
        ]

        print("Extracting motion vectors and IMs...")
        motion_vectors, information_masks = motionVectors.extract_motion_vectors_and_im_from_frames(
            frames, face_boxes
        )

        print("Computing per-frame feature tensor...")
        # Usiamo la variante che costruisce il tensor (numpy array)
        feature_matrix = featureExtraction.compute_features_video_tensor(
            motion_vectors, information_masks
        )


        print("Building JSON dump with per-frame info...")
        frames_info = []
        for idx, face in enumerate(video_faces):
            info = {"frame_index": idx}

            # Save image
            if face is not None: 
                filename = os.path.splitext(os.path.basename(video_path))[0]
                img_path = f"dataset/face_images/{filename}_{idx}.jpg"
                cv2.imwrite(img_path, face.image)
                info["image_path"] = img_path 
            else:
                info["image_path"] = None

            # Aggiungiamo il vettore di feature (mv + im) per questo frame
            info["features"] = feature_matrix[idx].tolist()

            im = information_masks[idx]
            info["im_mask"] = im.tolist() if im is not None else None

            frames_info.append(info)

        # Compongo il JSON finale
        output = {
            "video_path": video_path,
            "label": 1,    # 1 per real videos
            "frames": frames_info
        }

        with open("dataset/data.jsonl", "a") as f:
            json.dump(output, f, indent=2)
