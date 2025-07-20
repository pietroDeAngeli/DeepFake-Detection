import os
import json
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'

import tools.face_detection as faceDetection
import tools.tools as tools
import tools.motion_vectors as motionVectos
import tools.feature_computation as featureComputation


if __name__ == "__main__":
    video_path = "FF++/real/01__exit_phone_room.mp4"

    print("Initializing face detection...")
    detector = faceDetection.initialize_detector("models/face_detection_yunet_2023mar.onnx")

    print("Extracting faces...")
    results = faceDetection.extract_frames_with_faces(detector, video_path)

    # Prepariamo la lista di FaceBox|None
    frames, faces = zip(*results)
    # ora `frames` e `faces` sono due tuple; se ti servono liste:
    frames = list(frames)
    video_faces  = list(faces)
    print(video_faces)
    

    face_boxes = [ 
        face.box if face is not None else None
        for face in video_faces
    ]

    print("Extracting motion vectors and IMs...")
    results = motionVectos.extract_motion_vectors_and_im(
        frames, face_boxes
    )

    mv_x, mv_y, ims = zip(*results)

    # Se ti servono liste anziché tuple:
    mv_x  = list(mv_x)
    mv_y  = list(mv_y)
    ims   = list(ims)

    print("Computing per-frame feature tensor...")
    # Usiamo la variante che costruisce il tensor (numpy array)
    feature_matrix = featureComputation.compute_features_video_tensor(
        mv_x, mv_y, ims
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

        im = ims[idx]
        info["im_mask"] = im.tolist() if im is not None else None

        frames_info.append(info)

    # Compongo il JSON finale
    output = {
        "video_path": video_path,
        "label": 1,    # 1 per real videos
        "frames": frames_info
    }

    with open("informations.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Done. JSON written to informations.json")
