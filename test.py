import os
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'

import face_detection.face_detection as faceDetection
import motion_vectors.motion_vectors as motionVectors
import feature_extraction.feature_computation as featureExtraction

if __name__ == "__main__":
    video_path = "FF++/real/01__exit_phone_room.mp4"

    print("Initializing face detection...")
    video_faces = faceDetection.faces_detection([video_path])

    faces = video_faces[0]

    print("Extracting motion vectors...")
    face_boxes = [
        face.box if face is not None else None
        for face in faces
    ]

    motion_vectors = motionVectors.extract_motion_vectors(video_path, face_boxes)

    print("Computing features...")
    video_features = featureExtraction.compute_features_video(motion_vectors)

    print("Video features:", video_features)

    

