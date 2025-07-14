import os
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'

import face_detection.face_detection as faceDetection
import motion_vectors.motion_vectors as motionVectors

if __name__ == "__main__":
    video_path = "FF++/real/01__exit_phone_room.mp4"

    faces = faceDetection.faces_detection([video_path])
    faces_video = faces[0]

    box = faces_video[0].box
    print(f"Face box: {box.x}, {box.y}, {box.side}")

    face_boxes = [ face.box for face in faces_video ]

    motion_vectors = motionVectors.extract_motion_vectors(video_path, face_boxes)
    # Print the first few motion vectors for verification
    print(f"Motion Vectors: {motion_vectors[:5]}")






