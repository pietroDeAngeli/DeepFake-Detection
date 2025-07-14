import face_detection.face_detection as faceDetection

import tools.tools as tools

fake_dir = "../FF++/fake"
real_dir = "../FF++/real"

fake_videos = tools.get_dir_videos(fake_dir)
real_videos = tools.get_dir_videos(real_dir)

# Extract faces from fake and real videos
fake_faces = faceDetection.faces_detection(fake_videos)
real_faces = faceDetection.faces_detection(real_videos)

# Extract motion vectors from fake and real videos


# Aggregate the results




# FINETUNING?