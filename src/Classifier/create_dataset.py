import pandas as pd
import os

# Custom librariesù
from ..tools.tools import get_dir_videos
from ..motion_vectors.motion_vectors import extract_motion_vectors
from ..face_detection.face_detection import faces_detection
 
from ..face_detection.face_detection_tools import FaceBox

fake_dir = "../FF++/fake"
real_dir = "../FF++/real"

def create_dataset():
    # Get the list of video files in the directories
    fake_videos = get_dir_videos(fake_dir)
    real_videos = get_dir_videos(real_dir)

    # Extract faces from fake and real videos
    fake_faces = faces_detection(fake_videos)
    real_faces = faces_detection(real_videos)

    # Extract motion vectors from fake and real videos
    fake_motion_vectors = extract_motion_vectors(fake_videos)
    real_motion_vectors = extract_motion_vectors(real_videos)

    # Aggregate the results into a DataFrame
    data = []
    
    for face, mv in zip(fake_faces, fake_motion_vectors):
        data.append({'features': face, 'motion_vector': mv, 'label': 'fake'})
    
    for face, mv in zip(real_faces, real_motion_vectors):
        data.append({'features': face, 'motion_vector': mv, 'label': 'real'})

    df = pd.DataFrame(data)
    
    # Save the dataset to a CSV file
    df.to_csv('dataset.csv', index=False)
