import time
import av
import tools.face_detection as faceDetection
import tools.motion_vectors as motionVectors

def speed_test(file_path):
    # Simulate some processing on the file

    detector = faceDetection.initialize_detector(model_path="../models/face_detection_yunet_2023mar.onnx")
    
    start_time = time.time()
    res = faceDetection.extract_frames_with_faces(detector, file_path, unique_frames=True)
    end_time = time.time()

    print(f"Faces detected: {len(res) if res else 0}")
    elapsed_time = end_time - start_time
    print(f"Processing time for UNIQUE FRAMES{file_path}: {elapsed_time:.4f} seconds")

    start_time = time.time()
    results = faceDetection.extract_frames_with_faces(detector, file_path, unique_frames=False)
    end_time = time.time()
    print(f"Faces detected: {len(res) if res else 0}")
    elapsed_time = end_time - start_time
    print(f"Processing time for RANDOM FRAMES{file_path}: {elapsed_time:.4f} seconds")

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
    results = motionVectors.extract_motion_vectors_and_im(
        frames, face_boxes
    )

    results = [res for res in results if res is not None]

    # Extract data
    mv_x, mv_y, ims = zip(*results)
    mv_x  = list(mv_x)
    mv_y  = list(mv_y)
    ims   = list(ims)

    print(len(mv_x))


    return elapsed_time

if __name__ == "__main__":
    #video_file = "../FF++/fake/01_02__outside_talking_still_laughing__YVGY8LOK.mp4"
    video_file = "../FF++_complete/test/000.mp4"

    # Duration of the video:
    fh = av.open(video_file)
    video = fh.streams.video[0]
    duration_seconds = float(video.duration * video.time_base)
    print(f"Video Duration: {duration_seconds}")

    
    speed_test(video_file)