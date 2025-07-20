import sys
import os
import pytest
import av
import cv2
import tools.face_detection as faceDetection

# --- FIXTURE: Initialize detector once per module ---
@pytest.fixture(scope="module")
def detector():
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'models', 'face_detection_yunet_2023mar.onnx')
    )
    det = faceDetection.initialize_detector(model_path)
    assert det is not None, "Detector should be initialized and not None"
    return det

# --- TEST: initialize_detector ---
def test_initialize_detector(detector):
    # Fixture ensures initialization; this test just documents success
    assert detector is not None

# --- TEST: make_square_box boundary conditions ---
@pytest.mark.parametrize("x, y, w, h, img_w, img_h", [
    (100, 150, 200, 300, 1920, 1080),    # normal portrait
    (5, 10, 100, 50,   1920, 1080),      # near left/top
    (1800, 900, 200, 300, 1920, 1080),   # near right/bottom
    (300, 400, 150, 150, 1920, 1080),    # already square
])
def test_make_square_box(x, y, w, h, img_w, img_h):
    new_x, new_y, side = faceDetection.make_square_box(x, y, w, h, img_w, img_h)
    # The square must stay inside the image bounds
    assert 0 <= new_x <= img_w - side, f"new_x {new_x} out of bounds"
    assert 0 <= new_y <= img_h - side, f"new_y {new_y} out of bounds"
    assert side > 0, "side length must be positive"

# --- TEST: extract_frames_with_faces produces correct types ---
def test_extract_frames_with_faces(detector, tmp_path):
    video_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'input2.mp4')
    )

    # Extract a small number of frames
    results = faceDetection.extract_frames_with_faces(
        detector=detector,
        video_path=video_path,
        n_frames=10
    )
    assert results is not None, "Result should not be None"
    # Validate returned tuples and save images
    for idx, (frame, face) in enumerate(results):
        assert isinstance(frame, av.video.frame.VideoFrame), "Frame should be a VideoFrame"
        # Face object should have an image attribute
        assert hasattr(face, 'image'), "Face should have .image attribute"
        # Save the detected face crop for manual inspection
        out_file = f"temp/face_{idx}.png"
        cv2.imwrite(str(out_file), face.image)

# --- MAIN ENTRYPOINT ---
if __name__ == '__main__':
    # Run pytest on this file
    sys.exit(pytest.main([__file__, '-q', '--disable-warnings']))
