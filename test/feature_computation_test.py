import sys
import pytest
import numpy as np
import torch
import tools.feature_computation as featureComputation

# --- TEST compute_features_frame ---
def test_compute_features_frame_standardization_and_mask():
    # Create a small 2x2 example
    mv_x = np.array([[1.0, 2.0],
                     [3.0, 4.0]], dtype=np.float32)
    mv_y = np.array([[5.0, 6.0],
                     [7.0, 8.0]], dtype=np.float32)
    # Define im mask: 0 = inter-coded (to normalize), 1 = intra-coded
    im   = np.array([[0, 1],
                     [0, 1]], dtype=np.uint8)

    # Expected standardization for mv_x on positions where im==0: values [1,3]
    # mean_x = 2.0, std_x = 1.0
    # standardized mv_x at (0,0) = (1-2)/1 = -1, at (1,0) = (3-2)/1 = 1
    # other positions (im==1) should be set to 0

    # Expected for mv_y on positions where im==0: values [5,7]
    # mean_y = 6.0, std_y = 1.0
    # standardized mv_y at (0,0) = (5-6)/1 = -1, at (1,0) = (7-6)/1 = 1
    # other positions set to 0

    features = featureComputation.compute_features_frame(mv_x, mv_y, im)
    assert isinstance(features, torch.Tensor)
    # shape should be (H, W, 3)
    assert features.shape == (2, 2, 3)

    # Convert to numpy for inspection
    feat_np = features.numpy()

    # Check standardized values
    # Channel 0 = mv_x
    assert np.isclose(feat_np[0,0,0], -1.0)
    assert np.isclose(feat_np[1,0,0],  1.0)
    assert np.isclose(feat_np[0,1,0],  0.0)
    assert np.isclose(feat_np[1,1,0],  0.0)

    # Channel 1 = mv_y
    assert np.isclose(feat_np[0,0,1], -1.0)
    assert np.isclose(feat_np[1,0,1],  1.0)
    assert np.isclose(feat_np[0,1,1],  0.0)
    assert np.isclose(feat_np[1,1,1],  0.0)

    # Channel 2 = im (converted to float)
    assert np.isclose(feat_np[0,0,2], 0.0)
    assert np.isclose(feat_np[0,1,2], 1.0)
    assert np.isclose(feat_np[1,0,2], 0.0)
    assert np.isclose(feat_np[1,1,2], 1.0)

# --- TEST compute_features_video_tensor ---
def test_compute_features_video_tensor_stack():
    # Use two identical frames
    mv_x1 = np.zeros((3,3), dtype=np.float32)
    mv_y1 = np.zeros((3,3), dtype=np.float32)
    im1   = np.zeros((3,3), dtype=np.uint8)
    # second frame with ones
    mv_x2 = np.ones((3,3), dtype=np.float32)
    mv_y2 = np.ones((3,3), dtype=np.float32)
    im2   = np.ones((3,3), dtype=np.uint8)

    # Compute video tensor
    video_tensor = featureComputation.compute_features_video_tensor([mv_x1, mv_x2], [mv_y1, mv_y2], [im1, im2])
    assert isinstance(video_tensor, torch.Tensor)
    # shape should be (num_frames, H, W, 3)
    assert video_tensor.shape == (2, 3, 3, 3)

    # Frame 0 should be standardized zeros (std=0->set to zero) and im zeros
    frame0 = video_tensor[0].numpy()
    # mv channels should be zero
    assert np.allclose(frame0[:,:,0], 0.0)
    assert np.allclose(frame0[:,:,1], 0.0)
    assert np.allclose(frame0[:,:,2], 0.0)

    # Frame 1 mv channels: original ones mask==0 gives mean=1,std=0->set zeros
    frame1 = video_tensor[1].numpy()
    assert np.allclose(frame1[:,:,0], 0.0)
    assert np.allclose(frame1[:,:,1], 0.0)
    assert np.allclose(frame1[:,:,2], 1.0)

# --- MAIN ENTRYPOINT ---
if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '--disable-warnings']))
