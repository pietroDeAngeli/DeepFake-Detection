import numpy as np
import math

def compute_features_frame(mv: list):
    """
    Extracts statistical motion vector features from a single frame.
    
    Args:
        mv (list[MotionVector]): List of motion vectors for a single frame.
    
    Returns:
        dict: Dictionary containing frame-level motion features.
    """
    dx_list = []
    dy_list = []
    magnitude_list = []
    angle_bins = [0] * 8  # 8 directional bins (45° each)
    non_zero_count = 0

    for v in mv:
        if v.motion_scale == 0:
            continue  # avoid division by zero

        dx = v.motion_x / v.motion_scale
        dy = v.motion_y / v.motion_scale

        dx_list.append(dx)
        dy_list.append(dy)
        magnitude_list.append(math.sqrt(dx**2 + dy**2))

        if dx != 0 or dy != 0:
            non_zero_count += 1
            angle = math.atan2(dy, dx)
            bin_index = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
            angle_bins[bin_index] += 1

    n = len(mv)
    if n == 0:
        # Return zeros if no valid motion vectors
        return {
            "mean_dx": 0.0,
            "mean_dy": 0.0,
            "var_dx": 0.0,
            "var_dy": 0.0,
            "mean_magnitude": 0.0,
            "zero_mv_percent": 1.0,
            "direction_entropy": 0.0,
            "motion_density": 0.0
        }

    dx_array = np.array(dx_list)
    dy_array = np.array(dy_list)
    magnitude_array = np.array(magnitude_list)

    mean_dx = np.mean(dx_array) if dx_array.size > 0 else 0.0
    mean_dy = np.mean(dy_array) if dy_array.size > 0 else 0.0
    var_dx = np.var(dx_array) if dx_array.size > 0 else 0.0
    var_dy = np.var(dy_array) if dy_array.size > 0 else 0.0
    mean_magnitude = np.mean(magnitude_array) if magnitude_array.size > 0 else 0.0
    zero_mv_percent = 1 - (non_zero_count / n)

    # Direction entropy (in bits)
    total_bins = sum(angle_bins)
    direction_entropy = 0.0
    if total_bins > 0:
        probs = [count / total_bins for count in angle_bins if count > 0]
        direction_entropy = -sum(p * math.log2(p) for p in probs)

    motion_density = non_zero_count / n

    return {
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "var_dx": var_dx,
        "var_dy": var_dy,
        "mean_magnitude": mean_magnitude,
        "zero_mv_percent": zero_mv_percent,
        "direction_entropy": direction_entropy,
        "motion_density": motion_density
    }

def compute_features_video(video_mv: list[list]):
    """
    Computes average motion features for a video by aggregating frame-level features.
    
    Args:
        video_mv (list[list[MotionVector]]): A list of frames, each containing a list of motion vectors.
    
    Returns:
        dict: Dictionary containing average video-level motion features.
    """
    frame_features = []

    for frame_mv in video_mv:
        frame_feat = compute_features_frame(frame_mv)
        frame_features.append(frame_feat)

    if not frame_features:
        return {
            "mean_dx": 0.0,
            "mean_dy": 0.0,
            "var_dx": 0.0,
            "var_dy": 0.0,
            "mean_magnitude": 0.0,
            "zero_mv_percent": 1.0,
            "direction_entropy": 0.0,
            "motion_density": 0.0
        }

    # Aggregate by averaging each key
    keys = frame_features[0].keys()
    mean_features = {
        key: np.mean([f[key] for f in frame_features]) for key in keys
    }

    return mean_features



    
    

