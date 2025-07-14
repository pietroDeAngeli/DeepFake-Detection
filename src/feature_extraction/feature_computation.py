import numpy as np
import math

def compute_features_frame(mv: list):
    """
    Extracts statistical motion vector features from a single frame.
    
    Args:
        mv (list[MotionVector] | None): List of motion vectors for 
            the frame, or None if no motion vectors are available.
    Returns:
        dict: Dictionary with motion vector features:
    """
    # I-frame or no motion vectors
    defaults = {
        "mean_dx": 0.0,
        "mean_dy": 0.0,
        "var_dx": 0.0,
        "var_dy": 0.0,
        "mean_magnitude": 0.0,
        "zero_mv_percent": 1.0,
        "direction_entropy": 0.0,
        "motion_density": 0.0
    }

    # No MV
    if mv is None or len(mv) == 0:
        return defaults

    # Remove None values
    filtered = [v for v in mv if v is not None]
    if not filtered:
        return defaults

    dx_list = []
    dy_list = []
    magnitude_list = []
    angle_bins = [0] * 8  # 8 directions for angles
    non_zero_count = 0
    n = len(filtered)

    for v in filtered:
        # avoid division by zero

        dx = v.motion_x / v.motion_scale
        dy = v.motion_y / v.motion_scale
        dx_list.append(dx)
        dy_list.append(dy)
        magnitude_list.append(math.hypot(dx, dy))

        # count non-zero vectors and calculate angle
        if dx != 0 or dy != 0:
            non_zero_count += 1
            angle = math.atan2(dy, dx)
            bin_index = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
            angle_bins[bin_index] += 1

    # array numpy
    dx_arr = np.array(dx_list)
    dy_arr = np.array(dy_list)
    mag_arr = np.array(magnitude_list)

    mean_dx       = float(np.mean(dx_arr)) if dx_arr.size       > 0 else 0.0
    mean_dy       = float(np.mean(dy_arr)) if dy_arr.size       > 0 else 0.0
    var_dx        = float(np.var(dx_arr))  if dx_arr.size       > 0 else 0.0
    var_dy        = float(np.var(dy_arr))  if dy_arr.size       > 0 else 0.0
    mean_magnitude= float(np.mean(mag_arr))if mag_arr.size      > 0 else 0.0
    zero_mv_percent = 1.0 - (non_zero_count / n)
    motion_density  = non_zero_count / n

    # Angular entropy
    total_bins = sum(angle_bins)
    direction_entropy = 0.0
    if total_bins > 0:
        probs = [c / total_bins for c in angle_bins if c > 0]
        direction_entropy = -sum(p * math.log2(p) for p in probs)

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


def compute_features_video(video_mv: list):
    """
    Computes average motion features for a video aggregando i frame-level features.
    
    Args:
        video_mv (list[list[MotionVector]]): Frame-level motion vectors for the video.
    
    Returns:
        dict: Dictionary with average motion features for the video.
    """
    # feature evaluation for each frame
    video_features = [compute_features_frame(fm) for fm in video_mv]

    # se non ci sono frame restituisco default
    if not video_features:
        return compute_features_frame(None)

    # average on frame-level features
    mean_features = {
        key: float(np.mean([f[key] for f in video_features]))
        for key in video_features[0].keys()
    }

    return mean_features
