import numpy as np
import torch

import numpy as np
import torch

def compute_features_frame(
    mv_x: np.ndarray,
    mv_y: np.ndarray,
    im:    np.ndarray
) -> torch.Tensor:
    """
    Standardizza mv_x e mv_y solo sui pixel inter-coded (im == 0),
    mantiene im come maschera binaria, e restituisce un tensor
    shape (H, W, 3).
    """
    # Creiamo una maschera booleana per i blocchi inter-coded
    mask = (im == 0)

    # Estraiamo solo i valori inter-coded
    mvx_vals = mv_x[mask]
    mvy_vals = mv_y[mask]

    # Calcoliamo media e std solo su quei valori
    mean_x = mvx_vals.mean() if mvx_vals.size else 0.0
    std_x  = mvx_vals.std()  if mvx_vals.size else 1.0
    mean_y = mvy_vals.mean() if mvy_vals.size else 0.0
    std_y  = mvy_vals.std()  if mvy_vals.size else 1.0

    # Standardizziamo l'intera mappa, ma la scala è determinata
    # solo su mask=True
    mv_x_std = (mv_x - mean_x) / (std_x + 1e-6)
    mv_y_std = (mv_y - mean_y) / (std_y + 1e-6)

    # Rimettiamo a zero i pixel non-inter (opzionale, ma per chiarezza)
    mv_x_std[~mask] = 0.0
    mv_y_std[~mask] = 0.0

    # Impiliamo i tre canali (due MV + IM)
    features = np.stack((mv_x_std, mv_y_std, im.astype(np.float32)), axis=-1)
    return torch.from_numpy(features)  # dtype=torch.float32


def compute_features_video_tensor(mv_x: list[np.ndarray], mv_y: list[np.ndarray], im: list[np.ndarray]) -> list[torch.Tensor]:
    """
    Computes a feature tensor for a video from lists of motion vectors and information masks.

    Parameters:
        mv_x (list[np.ndarray]): List of motion vector x-component maps.
        mv_y (list[np.ndarray]): List of motion vector y-component maps.
        im (list[np.ndarray]): List of information masks.

    Returns:
        np.ndarray: Feature tensor for the video, shape (num_frames, height, width, 3).
    """
    features = []
    for x, y, mask in zip(mv_x, mv_y, im):
        features.append(compute_features_frame(x, y, mask))
    
    return torch.stack(features)
