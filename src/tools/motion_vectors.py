from av.sidedata.sidedata import Type
import numpy as np
import cv2

# Custom 
from tools.face_detection import FaceBox

class MotionVector:
    def __init__(self, source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale):
        self.source = source
        self.w = w
        self.h = h
        self.src_x = src_x
        self.src_y = src_y
        self.dst_x = dst_x
        self.dst_y = dst_y
        self.motion_x = motion_x
        self.motion_y = motion_y
        self.motion_scale = motion_scale

    def is_in_face(self, face: FaceBox) -> bool:
        """
        Check if the motion vector is within the bounds of the given face.
        dst is the center of the macroblock, so we adjust by 8 pixels to match the face box.
        8 because macroblocks are 16x16 pixels, and to get the top-left corner we subtract half the size.
        
        Parameters:
            face (Face): The face object to check against.
        
        Returns:
            bool: True if the motion vector is within the face bounds, False otherwise.
        """
        return face.x <= self.dst_x - 8 < face.x + face.side and face.y <= self.dst_y - 8 < face.y + face.side

def extract_motion_vectors_and_im(
    frames,
    faces: list[FaceBox | None],
    mb_size: int = 16,
    out_res: int = 224
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract per-frame motion-vector maps (mvx, mvy) and information mask (im)
    from decoded frames and corresponding face boxes, correctly handling
    face crops whose side is not a multiple of mb_size by preserving
    partial-block pixel sizes.

    Parameters:
        frames (list[VideoFrame]): List of decoded video frames.
        faces (list[FaceBox | None]): List of face boxes corresponding to each frame.
        mb_size (int): Size of the macroblocks in pixels (default is 16).
        out_res (int): Output resolution for the motion vector maps (default is 224).
    Returns:
        list[tuple[np.ndarray, np.ndarray, np.ndarray]]: List of tuples containing:
            - mvx (np.ndarray): Motion vector x-component map.
            - mvy (np.ndarray): Motion vector y-component map.
            - im (np.ndarray): Information mask, where 0 = inter-coded, 1 = intra-coded.
    """
    results = []

    for frame, face in zip(frames, faces):
        # 1) I-frames → no temporal info → all zeros
        if frame.pict_type == 0:
            mvx = np.zeros((out_res, out_res), dtype=np.float32)
            mvy = np.zeros((out_res, out_res), dtype=np.float32)
            im  = np.zeros((out_res, out_res), dtype=np.uint8)
            results.append((mvx, mvy, im))
            continue

        # 3) Grab motion-vectors in one call
        mv_data = frame.side_data.get(Type.MOTION_VECTORS)
        if mv_data is None:
            raise RuntimeError("Motion Vectors are not available")
        mv_list = mv_data  # iterable of dicts

        # 4) Prepare per-pixel face-crop maps so partial macroblocks keep correct size
        N = face.side  # may not be divisible by mb_size
        mvx_crop = np.zeros((N, N), dtype=np.float32)
        mvy_crop = np.zeros((N, N), dtype=np.float32)
        im_crop  = np.ones ((N, N), dtype=np.uint8)  # 1 = intra-coded

        # 5) Fill each macroblock region in the pixel map
        for mv in mv_list:
            # map absolute dst coords → coords relative to face crop
            rel_x = mv.dst_x - face.x
            rel_y = mv.dst_y - face.y

            # skip vectors outside the face square
            if not (0 <= rel_x < N and 0 <= rel_y < N):
                continue

            # determine macroblock indices (floor division)
            mb_i = int(rel_y) // mb_size
            mb_j = int(rel_x) // mb_size

            # compute pixel-region bounds for this macroblock
            y0 = mb_i * mb_size
            y1 = min((mb_i + 1) * mb_size, N)
            x0 = mb_j * mb_size
            x1 = min((mb_j + 1) * mb_size, N)

            # scaled motion-vector components
            mvx_val = mv.motion_x / mv.motion_scale
            mvy_val = mv.motion_y / mv.motion_scale

            # fill the region in the crop maps
            mvx_crop[y0:y1, x0:x1] = mvx_val
            mvy_crop[y0:y1, x0:x1] = mvy_val

            # mark this block as inter-coded (has MV)
            im_crop [y0:y1, x0:x1] = 0

        # 6) Upsample crop maps to fixed network input resolution
        mvx = cv2.resize(mvx_crop, (out_res, out_res), interpolation=cv2.INTER_LINEAR)
        mvy = cv2.resize(mvy_crop, (out_res, out_res), interpolation=cv2.INTER_LINEAR)
        im  = cv2.resize(im_crop , (out_res, out_res), interpolation=cv2.INTER_NEAREST)

        results.append((mvx, mvy, im))

    return results
