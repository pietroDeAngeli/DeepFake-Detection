import numpy as np
import cv2
import pytest
from av.sidedata.sidedata import Type
import tools.motion_vectors as motionVectos
import sys

# --- STUBS ---
class DummyFaceBox:
    def __init__(self, x, y, side):
        self.x = x
        self.y = y
        self.side = side

class DummyFrame:
    def __init__(self, pict_type, mv_list):
        self.pict_type = pict_type
        # side_data.get(Type.MOTION_VECTORS) deve restituire mv_list
        class SD:
            def __init__(self, mv_list): self._mv = mv_list
            def get(self, key):
                if key is Type.MOTION_VECTORS: return self._mv
                return None
        self.side_data = SD(mv_list)

# --- TEST I-FRAME ---
def test_iframe_zero_maps():
    f = DummyFrame(pict_type=0, mv_list=None)
    mvx, mvy, im = motionVectors.extract_motion_vectors_and_im([f], [None], mb_size=16, out_res=32)[0]
    assert mvx.shape == (32, 32)
    assert np.all(mvx == 0)
    assert np.all(mvy == 0)
    assert np.all(im  == 0)

# --- TEST P-FRAME SEMPLICE ---
def test_pframe_single_mv():
    face = DummyFaceBox(x=0, y=0, side=4)
    # un solo mv nella macroblock (0,0)
    mv = {'dst_x': 1, 'dst_y': 1, 'motion_x': 8, 'motion_y': 16, 'motion_scale': 2}
    f = DummyFrame(pict_type=1, mv_list=[mv])
    mvx, mvy, im = motionVectors.extract_motion_vectors_and_im([f], [face], mb_size=2, out_res=4)[0]
    # Macroblock 2x2 in alto a sinistra → ridimensionato 1:1 perché out_res == side
    assert np.all(mvx[0:2,0:2] == 4.0)
    assert np.all(mvy[0:2,0:2] == 8.0)
    assert np.all(im [0:2,0:2] == 0)
    # resto deve rimanere a valore di default
    assert np.all(im [2:, :] == 1)
    assert np.all(im [:, 2:] == 1)

# --- TEST MotionVector.is_in_face ---
def test_is_in_face_inside():
    face = DummyFaceBox(x=10, y=20, side=16)
    mv = motionVectors.MotionVector(None, None, None, None, None, dst_x=18, dst_y=28, motion_x=0, motion_y=0, motion_scale=1)
    assert mv.is_in_face(face)

def test_is_in_face_outside():
    face = DummyFaceBox(x=0, y=0, side=10)
    mv = motionVectors.MotionVector(None, None, None, None, None, dst_x=20, dst_y=20, motion_x=0, motion_y=0, motion_scale=1)
    assert not mv.is_in_face(face)

# --- MAIN ENTRYPOINT ---
if __name__ == '__main__':
    # Run pytest on this file
    sys.exit(pytest.main([__file__, '-q', '--disable-warnings']))