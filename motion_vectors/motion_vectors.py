import av
from av.sidedata.sidedata import Type

# Custom 
from ..face_detection.face_detection_tools import FaceBox

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
        
        Parameters:
            face (Face): The face object to check against.
        
        Returns:
            bool: True if the motion vector is within the face bounds, False otherwise.
        """
        return face.x <= self.src_x < face.x + face.side and face.y <= self.src_y < face.y + face.side


def extract_motion_vectors(video_path, faces):
    """
    Extract motion vectors of the faces from a video file and return them as a list of dictionaries.
    
    Parameters:
        video_path (str): Path to the video file.
        faces (list of Face): List of face frames extracted from the video.
    Returns:
        list of dict: A list where each dictionary contains motion vectors for a corresponding face frame.
    """
    # Open the video container and get the video stream
    container = av.open(video_path)
    stream = container.streams.video[0]

    # set the options to export motion vectors
    stream.codec_context.options = {"flags2": "+export_mvs"}

    all_flows = []

    # Demux and decode the video stream
    n_frame = 0
    for packet in container.demux(stream):
        for frame in packet.decode():

            vectors = []
            ts = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0

            mv_data = frame.side_data.get(Type.MOTION_VECTORS)
            if mv_data:
                for mv in mv_data:
                    vector = MotionVector(
                        source=mv.source,
                        w=mv.w,
                        h=mv.h,
                        src_x=mv.src_x,
                        src_y=mv.src_y,
                        dst_x=mv.dst_x,
                        dst_y=mv.dst_y,
                        motion_x=mv.motion_x,
                        motion_y=mv.motion_y,
                        motion_scale=mv.motion_scale
                    )

                    # Check if the vector is within the bounds of the corresponding face
                    if vector.is_in_face(faces[n_frame]):
                        vectors.append(vector)
            
            all_flows.append({"time": ts, "motion_vectors": vectors})
            n_frame += 1

    return all_flows