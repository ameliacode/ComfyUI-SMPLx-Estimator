from .editpose_nodes import ClickPose, MotionAGFormer, Pose3DEditor
from .smplx_nodes import SMPLXFit, SMPLXEditor
from .wholebody_nodes import WholeBodyHandDetector
from .nlf_nodes import NLFSMPLXEstimator

NODE_CLASSES = [
    ClickPose,
    MotionAGFormer,
    Pose3DEditor,
    SMPLXFit,
    SMPLXEditor,
    WholeBodyHandDetector,
    NLFSMPLXEstimator,
]

NODE_CLASS_MAPPINGS = {
    "ClickPose": ClickPose,
    "MotionAGFormer": MotionAGFormer,
    "3D Pose Editor": Pose3DEditor,
    "SMPLXFit": SMPLXFit,
    "SMPLXEditor": SMPLXEditor,
    "WholeBodyHandDetector": WholeBodyHandDetector,
    "NLFSMPLXEstimator": NLFSMPLXEstimator,
}
