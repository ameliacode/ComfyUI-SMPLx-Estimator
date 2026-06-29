"""Package entrypoint for the EditPose custom node."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_CLASSES

WEB_DIRECTORY = "./js"

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClickPose": "ClickPose",
    "MotionAGFormer": "MotionAGFormer",
    "3D Pose Editor": "3D Pose Editor",
    "SMPLXFit": "SMPL-X Fit",
    "SMPLXEditor": "SMPL-X Editor",
    "WholeBodyHandDetector": "Whole-Body Hand Detector",
    "NLFSMPLXEstimator": "NLF SMPL-X Estimator",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NODE_CLASSES",
    "WEB_DIRECTORY",
]

try:
    from comfy_env import wrap_nodes
except ImportError:
    wrap_nodes = None


if wrap_nodes is not None:
    wrap_nodes()
