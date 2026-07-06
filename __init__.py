"""Package entrypoint for the EditPose custom node."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_CLASSES

WEB_DIRECTORY = "./js"

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSMPLX": "Load SMPLx",
    "LoadNLF": "Load NLF",
    "LoadMultiHMR": "Load Multi-HMR",
    "LoadWiLoR": "Load WiLoR",
    "LoadSMIRK": "Load SMIRK",
    "NLFSMPLXEstimator": "Body: NLF",
    "MultiHMREstimator": "Full Body: Multi-HMR",
    "WiLoRHandEstimator": "Hand: WiLoR",
    "SMIRKFaceEstimator": "Face: SMIRK",
    "SMPLXEditor": "SMPL-X Editor",
    "ExportMesh": "Export Mesh",
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
