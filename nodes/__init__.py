from .model_loaders import LoadSMPLX, LoadNLF, LoadMultiHMR, LoadWiLoR, LoadSMIRK
from .nlf_nodes import NLFSMPLXEstimator
from .multihmr_nodes import MultiHMREstimator
from .wilor_nodes import WiLoRHandEstimator
from .smirk_nodes import SMIRKFaceEstimator
from .smplx_nodes import SMPLXEditor
from .export_nodes import ExportMesh

NODE_CLASSES = [
    LoadSMPLX, LoadNLF, LoadMultiHMR, LoadWiLoR, LoadSMIRK,
    NLFSMPLXEstimator,
    MultiHMREstimator,
    WiLoRHandEstimator,
    SMIRKFaceEstimator,
    SMPLXEditor,
    ExportMesh,
]

NODE_CLASS_MAPPINGS = {
    "LoadSMPLX": LoadSMPLX,
    "LoadNLF": LoadNLF,
    "LoadMultiHMR": LoadMultiHMR,
    "LoadWiLoR": LoadWiLoR,
    "LoadSMIRK": LoadSMIRK,
    "NLFSMPLXEstimator": NLFSMPLXEstimator,
    "MultiHMREstimator": MultiHMREstimator,
    "WiLoRHandEstimator": WiLoRHandEstimator,
    "SMIRKFaceEstimator": SMIRKFaceEstimator,
    "SMPLXEditor": SMPLXEditor,
    "ExportMesh": ExportMesh,
}
