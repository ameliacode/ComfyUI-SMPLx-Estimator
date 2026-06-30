from .model_loaders import LoadNLF, LoadMultiHMR, LoadWiLoR
from .nlf_nodes import NLFSMPLXEstimator
from .multihmr_nodes import MultiHMREstimator
from .wilor_nodes import WiLoRHandEstimator
from .smplx_nodes import SMPLXEditor
from .export_nodes import ExportMesh

NODE_CLASSES = [
    LoadNLF, LoadMultiHMR, LoadWiLoR,
    NLFSMPLXEstimator,
    MultiHMREstimator,
    WiLoRHandEstimator,
    SMPLXEditor,
    ExportMesh,
]

NODE_CLASS_MAPPINGS = {
    "LoadNLF": LoadNLF,
    "LoadMultiHMR": LoadMultiHMR,
    "LoadWiLoR": LoadWiLoR,
    "NLFSMPLXEstimator": NLFSMPLXEstimator,
    "MultiHMREstimator": MultiHMREstimator,
    "WiLoRHandEstimator": WiLoRHandEstimator,
    "SMPLXEditor": SMPLXEditor,
    "ExportMesh": ExportMesh,
}
