from .nlf_nodes import NLFSMPLXEstimator
from .multihmr_nodes import MultiHMREstimator
from .wilor_nodes import WiLoRHandEstimator
from .smplx_nodes import SMPLXEditor

NODE_CLASSES = [
    NLFSMPLXEstimator,
    MultiHMREstimator,
    WiLoRHandEstimator,
    SMPLXEditor,
]

NODE_CLASS_MAPPINGS = {
    "NLFSMPLXEstimator": NLFSMPLXEstimator,
    "MultiHMREstimator": MultiHMREstimator,
    "WiLoRHandEstimator": WiLoRHandEstimator,
    "SMPLXEditor": SMPLXEditor,
}
