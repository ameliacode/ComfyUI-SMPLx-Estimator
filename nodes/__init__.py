from .nlf_nodes import NLFSMPLXEstimator
from .multihmr_nodes import MultiHMREstimator
from .smplx_nodes import SMPLXEditor

NODE_CLASSES = [
    NLFSMPLXEstimator,
    MultiHMREstimator,
    SMPLXEditor,
]

NODE_CLASS_MAPPINGS = {
    "NLFSMPLXEstimator": NLFSMPLXEstimator,
    "MultiHMREstimator": MultiHMREstimator,
    "SMPLXEditor": SMPLXEditor,
}
