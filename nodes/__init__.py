from .multihmr_nodes import MultiHMREstimator
from .smplx_nodes import SMPLXEditor

NODE_CLASSES = [
    MultiHMREstimator,
    SMPLXEditor,
]

NODE_CLASS_MAPPINGS = {
    "MultiHMREstimator": MultiHMREstimator,
    "SMPLXEditor": SMPLXEditor,
}
