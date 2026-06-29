from .nlf_nodes import NLFSMPLXEstimator
from .wholebody_nodes import WholeBodyHandDetector
from .smplx_nodes import SMPLXEditor

NODE_CLASSES = [
    NLFSMPLXEstimator,
    WholeBodyHandDetector,
    SMPLXEditor,
]

NODE_CLASS_MAPPINGS = {
    "NLFSMPLXEstimator": NLFSMPLXEstimator,
    "WholeBodyHandDetector": WholeBodyHandDetector,
    "SMPLXEditor": SMPLXEditor,
}
