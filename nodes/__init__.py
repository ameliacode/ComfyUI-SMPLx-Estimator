from .model_loaders import (
    LoadNLF, DownloadAndLoadNLF,
    LoadMultiHMR, DownloadAndLoadMultiHMR,
    LoadWiLoR, DownloadAndLoadWiLoR,
)
from .nlf_nodes import NLFSMPLXEstimator
from .multihmr_nodes import MultiHMREstimator
from .wilor_nodes import WiLoRHandEstimator
from .smplx_nodes import SMPLXEditor

NODE_CLASSES = [
    LoadNLF, DownloadAndLoadNLF,
    LoadMultiHMR, DownloadAndLoadMultiHMR,
    LoadWiLoR, DownloadAndLoadWiLoR,
    NLFSMPLXEstimator,
    MultiHMREstimator,
    WiLoRHandEstimator,
    SMPLXEditor,
]

NODE_CLASS_MAPPINGS = {
    "LoadNLF": LoadNLF,
    "DownloadAndLoadNLF": DownloadAndLoadNLF,
    "LoadMultiHMR": LoadMultiHMR,
    "DownloadAndLoadMultiHMR": DownloadAndLoadMultiHMR,
    "LoadWiLoR": LoadWiLoR,
    "DownloadAndLoadWiLoR": DownloadAndLoadWiLoR,
    "NLFSMPLXEstimator": NLFSMPLXEstimator,
    "MultiHMREstimator": MultiHMREstimator,
    "WiLoRHandEstimator": WiLoRHandEstimator,
    "SMPLXEditor": SMPLXEditor,
}
