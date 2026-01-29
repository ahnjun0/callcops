# CallCops Models
# ====================
# Real-Time Audio Watermarking Neural Network Components

from .rtaw_net import RTAWEncoder, RTAWDecoder, RTAWNet
from .attention import MaskingAwareAttention, TemporalAttention
from .codec_simulator import DifferentiableCodecSimulator, G711Simulator, G729Simulator
from .losses import CallCopsLoss, MultiResolutionMelLoss

__all__ = [
    "RTAWEncoder",
    "RTAWDecoder",
    "RTAWNet",
    "MaskingAwareAttention",
    "TemporalAttention",
    "DifferentiableCodecSimulator",
    "G711Simulator",
    "G729Simulator",
    "CallCopsLoss",
    "MultiResolutionMelLoss",
]
