# CallCops Models v2.0
# ====================
# Frame-Wise Real-Time Audio Watermarking Neural Network Components

from .rtaw_net import (
    # Main classes
    CallCopsNet,
    Encoder,
    Decoder,
    Discriminator,
    # Building blocks
    ConvBlock,
    ResidualBlock,
    SEBlock,
    FrameWiseFusionBlock,
    FrameWiseBitExtractor,
    # Constants
    FRAME_SAMPLES,
    PAYLOAD_LENGTH,
    CYCLE_SAMPLES,
    SAMPLE_RATE,
    # Utilities
    align_to_frames,
    get_cyclic_bit_indices,
)

from .codec_simulator import (
    DifferentiableCodecSimulator,
    G711Simulator,
    G729Simulator,
)

from .losses import (
    CallCopsLoss,
    CallShieldLoss,
    MultiResolutionMelLoss,
    BitAccuracyLoss,
    AdversarialLoss,
)

__all__ = [
    # Main network
    "CallCopsNet",
    "Encoder",
    "Decoder",
    "Discriminator",
    # Building blocks
    "ConvBlock",
    "ResidualBlock",
    "SEBlock",
    "FrameWiseFusionBlock",
    "FrameWiseBitExtractor",
    # Constants
    "FRAME_SAMPLES",
    "PAYLOAD_LENGTH",
    "CYCLE_SAMPLES",
    "SAMPLE_RATE",
    # Utilities
    "align_to_frames",
    "get_cyclic_bit_indices",
    # Codec simulation
    "DifferentiableCodecSimulator",
    "G711Simulator",
    "G729Simulator",
    # Losses
    "CallCopsLoss",
    "CallShieldLoss",
    "MultiResolutionMelLoss",
    "BitAccuracyLoss",
    "AdversarialLoss",
]
