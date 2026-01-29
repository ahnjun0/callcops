# CallCops Models
# ===============
# Real-Time Audio Watermarking Neural Network Components

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
    # Legacy compatibility
    RTAWNet,
    RTAWEncoder,
    RTAWDecoder,
    MultiResolutionDiscriminator,
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
    # Legacy
    "RTAWNet",
    "RTAWEncoder",
    "RTAWDecoder",
    "MultiResolutionDiscriminator",
]
