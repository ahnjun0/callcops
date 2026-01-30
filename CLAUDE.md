# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CallCops

CallCops is a real-time audio watermarking system for Korean telephony authentication. It embeds imperceptible 128-bit watermarks into 8kHz call audio while maintaining PESQ ≥ 4.0 and achieving BER < 5% after G.729 codec compression.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Training (Phase 1: Bit-priority for mode collapse prevention)
python scripts/train.py \
    --config configs/default.yaml \
    --epochs 20

# Resume training
python scripts/train.py --resume checkpoints/latest.pth

# Background training with nohup
nohup stdbuf -oL python3 scripts/train.py \
    --config configs/default.yaml \
    --epochs 20 \
    > training_phase1.log 2>&1 &

# Evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/raw/validation

# ONNX Export (for Web/Mobile)
python scripts/export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir exported/onnx \
    --quantize --validate

# Mobile Export (TorchScript)
python scripts/export_mobile.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir exported \
    --formats torchscript lite quantized onnx

# Code formatting
black models/ scripts/ utils/
isort models/ scripts/ utils/
```

## Architecture

### Neural Network Pipeline (v2.0)

```
Audio [B,1,T] + Message [B,128]
         │
         ▼
    ┌─────────────────────────────────────────┐
    │              Encoder                     │
    ├─────────────────────────────────────────┤
    │  CausalConv1d Stack → SEBlock            │
    │         │                                │
    │         ▼                                │
    │  CrossModalFusionBlock (Linear Attention)│
    │  - Audio [B,C,T] attends to Message [B,C]│
    │  - O(T) complexity (not O(T²))           │
    │  - Bit pattern modulation                │
    │         │                                │
    │         ▼                                │
    │  Clamped Alpha [0.01, 0.3]               │
    │  - Prevents mode collapse                │
    └─────────────────────────────────────────┘
         │
         ▼
Watermarked [B,1,T] → CodecSimulator (G.711/G.729)
         │
         ▼
    ┌─────────────────────────────────────────┐
    │              Decoder                     │
    ├─────────────────────────────────────────┤
    │  CausalConv1d Stack                      │
    │         │                                │
    │         ▼                                │
    │  TemporalBitExtractor                    │
    │  - Preserves temporal info (not pool)    │
    │  - Segment-wise bit extraction           │
    │  - 128-bit classification                │
    └─────────────────────────────────────────┘
         │
         ▼
    Bits [B,128]
```

### Key Architecture Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `CrossModalFusionBlock` | O(T) Linear Attention for audio-message fusion | `rtaw_net.py:146` |
| `TemporalBitExtractor` | Segment-wise 128-bit extraction | `rtaw_net.py:237` |
| `Clamped Alpha` | Forces perturbation [0.01, 0.3] | `rtaw_net.py:304` |
| `SEBlock` | Squeeze-Excitation with LeakyReLU | `rtaw_net.py:124` |

### Key Design Decisions

1. **Clamped Alpha**: Learnable perturbation scale bounded to [0.01, 0.3] prevents mode collapse (100dB SNR issue)
2. **Linear Attention**: O(T) complexity instead of O(T²) - Audio attends to global message vector, not T×T self-attention
3. **LeakyReLU(0.2)**: Replaces ReLU in SEBlock to preserve negative watermark signals
4. **Temporal Bit Extraction**: Preserves time-domain info instead of global average pooling
5. **Curriculum Learning**: 3-phase training with decreasing bit weight, increasing audio weight

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `models/rtaw_net.py` | Encoder, Decoder, Discriminator, CrossModalFusionBlock, TemporalBitExtractor |
| `models/codec_simulator.py` | Differentiable G.711 (μ/A-law) and G.729 simulation |
| `models/losses.py` | `CallCopsLoss`: BCE + Mel + STFT + Adversarial + Detection |
| `scripts/train.py` | Training loop with GAN updates, AMP, checkpointing |
| `scripts/export_onnx.py` | ONNX export with INT8 quantization for Web/Mobile |
| `scripts/dataset.py` | 8kHz audio loading, augmentation, PayloadGenerator |
| `utils/messenger.py` | Telegram notifications for remote training monitoring |

## Curriculum Learning (Mode Collapse Prevention)

```yaml
# Phase 1 (Epoch 1-20): Bit Priority
lambda_bit: 50.0
lambda_audio: 0.01
lambda_adv: 0.001
# Target: SNR 30-40dB, BER < 0.3

# Phase 2 (Epoch 21-40): Balanced
lambda_bit: 20.0
lambda_audio: 0.5
lambda_adv: 0.05
# Target: SNR 40-45dB, BER < 0.1

# Phase 3 (Epoch 41+): Quality Focus
lambda_bit: 10.0
lambda_audio: 2.0
lambda_adv: 0.2
# Target: SNR > 45dB, BER < 0.05
```

## Audio Specifications

- **Sample Rate**: 8kHz (telephony standard)
- **Frame Size**: 40ms (320 samples) or variable length
- **Payload**: 128-bit message
- **Bandwidth**: 300-3400Hz (telephony band)

## Configuration

All hyperparameters are in `configs/default.yaml`. Key sections:
- `audio`: Sample rate, frame size, bit depth
- `watermark`: Payload length (128-bit)
- `model`: Channel dimensions [32, 64, 128, 256], residual blocks
- `training`: Learning rate, loss weights (λ_bit, λ_audio, λ_adv, λ_det, λ_stft)
- `codec`: G.711/G.729 simulation parameters
- `augmentation`: Noise SNR range, bandpass settings

## Quality Targets

| Metric | Target | Measured By |
|--------|--------|-------------|
| PESQ | ≥ 4.0 | `compute_pesq_batch()` |
| BER (G.729) | < 5% | `compute_ber()` |
| Latency | < 200ms | Causal convolutions |
| Model Size | < 10MB | ONNX INT8 quantization |

## Monitoring

```bash
# TensorBoard
tensorboard --logdir logs/

# Telegram Notifications (requires .env)
# TELEGRAM_BOT_TOKEN=xxx
# TELEGRAM_CHAT_ID=xxx
```
