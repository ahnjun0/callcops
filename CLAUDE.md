# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CallCops

CallCops is a real-time audio watermarking system for Korean telephony authentication. It embeds imperceptible 128-bit watermarks into 8kHz call audio while maintaining PESQ ≥ 4.0 and achieving BER < 5% after G.729 codec compression.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Training (default paths: data/raw/training, data/raw/validation)
python scripts/train.py \
    --config configs/default.yaml \
    --epochs 100

# Training with custom paths
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir data/raw/training \
    --val_dir data/raw/validation \
    --epochs 100

# Resume training from checkpoint
python scripts/train.py \
    --resume checkpoints/checkpoint_epoch50.pt

# Evaluation (PESQ, BER, codec robustness)
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data/raw/validation \
    --output results/evaluation.yaml

# Export for Android (TorchScript Lite, ONNX, quantized)
python scripts/export_mobile.py \
    --checkpoint checkpoints/best_model.pt \
    --output_dir exported \
    --formats torchscript lite quantized onnx \
    --benchmark

# Code formatting
black models/ scripts/ utils/
isort models/ scripts/ utils/

# Run tests
pytest
```

## Architecture

### Neural Network Pipeline

```
Audio [B,1,320] ──► RTAWEncoder ──► Watermarked [B,1,320]
                      │
                      ├── CausalConv1d Stack (no future frame reference)
                      ├── MaskingAwareAttention (embed in high-energy regions)
                      └── Perturbation Scaling (< 1% amplitude change)

Watermarked ──► CodecSimulator ──► Degraded ──► RTAWDecoder ──► Bits [B,128]
                (G.711/G.729)                       │
                                                   └── TemporalAttention + Classifier
```

### Key Design Decisions

1. **Causal Convolutions**: All Conv1d layers use left-padding only to enable real-time streaming with < 200ms latency
2. **Straight-Through Estimator**: Codec quantization uses STE for gradient flow through non-differentiable operations
3. **Curriculum Learning**: Codec augmentation difficulty increases progressively during training
4. **Multi-Resolution Loss**: Mel-spectrogram loss at multiple FFT sizes (64, 128, 256) for perceptual quality

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `models/rtaw_net.py` | Main encoder/decoder/discriminator networks |
| `models/attention.py` | Masking-aware attention for imperceptible embedding |
| `models/codec_simulator.py` | Differentiable G.711 (μ/A-law) and G.729 simulation |
| `models/losses.py` | Combined loss: `λ_bit·L_BCE + λ_audio·L_Mel + λ_adv·L_GAN` |
| `scripts/dataset.py` | 8kHz audio loading, 40ms framing, real-time augmentation |

## Audio Specifications

- **Sample Rate**: 8kHz (telephony standard)
- **Frame Size**: 40ms (320 samples)
- **Payload**: 128-bit cyclic (1 bit per frame)
- **Bandwidth**: 300-3400Hz (telephony band)

## Configuration

All hyperparameters are in `configs/default.yaml`. Key sections:
- `audio`: Sample rate, frame size, bit depth
- `watermark`: Payload length, sync pattern
- `model.encoder/decoder`: Channel dimensions, kernel sizes, attention heads
- `training`: Learning rate, loss weights (λ_bit, λ_audio, λ_adv)
- `codec`: Supported codecs and simulation parameters
- `augmentation`: Noise SNR range, bandpass settings

## Quality Targets

| Metric | Target | Measured By |
|--------|--------|-------------|
| PESQ | ≥ 4.0 | `compute_pesq_batch()` |
| BER (G.729) | < 5% | `compute_ber()` |
| Latency | < 200ms | `model.estimate_latency_ms()` |
| Model Size | < 10MB | Mobile export |
