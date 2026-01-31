# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CallCops

CallCops is a real-time audio watermarking system for Korean telephony authentication. It embeds imperceptible 128-bit watermarks into 8kHz call audio while maintaining PESQ ≥ 4.0 and achieving BER < 5% after G.729 codec compression.

## Current Status & Issues (Updated: 2026-01-31)

### 🔴 Active Training Issue
**Problem**: Train/Val loss increases from epoch 4 onwards
- Epoch 4: Train=5.25, Val=10.06, BER=5.46%, SNR=17.2dB
- Pattern: Very fast initial convergence then divergence

**Suspected Causes**:
1. **Overfitting to small dataset**: Model fits training data in 3 epochs
2. **Loss weight imbalance**: lambda_l1=30 may be too aggressive
3. **Learning rate too high**: 5e-5 may cause oscillation after convergence
4. **Codec simulation**: Enabled codec may introduce instability

**Potential Solutions**:
1. Learning rate scheduler (cosine annealing, reduce on plateau)
2. Early stopping with patience
3. Data augmentation (noise, time stretch, pitch shift)
4. Reduce lambda_l1 from 30 to 10-15
5. Increase dataset diversity

### 📊 Training History

| Attempt | Config | Result | Notes |
|---------|--------|--------|-------|
| v1 | lambda_bit=50, audio=0.01, no codec | Very high SNR (100dB) | Mode collapse, no watermark |
| v2 | `Clamped Alpha` | Forces perturbation [0.6, 1.0] | `rtaw_net.py:421` | Still high SNR |
| v3 | Curriculum learning 3-phase | Unstable | Loss oscillation |
| v4 | Frame-wise encoder (40ms/bit) | Better BER | Architecture change |
| v5 | lambda_l1=50, audio=10, L1 loss | Loss diverged | L1 too strong |
| v6 | lambda_l1=30, lr=5e-5, codec on | Loss diverges epoch 4 | Current attempt |

---

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Training with current config
python scripts/train.py --config configs/default.yaml --epochs 100

# Resume training
python scripts/train.py --resume checkpoints/latest.pth

# Background training with nohup
nohup stdbuf -oL python3 scripts/train.py \
    --config configs/default.yaml \
    --epochs 100 \
    > training.log 2>&1 &

# Monitor training
tail -f training.log

# Evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/raw/validation

# ONNX Export (for Web/Mobile)
python scripts/export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir exported/onnx \
    --quantize --validate

# Code formatting
black models/ scripts/ utils/
isort models/ scripts/ utils/
```

## Architecture (v2.0 Frame-Wise)

### Core Design: 40ms Frame = 1 Bit

```
Audio [B,1,T] + Message [B,128]
         │
         ▼
    ┌─────────────────────────────────────────┐
    │              Encoder v2.0                │
    ├─────────────────────────────────────────┤
    │  Audio Encoder (CausalConv1d Stack)      │
    │         │                                │
    │         ▼                                │
    │  FrameWiseFusion                         │
    │  - T/320 frames                          │
    │  - Frame i → Bit (i % 128)               │
    │  - Localized bit injection               │
    │         │                                │
    │         ▼                                │
    │  Residual Refinement                     │
    │         │                                │
    │         ▼                                │
    │  Output: α × tanh(perturbation)          │
    │  α ∈ [0.01, 0.3] (learnable, clamped)    │
    └─────────────────────────────────────────┘
         │
         ▼
Watermarked [B,1,T] = Original + Perturbation
         │
         ▼ (Optional: Codec Simulation)
    ┌─────────────────────────────────────────┐
    │              Decoder v2.0                │
    ├─────────────────────────────────────────┤
    │  Feature Extraction                      │
    │         │                                │
    │         ▼                                │
    │  FrameWiseBitExtractor                   │
    │  - Extract 1 bit per 40ms frame          │
    │  - Output: [B, num_frames] probabilities │
    │         │                                │
    │         ▼                                │
    │  Alpha [0.6, 1.0] (Train: 1.0, Eval: 0.6)│
    │  - Prevents spikes by enforcing smooth   │
    │    deformation over larger amplitude     │
    │         │                                │
    │         ▼                                │
    │  128-bit Aggregation                     │
    │  - Average bits with same (i % 128)      │
    └─────────────────────────────────────────┘
         │
         ▼
    Bits [B,128] (probabilities 0-1)
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `models/rtaw_net.py` | Encoder, Decoder, Discriminator, FrameWiseFusion |
| `models/codec_simulator.py` | Differentiable G.711, G.729, AMR-NB simulation |
| `models/losses.py` | CallCopsLoss: BCE + Mel + STFT + L1 + Adversarial |
| `scripts/train.py` | Training loop with GAN, AMP, checkpointing |
| `scripts/export_onnx.py` | ONNX export with INT8 quantization |
| `scripts/dataset.py` | 8kHz audio loading, PayloadGenerator |
| `utils/messenger.py` | Telegram notifications |

## Loss Configuration

### Current Settings (configs/default.yaml)

```yaml
lambda_bit: 2.0      # Reduced to prioritize audio quality first
lambda_audio: 50.0   # Significantly increased for smooth perturbations
lambda_stft: 10.0    # Increased for spectral fidelity
lambda_l1: 1.0       # Reduced to avoid conflict with Mel loss
lambda_adv: 0.05     # GAN adversarial loss
lambda_det: 0.1      # Detection confidence loss
```

### Loss Function Components

| Loss | Formula | Purpose |
|------|---------|---------|
| L_BCE | BCE(pred_bits, true_bits) | Bit accuracy |
| L_Mel | L1(mel(pred), mel(orig)) | Perceptual quality |
| L_STFT | Multi-res STFT difference | Spectral quality |
| L_L1 | L1(watermarked, original) | Direct waveform SNR |
| L_Adv | GAN generator loss | Natural audio |
| L_Det | BCE(detection, 1) | Watermark presence |

**Total = λ_bit × L_BCE + λ_audio × L_Mel + λ_stft × L_STFT + λ_l1 × L_L1 + λ_adv × L_Adv + λ_det × L_Det**

## Web Demo (callcops-preview)

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| EmbedPanel | `components/EmbedPanel.jsx` | Watermark embedding UI |
| RealtimeEmbedDemo | `components/RealtimeEmbedDemo.jsx` | Live mic streaming demo |
| RealtimeOscilloscope | `components/RealtimeOscilloscope.jsx` | Real-time waveform + spectrogram |
| BitMatrixView | `components/BitMatrixView.jsx` | 128-bit visualization |
| WaveformView | `components/WaveformView.jsx` | WaveSurfer.js integration |
| ProgressiveDetection | `components/ProgressiveDetection.jsx` | Animated bit reveal |
| CRCVerificationPanel | `components/CRCVerificationPanel.jsx` | CRC-8 verification |
| AudioComparisonPanel | `components/AudioComparisonPanel.jsx` | SNR/PSNR metrics |

### ONNX Models

| Model | Path | Size | Purpose |
|-------|------|------|---------|
| Encoder INT8 | `public/models/encoder_int8.onnx` | ~2MB | Watermark embedding |
| Decoder INT8 | `public/models/decoder_int8.onnx` | ~2MB | Watermark extraction |

## Audio Specifications

- **Sample Rate**: 8kHz (telephony standard)
- **Frame Size**: 40ms (320 samples)
- **Payload**: 128-bit cyclic message
- **Cycle Length**: 5.12 seconds (128 × 40ms)
- **Bandwidth**: 300-3400Hz (telephony band)

## Quality Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| PESQ | ≥ 4.0 | Not yet achieved |
| SNR | > 30dB | ~17dB (epoch 4) |
| BER | < 5% | ~5.5% (epoch 4) |
| BER (G.729) | < 5% | Not tested |
| Latency | < 200ms | Achieved |

## Known Issues & Solutions

### 1. Mode Collapse (Very High SNR, No Watermark)
**Cause**: Model learns to output zero perturbation
**Solution**: Clamped alpha [0.01, 0.3], bit-priority phase

### 2. Loss Divergence After Few Epochs
**Cause**: Learning rate too high, loss imbalance
**Solution**: LR scheduler, reduce lambda_l1

### 3. Small Audio Volume After Watermarking
**Cause**: Clipping from perturbation addition
**Solution**: RMS normalization after watermarking

### 4. WebAudio 8kHz Resampling
**Note**: Web demo uses OfflineAudioContext for resampling (polyphase interpolation)

## Recommended Next Steps

1. **Add LR Scheduler**: ReduceLROnPlateau or CosineAnnealing
2. **Data Augmentation**: Time stretch, pitch shift, room impulse
3. **Reduce lambda_l1**: Try 10-15 instead of 30
4. **Early Stopping**: Patience 10-20 epochs
5. **Larger Dataset**: More diverse audio samples
6. **Gradient Clipping**: Already at 1.0, may need lower

## Development Notes

### 2026-01-31 Session Summary

**Web Demo Enhancements:**
1. RealtimeEmbedDemo - Live mic streaming with chunked processing
2. RealtimeOscilloscope - Canvas-based waveform + FFT spectrogram
3. BitMatrixView progressive reveal - Syncs with audio playback
4. ProgressiveDetection - Animated bit scan effect
5. CRC-8 verification - Error detection and correction
6. AudioComparisonPanel - SNR, PSNR, correlation metrics

**Model Changes:**
1. Added L1 waveform loss for direct SNR optimization
2. Enabled codec simulation (G.711, G.729, AMR-NB)
3. Adjusted alpha range to [0.01, 0.3]
4. Frame-wise architecture (40ms = 1 bit)

**Configuration Changes:**
- learning_rate: 5e-5
- lambda_l1: 30.0 (may be too high)
- codec.enabled: true

## Monitoring

```bash
# TensorBoard
tensorboard --logdir logs/

# Watch training log
tail -f training.log | grep -E "(Epoch|Loss|BER|SNR)"

# Telegram Notifications (requires .env)
# TELEGRAM_BOT_TOKEN=xxx
# TELEGRAM_CHAT_ID=xxx
```
