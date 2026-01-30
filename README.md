# CallCops

**한국어 전화 통화 인증을 위한 실시간 오디오 워터마킹 시스템**

CallCops은 전화망(8kHz)에서 동작하는 딥러닝 기반 오디오 워터마킹 시스템입니다. 사람이 인지할 수 없는 128-bit 워터마크를 통화 음성에 삽입하고, G.729 코덱 압축 후에도 5% 미만의 비트 오류율로 추출할 수 있습니다.

## 주요 특징

- **실시간 처리**: Causal Convolution 기반 < 200ms 지연
- **고음질 유지**: PESQ ≥ 4.0 (MOS 스케일)
- **코덱 강건성**: G.711/G.729 압축 후 BER < 5%
- **모바일 지원**: ONNX Runtime Web/Mobile 최적화 (< 10MB)

## 프로젝트 구조

```
call/
├── configs/
│   └── default.yaml          # 학습 설정 (Curriculum Learning)
├── models/
│   ├── rtaw_net.py           # Encoder/Decoder/Discriminator
│   ├── codec_simulator.py    # 미분 가능 코덱 시뮬레이터
│   ├── losses.py             # 복합 손실 함수
│   └── __init__.py           # 모듈 export
├── scripts/
│   ├── train.py              # 학습 스크립트
│   ├── evaluate.py           # 평가 스크립트
│   ├── dataset.py            # 데이터 파이프라인
│   ├── export_onnx.py        # ONNX 변환 (Web/Mobile)
│   └── export_mobile.py      # TorchScript 변환
├── utils/
│   ├── audio_utils.py        # 오디오 처리 유틸리티
│   ├── metrics.py            # 평가 메트릭 (PESQ, BER, SNR)
│   └── messenger.py          # Telegram 알림
├── data/
│   └── raw/                  # AI Hub 데이터셋
├── checkpoints/              # 모델 체크포인트
├── exported/                 # ONNX/TorchScript 출력
└── logs/                     # TensorBoard 로그
```

## 설치

```bash
git clone https://github.com/your-repo/callcops.git
cd callcops
pip install -r requirements.txt
```

### 필수 의존성

- Python >= 3.10
- PyTorch >= 2.4.0
- torchaudio >= 2.4.0

## 사용법

### 1. 학습 (Curriculum Learning)

```bash
# Phase 1: Bit 우선 학습 (Mode Collapse 방지)
python scripts/train.py --config configs/default.yaml --epochs 20

# Background 학습
nohup stdbuf -oL python3 scripts/train.py \
    --config configs/default.yaml \
    --epochs 20 > training.log 2>&1 &

# 체크포인트에서 재개
python scripts/train.py --resume checkpoints/latest.pth
```

### 2. 평가

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/raw/validation
```

### 3. ONNX 내보내기 (Web/Mobile)

```bash
`python scripts/export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir exported/onnx \
    --quantize --validate`
```

출력:
```
exported/onnx/
├── encoder.onnx        # FP32 Encoder
├── encoder_int8.onnx   # INT8 Quantized
├── decoder.onnx        # FP32 Decoder
└── decoder_int8.onnx   # INT8 Quantized
```

## 아키텍처 (v2.0)

```
┌─────────────────────────────────────────────────────────────┐
│                     CallCopsNet                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Audio [B,1,T] + Message [B,128]                          │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────────────────────────────┐                  │
│   │  Encoder                             │                  │
│   │  ├─ CausalConv1d + SEBlock          │                  │
│   │  ├─ CrossModalFusionBlock           │ ← Linear O(T)    │
│   │  └─ Clamped Alpha [0.01, 0.3]       │ ← Mode Collapse 방지│
│   └─────────────────────────────────────┘                  │
│         │                                                   │
│         ▼                                                   │
│   Watermarked [B,1,T] → CodecSimulator (G.711/G.729)       │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────────────────────────────┐                  │
│   │  Decoder                             │                  │
│   │  ├─ CausalConv1d                    │                  │
│   │  └─ TemporalBitExtractor            │ ← 시간 정보 보존  │
│   └─────────────────────────────────────┘                  │
│         │                                                   │
│         ▼                                                   │
│   Bits [B,128]                                             │
│                                                             │
│   Loss = λ_bit·BCE + λ_audio·Mel + λ_stft·STFT + λ_adv·GAN │
└─────────────────────────────────────────────────────────────┘
```

## Curriculum Learning

| Phase | Epochs | λ_bit | λ_audio | λ_adv | 목표 |
|-------|--------|-------|---------|-------|------|
| 1 | 1-20 | 50.0 | 0.01 | 0.001 | SNR 30-40dB, BER < 0.3 |
| 2 | 21-40 | 20.0 | 0.5 | 0.05 | SNR 40-45dB, BER < 0.1 |
| 3 | 41+ | 10.0 | 2.0 | 0.2 | SNR > 45dB, BER < 0.05 |

## 품질 목표

| 메트릭 | 목표 | 설명 |
|--------|------|------|
| PESQ | ≥ 4.0 | 음질 보존 (MOS 스케일) |
| BER (G.729) | < 5% | 코덱 압축 후 비트 오류율 |
| 지연 | < 200ms | 실시간 처리 |
| 모델 크기 | < 10MB | ONNX INT8 양자화 |

## 모니터링

```bash
# TensorBoard
tensorboard --logdir logs/

# Telegram 알림 (.env 설정 필요)
# TELEGRAM_BOT_TOKEN=xxx
# TELEGRAM_CHAT_ID=xxx
```

## Demo

Web Demo (ONNX Runtime Web): [callcops-demo](https://github.com/your-repo/callcops-demo)

## 라이선스

MIT License