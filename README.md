# CallCops

**한국어 전화 통화 인증을 위한 실시간 오디오 워터마킹 시스템**

CallCops은 전화망(8kHz)에서 동작하는 딥러닝 기반 오디오 워터마킹 시스템입니다. 사람이 인지할 수 없는 128-bit 워터마크를 통화 음성에 삽입하고, G.729 코덱 압축 후에도 5% 미만의 비트 오류율로 추출할 수 있습니다.

## 주요 특징

- **실시간 처리**: Causal Convolution 기반 < 200ms 지연
- **고음질 유지**: PESQ ≥ 4.0 (MOS 스케일)
- **코덱 강건성**: G.711/G.729 압축 후 BER < 5%
- **모바일 지원**: Android Lite Interpreter 최적화 (< 10MB)

## 프로젝트 구조

```
call/
├── configs/
│   └── default.yaml        # 학습 설정
├── models/
│   ├── rtaw_net.py         # 인코더/디코더/판별기
│   ├── attention.py        # 마스킹 기반 어텐션
│   ├── codec_simulator.py  # 미분 가능 코덱 시뮬레이터
│   └── losses.py           # 복합 손실 함수
├── scripts/
│   ├── train.py            # 학습 스크립트
│   ├── evaluate.py         # 평가 스크립트
│   ├── dataset.py          # 데이터 파이프라인
│   └── export_mobile.py    # 모바일 변환
├── utils/
│   ├── audio_utils.py      # 오디오 처리 유틸리티
│   └── metrics.py          # 평가 메트릭
├── data/
│   ├── train/              # 학습 데이터 (*.wav)
│   ├── val/                # 검증 데이터
│   └── test/               # 테스트 데이터
├── checkpoints/            # 모델 체크포인트
├── logs/                   # TensorBoard 로그
└── exported/               # 모바일 내보내기 결과
```

## 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/callcops.git
cd callcops

# 의존성 설치
pip install -r requirements.txt
```

### 필수 의존성

- Python >= 3.10
- PyTorch >= 2.4.0
- torchaudio >= 2.4.0

## 데이터 준비

`data/train/`, `data/val/`, `data/test/` 폴더에 8kHz WAV 파일을 넣으세요.

**지원 형식**: `.wav`, `.flac`, `.mp3`, `.ogg`

```bash
data/
├── train/
│   ├── call_001.wav
│   ├── call_002.wav
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

> **참고**: 다른 샘플레이트의 오디오도 자동으로 8kHz로 리샘플링됩니다.

## 사용법

### 1. 학습

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir data/train \
    --val_dir data/val \
    --epochs 100
```

**체크포인트에서 재개:**
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir data/train \
    --resume checkpoints/checkpoint_epoch50.pt
```

### 2. 평가

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data/test \
    --output results/evaluation.yaml
```

### 3. 모바일 내보내기

```bash
python scripts/export_mobile.py \
    --checkpoint checkpoints/best_model.pt \
    --output_dir exported \
    --formats torchscript lite onnx \
    --benchmark
```

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        Training Loop                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Audio ──► RTAWEncoder ──► Watermarked ──► CodecSim ──┐   │
│   [B,1,320]     │            [B,1,320]      (G.711/729) │   │
│                 │                                       │   │
│                 ├── CausalConv1d                        │   │
│                 ├── MaskingAttention                    │   │
│                 └── Perturbation (< 1%)                 │   │
│                                                         ▼   │
│   Bits ◄────── RTAWDecoder ◄────────── Degraded Audio  │   │
│   [B,128]           │                                       │
│                     ├── TemporalAttention                   │
│                     └── BitClassifier                       │
│                                                             │
│   Loss = λ_bit·BCE + λ_audio·Mel + λ_adv·GAN               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 품질 목표

| 메트릭 | 목표 | 설명 |
|--------|------|------|
| PESQ | ≥ 4.0 | 음질 보존 (MOS 스케일) |
| BER (G.729) | < 5% | 코덱 압축 후 비트 오류율 |
| 지연 | < 200ms | 실시간 처리 |
| 모델 크기 | < 10MB | 모바일 배포 |

## 설정

`configs/default.yaml`에서 주요 하이퍼파라미터를 조정할 수 있습니다:

```yaml
audio:
  sample_rate: 8000       # 전화망 표준
  frame_ms: 40            # 40ms 프레임

watermark:
  payload_length: 128     # 128-bit 페이로드

training:
  batch_size: 32
  learning_rate: 0.0001
  lambda_bit: 1.0         # 비트 손실 가중치
  lambda_audio: 10.0      # 오디오 품질 가중치
  lambda_adv: 0.1         # GAN 손실 가중치
```

## TensorBoard 모니터링

```bash
tensorboard --logdir logs/
```

## 라이선스

MIT License