# callcops 데이터 구조 명세

## 1. 개요
- **데이터셋**: AI Hub 저음질 전화망 음성인식 데이터 (Dataset Key: 571)
- **선택 도메인**: D03 (HR/채용/컨설팅)
- **포맷**: 8kHz/16kHz, 16bit, Mono, PCM/WAV

## 2. 디렉토리 구조
~/callcops/data/raw/
├── training/              # 학습용 세션 (TS_D03 기반)
│   ├── S001342/           # 개별 통화 세션 폴더 (Session ID)
│   │   ├── S001342.json   # 통합 메타데이터 (화자, 성별, 나이, 발화 리스트)
│   │   ├── 0001.wav       # 발화 단위 오디오
│   │   ├── 0001.txt       # 발화 단위 전사 텍스트
│   │   └── ...
│   └── ...
└── validation/            # 검증용 세션 (VS_D03 기반)
    ├── S002202/
    │   ├── S002202.json
    │   ├── 0001.wav
    │   ├── 0001.txt
    │   └── ...
    └── ...

## 3. 주요 메타데이터 (JSON 필드)
- `dataSet.typeInfo.speakers`: 화자 타입(상담사/고객), 성별, 나이, 전화망 종류
- `dataSet.dialogs`: 오디오 파일 경로(`audioPath`), 발화 길이(`duration`), 전사 내용(`text`)