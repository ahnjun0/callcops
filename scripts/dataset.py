"""
CallCops: Data Pipeline
=======================

AI Hub 저음질 전화망 음성인식 데이터셋(Dataset 571) 로딩 파이프라인.

데이터 구조:
    data/raw/training/
    ├── S001342/
    │   ├── S001342.json   # 메타데이터 (speakers, dialogs)
    │   ├── 0001.wav       # 발화 단위 오디오
    │   └── ...
    └── ...

주요 기능:
- JSON 메타데이터에서 화자 타입(상담사/고객) 매핑
- 8kHz 정규화 (16kHz → 8kHz 리샘플링)
- 3초 미만 오디오 필터링
- 실시간 증강: 대역 통과 필터, 가우시안 노이즈
"""

import json
import random
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np


# =============================================================================
# Audio Augmentation
# =============================================================================

class AudioAugmentation:
    """
    실시간 오디오 증강 파이프라인
    ============================

    전화망 환경을 시뮬레이션하는 증강 기법들.

    증강 기법:
    1. 대역 통과 필터 (300-3400Hz)
    2. 가우시안 노이즈 (SNR 15-40dB)
    3. 음량 변화 (±6dB)
    4. 클리핑 (과부하 시뮬레이션)
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        bandpass_low: int = 300,
        bandpass_high: int = 3400,
        snr_range: Tuple[float, float] = (15, 40),
        volume_range: Tuple[float, float] = (-6, 6),
        prob_bandpass: float = 0.5,
        prob_noise: float = 0.5,
        prob_volume: float = 0.3,
        prob_clip: float = 0.1
    ):
        self.sample_rate = sample_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.snr_range = snr_range
        self.volume_range = volume_range

        self.prob_bandpass = prob_bandpass
        self.prob_noise = prob_noise
        self.prob_volume = prob_volume
        self.prob_clip = prob_clip

        self._init_bandpass_filter()

    def _init_bandpass_filter(self):
        """대역 통과 필터 초기화 (전화망 표준: 300-3400Hz)"""
        nyquist = self.sample_rate / 2
        low_norm = self.bandpass_low / nyquist
        high_norm = self.bandpass_high / nyquist

        filter_len = 65
        t = torch.arange(filter_len) - filter_len // 2
        t = t.float()
        t[filter_len // 2] = 1e-10

        sinc_high = torch.sin(2 * np.pi * high_norm * t) / (np.pi * t)
        sinc_low = torch.sin(2 * np.pi * low_norm * t) / (np.pi * t)

        sinc_high[filter_len // 2] = 2 * high_norm
        sinc_low[filter_len // 2] = 2 * low_norm

        bp_filter = sinc_high - sinc_low
        window = torch.hamming_window(filter_len)
        bp_filter = bp_filter * window

        self.bandpass_filter = bp_filter / bp_filter.sum()

    def apply_bandpass(self, audio: torch.Tensor) -> torch.Tensor:
        """대역 통과 필터 적용"""
        if random.random() > self.prob_bandpass:
            return audio

        squeeze_output = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
            squeeze_output = True

        filter_kernel = self.bandpass_filter.view(1, 1, -1).to(audio.device)
        padding = filter_kernel.shape[-1] // 2

        filtered = F.conv1d(audio, filter_kernel, padding=padding)

        if squeeze_output:
            filtered = filtered.squeeze(0)

        return filtered

    def add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """가우시안 노이즈 추가"""
        if random.random() > self.prob_noise:
            return audio

        snr_db = random.uniform(*self.snr_range)
        signal_power = torch.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(audio) * torch.sqrt(noise_power + 1e-10)

        return audio + noise

    def adjust_volume(self, audio: torch.Tensor) -> torch.Tensor:
        """음량 조절"""
        if random.random() > self.prob_volume:
            return audio

        db_change = random.uniform(*self.volume_range)
        gain = 10 ** (db_change / 20)

        return audio * gain

    def apply_clipping(self, audio: torch.Tensor) -> torch.Tensor:
        """클리핑 (과부하 시뮬레이션)"""
        if random.random() > self.prob_clip:
            return audio

        threshold = random.uniform(0.7, 0.95)
        return torch.clamp(audio, -threshold, threshold)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """전체 증강 파이프라인 적용"""
        audio = self.apply_bandpass(audio)
        audio = self.add_noise(audio)
        audio = self.adjust_volume(audio)
        audio = self.apply_clipping(audio)
        audio = torch.clamp(audio, -1.0, 1.0)

        return audio


# =============================================================================
# Payload Generator (워터마크용)
# =============================================================================

class PayloadGenerator:
    """
    워터마크 페이로드 생성기 (128-bit Cyclic Payload)

    구조:
    - 16-bit 동기화 패턴 (고정)
    - 32-bit 타임스탬프
    - 64-bit 인증 데이터
    - 16-bit CRC 체크섬
    """

    def __init__(
        self,
        payload_length: int = 128,
        sync_bits: int = 16,
        timestamp_bits: int = 32,
        auth_bits: int = 64,
        crc_bits: int = 16
    ):
        self.payload_length = payload_length
        self.sync_bits = sync_bits
        self.timestamp_bits = timestamp_bits
        self.auth_bits = auth_bits
        self.crc_bits = crc_bits

        self.sync_pattern = torch.tensor(
            [int(b) for b in "1010101010101010"],
            dtype=torch.float32
        )

    def generate(self, batch_size: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """랜덤 페이로드 생성"""
        if seed is not None:
            torch.manual_seed(seed)

        payloads = []
        for _ in range(batch_size):
            sync = self.sync_pattern.clone()
            timestamp = torch.randint(0, 2, (self.timestamp_bits,), dtype=torch.float32)
            auth = torch.randint(0, 2, (self.auth_bits,), dtype=torch.float32)
            crc = self._compute_crc(torch.cat([sync, timestamp, auth]))
            payload = torch.cat([sync, timestamp, auth, crc])
            payloads.append(payload)

        return torch.stack(payloads)

    def _compute_crc(self, data: torch.Tensor) -> torch.Tensor:
        """간단한 CRC 계산 (XOR 기반)"""
        crc = torch.zeros(self.crc_bits, dtype=torch.float32)
        for i, bit in enumerate(data):
            crc[i % self.crc_bits] = (crc[i % self.crc_bits] + bit) % 2
        return crc

    def verify_sync(self, payload: torch.Tensor) -> bool:
        """동기화 패턴 검증"""
        extracted_sync = payload[:self.sync_bits]
        return torch.all(extracted_sync == self.sync_pattern).item()


# =============================================================================
# Main Dataset: CallCopsDataset
# =============================================================================

class CallCopsDataset(Dataset):
    """
    CallCops 메인 데이터셋
    ======================

    AI Hub 저음질 전화망 음성인식 데이터셋(Dataset 571) 로딩.

    Args:
        data_dir: 데이터 디렉토리 (data/raw/training 또는 data/raw/validation)
        sample_rate: 목표 샘플링 레이트 (8kHz)
        min_duration: 최소 오디오 길이 (초)
        augmentation: 오디오 증강 파이프라인
        mode: "train" 또는 "val"

    Returns:
        (audio_tensor, text, speaker_type) 튜플
        - audio_tensor: [1, T] 정규화된 오디오
        - text: 전사 텍스트
        - speaker_type: 화자 타입 ("상담사" 또는 "고객")
    """

    # 화자 타입 매핑 (내부 정규화용)
    SPEAKER_TYPE_MAP = {
        "상담사": "counselor",
        "고객": "customer",
        "counselor": "counselor",
        "customer": "customer",
    }

    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_rate: int = 8000,
        min_duration: float = 3.0,
        max_duration: Optional[float] = None,
        augmentation: Optional[AudioAugmentation] = None,
        mode: str = "train",
        verbose: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.augmentation = augmentation if mode == "train" else None
        self.mode = mode

        # 데이터 항목 리스트
        self.items: List[Dict[str, Any]] = []

        # 데이터 로드
        self._load_dataset()

        if verbose:
            self._print_stats()

    def _load_dataset(self):
        """
        세션 폴더를 순회하며 JSON 메타데이터를 파싱하고 데이터 항목 리스트 생성.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        session_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        if not session_dirs:
            print(f"[Warning] No session directories found in {self.data_dir}")
            return

        for session_dir in session_dirs:
            # JSON 메타데이터 파일 찾기
            json_files = list(session_dir.glob("*.json"))

            if not json_files:
                continue

            json_path = json_files[0]  # 세션당 하나의 JSON 파일

            try:
                metadata = self._parse_json(json_path)
            except Exception as e:
                print(f"Warning: Failed to parse {json_path}: {e}")
                continue

            # 화자 ID → 타입 매핑 생성
            speaker_map = self._build_speaker_map(metadata)

            # dialogs 처리
            dialogs = self._get_dialogs(metadata)

            for dialog in dialogs:
                item = self._process_dialog(dialog, session_dir, speaker_map)

                if item is not None:
                    self.items.append(item)

    def _parse_json(self, json_path: Path) -> Dict[str, Any]:
        """JSON 파일 파싱"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_speaker_map(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        화자 ID → 화자 타입 매핑 생성

        JSON 구조:
        dataSet.typeInfo.speakers 리스트 참조.
        "type" 값이 "상담사1", "고객1" 등으로 되어 있어 부분 문자열 매칭 필요.
        """
        speaker_map = {}
        speakers = []

        # JSON 구조 접근: dataSet -> typeInfo -> speakers
        if "dataSet" in metadata:
            dataset = metadata["dataSet"]
            if "typeInfo" in dataset and "speakers" in dataset["typeInfo"]:
                speakers = dataset["typeInfo"]["speakers"]
            # Fallback for flexibility
            elif "speakers" in dataset:
                speakers = dataset["speakers"]
        elif "speakers" in metadata:
            speakers = metadata["speakers"]

        for speaker in speakers:
            speaker_id = speaker.get("id") or speaker.get("speaker_id")
            speaker_type = speaker.get("type") or speaker.get("speaker_type", "unknown")

            if speaker_id:
                # 부분 문자열 매칭으로 타입 정규화
                if "상담사" in speaker_type:
                    normalized_type = "counselor"
                elif "고객" in speaker_type:
                    normalized_type = "customer"
                else:
                    normalized_type = self.SPEAKER_TYPE_MAP.get(speaker_type, "unknown")
                
                speaker_map[speaker_id] = normalized_type

        return speaker_map

    def _get_dialogs(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        dialogs 리스트 추출
        JSON 구조: dataSet -> dialogs
        """
        if "dataSet" in metadata:
            dataset = metadata["dataSet"]
            if "dialogs" in dataset:
                return dataset["dialogs"]
            elif "utterances" in dataset:
                return dataset["utterances"]
        elif "dialogs" in metadata:
            return metadata["dialogs"]
        
        return []

    def _process_dialog(
        self,
        dialog: Dict[str, Any],
        session_dir: Path,
        speaker_map: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """
        개별 dialog 항목 처리

        이슈 해결:
        1. 경로 불일치: JSON의 'audioPath'는 디렉토리 포함(D03/...). 
           실제 파일은 session_dir 바로 아래에 있음. -> 파일명만 추출.
        2. 화자 키 불일치: 'speaker_id' 대신 'speaker' 키 사용.
        3. 안정성: duration float 변환.
        """
        # 1. 오디오 경로 처리
        raw_audio_path = dialog.get("audioPath") or dialog.get("audio_path")
        if not raw_audio_path:
            return None
        
        # 파일명만 추출 (예: "D03/.../0001.wav" -> "0001.wav")
        filename = Path(raw_audio_path).name
        full_audio_path = session_dir / filename

        # 파일 존재 확인
        if not full_audio_path.exists():
            # 디버깅용: 원래 경로로도 한번 체크 (혹시 구조가 맞을 수도 있으니)
            if (session_dir / raw_audio_path).exists():
                full_audio_path = session_dir / raw_audio_path
            else:
                return None

        # 2. duration 안전하게 가져오기
        duration = dialog.get("duration", 0)
        if isinstance(duration, str):
            try:
                duration = float(duration)
            except ValueError:
                duration = 0

        # 최소 길이 필터링
        if duration < self.min_duration:
            return None

        # 최대 길이 필터링
        if self.max_duration and duration > self.max_duration:
            return None

        # 텍스트
        text = dialog.get("text") or dialog.get("transcription", "")

        # 3. 화자 ID 가져오기 (speaker 키 우선)
        speaker_id = dialog.get("speaker") or dialog.get("speaker_id") or dialog.get("speakerId")
        speaker_type = speaker_map.get(speaker_id, "unknown")

        return {
            "audio_path": full_audio_path,
            "text": text,
            "speaker_type": speaker_type,
            "duration": duration,
            "session_id": session_dir.name,
            "speaker_id": speaker_id
        }

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """
        오디오 파일 로드 및 전처리

        전처리:
        1. 모노 변환
        2. 8kHz 리샘플링 (16kHz → 8kHz)
        3. 정규화 [-1, 1]
        """
        waveform, sr = torchaudio.load(audio_path)

        # 모노 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 리샘플링
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        # 정규화
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        return waveform  # [1, T]

    def _print_stats(self):
        """데이터셋 통계 출력"""
        print("=" * 60)
        print(f"CallCopsDataset Initialized ({self.mode})")
        print("=" * 60)
        print(f"  Data directory: {self.data_dir}")
        print(f"  Total samples: {len(self.items)}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Min duration: {self.min_duration} sec")

        # 화자 타입별 통계
        speaker_counts = {}
        total_duration = 0

        for item in self.items:
            st = item["speaker_type"]
            speaker_counts[st] = speaker_counts.get(st, 0) + 1
            total_duration += item["duration"]

        print(f"  Total duration: {total_duration / 3600:.2f} hours")
        print(f"  Speaker types:")
        for st, count in sorted(speaker_counts.items()):
            print(f"    - {st}: {count} ({100 * count / len(self.items):.1f}%)")
        print("=" * 60)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        단일 샘플 반환

        Returns:
            audio_tensor: [1, T] 오디오 텐서
            text: 전사 텍스트
            speaker_type: 화자 타입 ("counselor" 또는 "customer")
        """
        item = self.items[idx]

        # 오디오 로드
        audio = self._load_audio(item["audio_path"])

        # 증강 적용 (학습 모드에서만)
        if self.augmentation is not None:
            audio = self.augmentation(audio)

        return audio, item["text"], item["speaker_type"]


# =============================================================================
# Watermarking Dataset (워터마킹 학습용)
# =============================================================================

class CallCopsWatermarkDataset(Dataset):
    """
    워터마킹 학습용 데이터셋

    CallCopsDataset을 래핑하여 워터마크 페이로드 생성 추가.

    Returns:
        dict containing:
        - audio: [1, T] 오디오 텐서
        - bits: [128] 워터마크 비트
        - text: 전사 텍스트
        - speaker_type: 화자 타입
    """

    def __init__(
        self,
        base_dataset: CallCopsDataset,
        payload_generator: Optional[PayloadGenerator] = None,
        frame_ms: int = 40,
        max_frames: int = 128
    ):
        self.base_dataset = base_dataset
        self.payload_generator = payload_generator or PayloadGenerator()
        self.frame_ms = frame_ms
        self.max_frames = max_frames
        self.frame_samples = int(base_dataset.sample_rate * frame_ms / 1000)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audio, text, speaker_type = self.base_dataset[idx]

        # 최대 길이 제한
        max_samples = self.max_frames * self.frame_samples

        if audio.shape[-1] > max_samples:
            # 랜덤 세그먼트 추출
            start = random.randint(0, audio.shape[-1] - max_samples)
            audio = audio[:, start:start + max_samples]

        # 페이로드 생성
        bits = self.payload_generator.generate(batch_size=1).squeeze(0)

        return {
            "audio": audio,
            "bits": bits,
            "text": text,
            "speaker_type": speaker_type
        }


# =============================================================================
# Collate Functions
# =============================================================================

def collate_fn_basic(
    batch: List[Tuple[torch.Tensor, str, str]]
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    기본 collate 함수 (CallCopsDataset용)

    가변 길이 오디오를 패딩하여 배치로 묶음.
    """
    audios = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    speaker_types = [item[2] for item in batch]

    # 최대 길이로 패딩
    max_len = max(a.shape[-1] for a in audios)
    padded_audios = []

    for audio in audios:
        if audio.shape[-1] < max_len:
            padding = max_len - audio.shape[-1]
            audio = F.pad(audio, (0, padding))
        padded_audios.append(audio)

    return torch.stack(padded_audios), texts, speaker_types


def collate_fn_watermark(
    batch: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    워터마킹 collate 함수 (CallCopsWatermarkDataset용)
    """
    audios = [item["audio"] for item in batch]
    bits = [item["bits"] for item in batch]
    texts = [item["text"] for item in batch]
    speaker_types = [item["speaker_type"] for item in batch]

    # 최대 길이로 패딩
    max_len = max(a.shape[-1] for a in audios)
    padded_audios = []

    for audio in audios:
        if audio.shape[-1] < max_len:
            padding = max_len - audio.shape[-1]
            audio = F.pad(audio, (0, padding))
        padded_audios.append(audio)

    return {
        "audio": torch.stack(padded_audios),
        "bits": torch.stack(bits),
        "texts": texts,
        "speaker_types": speaker_types,
        "lengths": torch.tensor([a.shape[-1] for a in audios])
    }


# =============================================================================
# DataLoader Factory Functions
# =============================================================================

def create_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    mode: str = "train",
    sample_rate: int = 8000,
    min_duration: float = 3.0,
    augmentation_config: Optional[Dict[str, Any]] = None,
    for_watermarking: bool = True,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    DataLoader 팩토리 함수

    Args:
        data_dir: 데이터 디렉토리 (data/raw/training 또는 data/raw/validation)
        batch_size: 배치 크기
        num_workers: 워커 수
        mode: "train" 또는 "val"
        sample_rate: 샘플링 레이트 (8kHz)
        min_duration: 최소 오디오 길이 (초)
        augmentation_config: 증강 설정
        for_watermarking: 워터마킹 데이터셋 사용 여부
        pin_memory: GPU 전송 가속 여부
        **kwargs: CallCopsDataset에 전달할 추가 인자

    Returns:
        PyTorch DataLoader
    """
    # 증강 설정
    augmentation = None
    if mode == "train":
        aug_config = augmentation_config or {}
        augmentation = AudioAugmentation(sample_rate=sample_rate, **aug_config)

    # 기본 데이터셋 생성 (pin_memory는 Dataset 인자가 아님)
    base_dataset = CallCopsDataset(
        data_dir=data_dir,
        sample_rate=sample_rate,
        min_duration=min_duration,
        augmentation=augmentation,
        mode=mode,
        **kwargs
    )

    # 워터마킹 데이터셋 래핑
    if for_watermarking:
        dataset = CallCopsWatermarkDataset(base_dataset)
        collate = collate_fn_watermark
    else:
        dataset = base_dataset
        collate = collate_fn_basic

    shuffle = (mode == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        drop_last=(mode == "train")
    )


def create_train_val_loaders(
    train_dir: Union[str, Path] = "data/raw/training",
    val_dir: Union[str, Path] = "data/raw/validation",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    학습/검증 DataLoader 쌍 생성

    Args:
        train_dir: 학습 데이터 디렉토리
        val_dir: 검증 데이터 디렉토리

    Returns:
        (train_loader, val_loader) 튜플
    """
    train_loader = create_dataloader(
        data_dir=train_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="train",
        **kwargs
    )

    val_loader = create_dataloader(
        data_dir=val_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="val",
        **kwargs
    )

    return train_loader, val_loader


# =============================================================================
# Main (테스트용)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CallCops Data Pipeline Test")
    print("=" * 60)

    # 1. PayloadGenerator 테스트
    print("\n[1] PayloadGenerator Test")
    pg = PayloadGenerator()
    payload = pg.generate(batch_size=4)
    print(f"    Payload shape: {payload.shape}")
    print(f"    Sync pattern valid: {pg.verify_sync(payload[0])}")

    # 2. AudioAugmentation 테스트
    print("\n[2] AudioAugmentation Test")
    aug = AudioAugmentation(sample_rate=8000)
    dummy_audio = torch.randn(1, 8000)  # 1초 @ 8kHz
    augmented = aug(dummy_audio)
    print(f"    Input shape: {dummy_audio.shape}")
    print(f"    Output shape: {augmented.shape}")
    print(f"    Input range: [{dummy_audio.min():.3f}, {dummy_audio.max():.3f}]")
    print(f"    Output range: [{augmented.min():.3f}, {augmented.max():.3f}]")

    # 3. 데이터셋 테스트 (데이터가 있는 경우)
    print("\n[3] Dataset Test")
    train_dir = Path("data/raw/training")

    if train_dir.exists():
        try:
            dataset = CallCopsDataset(
                data_dir=train_dir,
                sample_rate=8000,
                min_duration=3.0,
                mode="train"
            )

            if len(dataset) > 0:
                audio, text, speaker_type = dataset[0]
                print(f"    Audio shape: {audio.shape}")
                print(f"    Text: {text[:50]}..." if len(text) > 50 else f"    Text: {text}")
                print(f"    Speaker type: {speaker_type}")
        except Exception as e:
            print(f"    Dataset loading failed: {e}")
    else:
        print(f"    Train directory not found: {train_dir}")
        print("    Place your data in data/raw/training/ to test.")

    print("\n" + "=" * 60)
    print("Data pipeline configuration:")
    print("  - Sample rate: 8000 Hz")
    print("  - Min duration: 3.0 sec")
    print("  - Speaker types: counselor, customer")
    print("  - JSON metadata: dataSet.typeInfo.speakers, dataSet.dialogs")
    print("=" * 60)
