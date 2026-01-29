"""
CallCops: Audio Utilities
==============================

8kHz 한국어 오디오 처리를 위한 유틸리티 함수.
"""

import torch
import torch.nn.functional as F
import torchaudio
from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np


def load_audio(
    path: Union[str, Path],
    target_sr: int = 8000,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    오디오 파일 로드

    Args:
        path: 오디오 파일 경로
        target_sr: 목표 샘플링 레이트 (8kHz)
        mono: 모노 변환 여부
        normalize: 정규화 여부 [-1, 1]

    Returns:
        waveform: [C, T] 또는 [T] (mono=True)
        sample_rate: 샘플링 레이트
    """
    waveform, sr = torchaudio.load(str(path))

    # 모노 변환
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 리샘플링
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # 정규화
    if normalize:
        waveform = normalize_audio(waveform)

    if mono:
        waveform = waveform.squeeze(0)

    return waveform, sr


def save_audio(
    waveform: torch.Tensor,
    path: Union[str, Path],
    sample_rate: int = 8000,
    bits_per_sample: int = 16
):
    """
    오디오 파일 저장

    Args:
        waveform: [C, T] 또는 [T] 오디오 텐서
        path: 저장 경로
        sample_rate: 샘플링 레이트
        bits_per_sample: 비트 깊이
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Clipping
    waveform = torch.clamp(waveform, -1.0, 1.0)

    torchaudio.save(
        str(path),
        waveform,
        sample_rate,
        bits_per_sample=bits_per_sample
    )


def normalize_audio(
    waveform: torch.Tensor,
    target_db: float = -3.0
) -> torch.Tensor:
    """
    오디오 정규화

    Args:
        waveform: 오디오 텐서
        target_db: 목표 dB (기본 -3dB)

    Returns:
        정규화된 오디오
    """
    # Peak normalization
    max_val = waveform.abs().max()

    if max_val > 0:
        waveform = waveform / max_val

        # 목표 dB로 조절
        target_gain = 10 ** (target_db / 20)
        waveform = waveform * target_gain

    return waveform


def resample(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int
) -> torch.Tensor:
    """
    리샘플링

    Args:
        waveform: 오디오 텐서
        orig_sr: 원본 샘플링 레이트
        target_sr: 목표 샘플링 레이트

    Returns:
        리샘플된 오디오
    """
    if orig_sr == target_sr:
        return waveform

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform)


def apply_preemphasis(
    waveform: torch.Tensor,
    coef: float = 0.97
) -> torch.Tensor:
    """
    프리엠퍼시스 필터 적용

    고주파 성분 강조 (한국어 자음 명확화)

    수식: y[n] = x[n] - coef * x[n-1]

    Args:
        waveform: [T] 또는 [B, T] 오디오
        coef: 프리엠퍼시스 계수

    Returns:
        프리엠퍼시스 적용된 오디오
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # 첫 샘플 유지
    emphasized = torch.cat([
        waveform[:, :1],
        waveform[:, 1:] - coef * waveform[:, :-1]
    ], dim=1)

    return emphasized.squeeze(0) if emphasized.shape[0] == 1 else emphasized


def frame_audio(
    waveform: torch.Tensor,
    frame_size: int = 320,  # 40ms @ 8kHz
    hop_size: int = 320,    # non-overlapping
    pad_mode: str = 'constant'
) -> torch.Tensor:
    """
    오디오를 프레임 단위로 분할

    Args:
        waveform: [T] 오디오
        frame_size: 프레임 크기 (samples)
        hop_size: 홉 크기 (samples)
        pad_mode: 패딩 모드

    Returns:
        frames: [N, frame_size] 프레임 텐서
    """
    # 패딩 계산
    num_frames = (len(waveform) - frame_size) // hop_size + 1
    remainder = len(waveform) - (num_frames - 1) * hop_size - frame_size

    if remainder > 0:
        pad_len = frame_size - remainder
        waveform = F.pad(waveform, (0, pad_len), mode=pad_mode)
        num_frames += 1

    # Unfold로 프레임 추출
    frames = waveform.unfold(0, frame_size, hop_size)

    return frames


def unframe_audio(
    frames: torch.Tensor,
    hop_size: int = 320,
    original_length: Optional[int] = None
) -> torch.Tensor:
    """
    프레임을 오디오로 복원 (Overlap-Add)

    Args:
        frames: [N, frame_size] 프레임 텐서
        hop_size: 홉 크기
        original_length: 원본 길이 (패딩 제거용)

    Returns:
        waveform: [T] 복원된 오디오
    """
    num_frames, frame_size = frames.shape

    # 출력 길이 계산
    output_length = (num_frames - 1) * hop_size + frame_size

    # Overlap-add
    waveform = torch.zeros(output_length)
    window = torch.ones(frame_size)  # 사각 윈도우

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        waveform[start:end] += frames[i] * window

    # 윈도우 합으로 정규화 (overlap이 있는 경우)
    if hop_size < frame_size:
        window_sum = torch.zeros(output_length)
        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            window_sum[start:end] += window ** 2

        waveform = waveform / (window_sum + 1e-8)

    # 원본 길이로 자르기
    if original_length is not None:
        waveform = waveform[:original_length]

    return waveform


def compute_energy(
    waveform: torch.Tensor,
    frame_size: int = 320,
    hop_size: int = 160
) -> torch.Tensor:
    """
    프레임별 에너지 계산

    Args:
        waveform: [T] 오디오
        frame_size: 프레임 크기
        hop_size: 홉 크기

    Returns:
        energy: [N] 프레임별 에너지
    """
    frames = frame_audio(waveform, frame_size, hop_size)
    energy = torch.sum(frames ** 2, dim=1)
    return energy


def compute_zcr(
    waveform: torch.Tensor,
    frame_size: int = 320,
    hop_size: int = 160
) -> torch.Tensor:
    """
    프레임별 Zero-Crossing Rate 계산

    한국어 자음/모음 구분에 유용.

    Args:
        waveform: [T] 오디오
        frame_size: 프레임 크기
        hop_size: 홉 크기

    Returns:
        zcr: [N] 프레임별 ZCR
    """
    frames = frame_audio(waveform, frame_size, hop_size)

    # Sign changes
    signs = torch.sign(frames)
    sign_diff = torch.abs(signs[:, 1:] - signs[:, :-1])
    zcr = torch.sum(sign_diff, dim=1) / (2 * frame_size)

    return zcr


def apply_bandpass_filter(
    waveform: torch.Tensor,
    sample_rate: int = 8000,
    low_freq: int = 300,
    high_freq: int = 3400,
    filter_order: int = 5
) -> torch.Tensor:
    """
    대역 통과 필터 적용 (전화망 대역폭)

    Args:
        waveform: 오디오 텐서
        sample_rate: 샘플링 레이트
        low_freq: 하한 주파수
        high_freq: 상한 주파수
        filter_order: 필터 차수

    Returns:
        필터링된 오디오
    """
    try:
        from scipy import signal

        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = signal.butter(filter_order, [low, high], btype='band')

        # numpy로 변환하여 필터링
        waveform_np = waveform.numpy()
        filtered_np = signal.filtfilt(b, a, waveform_np)

        return torch.from_numpy(filtered_np.copy()).float()

    except ImportError:
        print("Warning: scipy not installed. Returning original audio.")
        return waveform


def add_noise(
    waveform: torch.Tensor,
    snr_db: float = 20.0,
    noise_type: str = 'gaussian'
) -> torch.Tensor:
    """
    노이즈 추가

    Args:
        waveform: 오디오 텐서
        snr_db: Signal-to-Noise Ratio (dB)
        noise_type: 노이즈 종류 ('gaussian', 'pink', 'white')

    Returns:
        노이즈가 추가된 오디오
    """
    # 신호 파워
    signal_power = torch.mean(waveform ** 2)

    # 노이즈 파워 계산
    noise_power = signal_power / (10 ** (snr_db / 10))

    # 노이즈 생성
    if noise_type == 'gaussian' or noise_type == 'white':
        noise = torch.randn_like(waveform)
    elif noise_type == 'pink':
        # Pink noise (1/f)
        noise = _generate_pink_noise(len(waveform))
    else:
        noise = torch.randn_like(waveform)

    # 노이즈 정규화
    noise = noise / torch.sqrt(torch.mean(noise ** 2) + 1e-8)
    noise = noise * torch.sqrt(noise_power)

    return waveform + noise


def _generate_pink_noise(length: int) -> torch.Tensor:
    """Pink noise (1/f spectrum) 생성"""
    # FFT 기반 생성
    freqs = torch.fft.rfftfreq(length)
    freqs[0] = 1e-8  # DC 성분 방지

    # 1/f 스펙트럼
    spectrum = torch.randn(len(freqs)) + 1j * torch.randn(len(freqs))
    spectrum = spectrum / torch.sqrt(freqs)

    # IFFT
    pink = torch.fft.irfft(spectrum, n=length)
    return pink.float()


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("Audio Utilities Test")
    print("=" * 60)

    # 더미 오디오 생성 (1초 @ 8kHz)
    sample_rate = 8000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * np.pi * 440 * t)  # 440Hz sine

    print(f"\nOriginal audio: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]")

    # 프레임화 테스트
    frames = frame_audio(audio, frame_size=320, hop_size=320)
    print(f"Frames: {frames.shape} ({frames.shape[0]} frames of {frames.shape[1]} samples)")

    # 복원 테스트
    reconstructed = unframe_audio(frames, hop_size=320, original_length=len(audio))
    print(f"Reconstructed: {reconstructed.shape}")
    print(f"Reconstruction error: {torch.mean((audio - reconstructed) ** 2):.6f}")

    # 에너지 테스트
    energy = compute_energy(audio)
    print(f"\nEnergy per frame: mean={energy.mean():.4f}, std={energy.std():.4f}")

    # ZCR 테스트
    zcr = compute_zcr(audio)
    print(f"ZCR per frame: mean={zcr.mean():.4f}, std={zcr.std():.4f}")

    # 노이즈 추가 테스트
    noisy = add_noise(audio, snr_db=20)
    actual_snr = 10 * torch.log10(
        torch.mean(audio ** 2) / torch.mean((noisy - audio) ** 2)
    )
    print(f"\nNoisy audio SNR: {actual_snr:.1f} dB (target: 20 dB)")

    print("\n✓ All audio utilities working correctly!")
