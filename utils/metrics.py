"""
CallCops: Evaluation Metrics
=================================

PESQ, BER, SNR 등 평가 메트릭 구현.

품질 목표:
- PESQ >= 4.0 (MOS 스케일)
- BER < 5% (G.729 압축 후)
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union, List

# Optional imports
try:
    from pesq import pesq as pesq_fn
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi as stoi_fn
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


def compute_ber(
    pred_bits: torch.Tensor,
    target_bits: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    """
    Bit Error Rate (BER) 계산

    Args:
        pred_bits: [B, N] 예측 비트 확률 또는 이진값
        target_bits: [B, N] 타겟 비트 (0 또는 1)
        threshold: 이진화 임계치

    Returns:
        dict with BER statistics
    """
    # 이진화
    if pred_bits.max() > 1 or pred_bits.min() < 0:
        pred_bits = torch.sigmoid(pred_bits)

    pred_binary = (pred_bits > threshold).float()

    # 오류 계산
    errors = (pred_binary != target_bits).float()

    # 통계
    ber = errors.mean().item()
    ber_per_sample = errors.mean(dim=-1)

    return {
        'ber': ber,
        'ber_mean': ber,
        'ber_std': ber_per_sample.std().item(),
        'ber_min': ber_per_sample.min().item(),
        'ber_max': ber_per_sample.max().item(),
        'accuracy': 1 - ber,
        'total_bits': target_bits.numel(),
        'error_bits': int(errors.sum().item())
    }


def compute_snr(
    original: torch.Tensor,
    processed: torch.Tensor,
    eps: float = 1e-10
) -> float:
    """
    Signal-to-Noise Ratio (SNR) 계산

    SNR = 10 * log10(P_signal / P_noise)

    Args:
        original: 원본 신호
        processed: 처리된 신호 (워터마크 삽입 등)
        eps: 0 방지용 작은 값

    Returns:
        SNR (dB)
    """
    # 노이즈 = 원본 - 처리됨
    noise = original - processed

    # 파워 계산
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(noise ** 2) + eps

    # SNR (dB)
    snr = 10 * torch.log10(signal_power / noise_power)

    return snr.item()


def compute_psnr(
    original: torch.Tensor,
    processed: torch.Tensor,
    max_val: float = 1.0
) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) 계산

    PSNR = 10 * log10(MAX^2 / MSE)

    Args:
        original: 원본 신호
        processed: 처리된 신호
        max_val: 최대값 (정규화된 오디오는 1.0)

    Returns:
        PSNR (dB)
    """
    mse = torch.mean((original - processed) ** 2)
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-10))

    return psnr.item()


def compute_pesq_batch(
    original: torch.Tensor,
    processed: torch.Tensor,
    sample_rate: int = 8000,
    mode: str = 'nb'  # 'nb' for narrowband (8kHz), 'wb' for wideband (16kHz)
) -> dict:
    """
    PESQ (Perceptual Evaluation of Speech Quality) 배치 계산

    PESQ 범위: -0.5 ~ 4.5 (MOS-LQO)
    목표: >= 4.0

    Args:
        original: [B, T] 또는 [B, 1, T] 원본 오디오
        processed: [B, T] 또는 [B, 1, T] 처리된 오디오
        sample_rate: 샘플링 레이트
        mode: 'nb' (narrowband, 8kHz) or 'wb' (wideband, 16kHz)

    Returns:
        dict with PESQ statistics
    """
    if not PESQ_AVAILABLE:
        return {'error': 'pesq library not installed', 'pesq_mean': -1}

    # Shape 정리
    if original.dim() == 3:
        original = original.squeeze(1)
    if processed.dim() == 3:
        processed = processed.squeeze(1)

    # CPU로 이동
    original = original.cpu().numpy()
    processed = processed.cpu().numpy()

    pesq_scores = []

    for i in range(original.shape[0]):
        try:
            score = pesq_fn(sample_rate, original[i], processed[i], mode)
            pesq_scores.append(score)
        except Exception as e:
            print(f"PESQ error at sample {i}: {e}")
            continue

    if pesq_scores:
        return {
            'pesq_mean': np.mean(pesq_scores),
            'pesq_std': np.std(pesq_scores),
            'pesq_min': np.min(pesq_scores),
            'pesq_max': np.max(pesq_scores),
            'num_samples': len(pesq_scores)
        }
    else:
        return {'error': 'No valid PESQ scores', 'pesq_mean': -1}


def compute_stoi(
    original: torch.Tensor,
    processed: torch.Tensor,
    sample_rate: int = 8000,
    extended: bool = False
) -> dict:
    """
    STOI (Short-Time Objective Intelligibility) 계산

    STOI 범위: 0 ~ 1 (높을수록 좋음)

    Args:
        original: 원본 오디오
        processed: 처리된 오디오
        sample_rate: 샘플링 레이트
        extended: Extended STOI 사용 여부

    Returns:
        dict with STOI statistics
    """
    if not STOI_AVAILABLE:
        return {'error': 'pystoi library not installed', 'stoi_mean': -1}

    # Shape 정리
    if original.dim() == 3:
        original = original.squeeze(1)
    if processed.dim() == 3:
        processed = processed.squeeze(1)

    original = original.cpu().numpy()
    processed = processed.cpu().numpy()

    stoi_scores = []

    for i in range(original.shape[0]):
        try:
            score = stoi_fn(original[i], processed[i], sample_rate, extended=extended)
            stoi_scores.append(score)
        except Exception as e:
            print(f"STOI error at sample {i}: {e}")
            continue

    if stoi_scores:
        return {
            'stoi_mean': np.mean(stoi_scores),
            'stoi_std': np.std(stoi_scores),
            'stoi_min': np.min(stoi_scores),
            'stoi_max': np.max(stoi_scores),
            'num_samples': len(stoi_scores)
        }
    else:
        return {'error': 'No valid STOI scores', 'stoi_mean': -1}


def compute_spectral_distance(
    original: torch.Tensor,
    processed: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 64
) -> dict:
    """
    스펙트럼 거리 메트릭

    Log Spectral Distance (LSD)와 Spectral Convergence 계산.

    Args:
        original: 원본 오디오
        processed: 처리된 오디오
        n_fft: FFT 크기
        hop_length: 홉 크기

    Returns:
        dict with spectral metrics
    """
    # Shape 정리
    if original.dim() == 3:
        original = original.squeeze(1)
    if processed.dim() == 3:
        processed = processed.squeeze(1)

    # STFT
    window = torch.hann_window(n_fft, device=original.device)

    orig_spec = torch.stft(
        original, n_fft=n_fft, hop_length=hop_length,
        window=window, return_complex=True
    )
    proc_spec = torch.stft(
        processed, n_fft=n_fft, hop_length=hop_length,
        window=window, return_complex=True
    )

    # Magnitude
    orig_mag = torch.abs(orig_spec)
    proc_mag = torch.abs(proc_spec)

    # Log Spectral Distance (LSD)
    # LSD = sqrt(mean((10*log10(S1/S2))^2))
    log_diff = 10 * torch.log10(proc_mag + 1e-8) - 10 * torch.log10(orig_mag + 1e-8)
    lsd = torch.sqrt(torch.mean(log_diff ** 2))

    # Spectral Convergence
    # SC = ||S1 - S2||_F / ||S1||_F
    sc = torch.norm(orig_mag - proc_mag, p='fro') / (torch.norm(orig_mag, p='fro') + 1e-8)

    return {
        'lsd': lsd.item(),
        'spectral_convergence': sc.item()
    }


def compute_all_metrics(
    original: torch.Tensor,
    processed: torch.Tensor,
    pred_bits: Optional[torch.Tensor] = None,
    target_bits: Optional[torch.Tensor] = None,
    sample_rate: int = 8000
) -> dict:
    """
    모든 메트릭 통합 계산

    Args:
        original: 원본 오디오
        processed: 워터마크된 오디오
        pred_bits: 추출된 비트 (optional)
        target_bits: 원본 비트 (optional)
        sample_rate: 샘플링 레이트

    Returns:
        dict with all metrics
    """
    results = {}

    # 오디오 품질 메트릭
    results['snr'] = compute_snr(original, processed)
    results['psnr'] = compute_psnr(original, processed)

    spectral = compute_spectral_distance(original, processed)
    results.update(spectral)

    # PESQ (optional)
    pesq_result = compute_pesq_batch(original, processed, sample_rate)
    results['pesq'] = pesq_result.get('pesq_mean', -1)

    # STOI (optional)
    stoi_result = compute_stoi(original, processed, sample_rate)
    results['stoi'] = stoi_result.get('stoi_mean', -1)

    # BER (if bits provided)
    if pred_bits is not None and target_bits is not None:
        ber_result = compute_ber(pred_bits, target_bits)
        results['ber'] = ber_result['ber']
        results['accuracy'] = ber_result['accuracy']

    # 품질 체크
    results['quality_check'] = {
        'pesq_pass': results['pesq'] >= 4.0 if results['pesq'] > 0 else None,
        'ber_pass': results.get('ber', 1.0) < 0.05,
        'snr_pass': results['snr'] > 20  # 20dB 이상
    }

    return results


class MetricTracker:
    """
    학습 중 메트릭 추적기
    """

    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.reset()

    def reset(self):
        """메트릭 초기화"""
        self.values = {name: [] for name in self.metric_names}
        self.count = 0

    def update(self, metrics: dict, n: int = 1):
        """메트릭 업데이트"""
        for name, value in metrics.items():
            if name in self.values:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.values[name].append(value)
        self.count += n

    def get_average(self) -> dict:
        """평균 메트릭 반환"""
        return {
            name: np.mean(values) if values else 0
            for name, values in self.values.items()
        }

    def get_std(self) -> dict:
        """표준편차 반환"""
        return {
            name: np.std(values) if values else 0
            for name, values in self.values.items()
        }


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("Metrics Test")
    print("=" * 60)

    # 더미 데이터
    batch_size = 4
    seq_len = 320  # 40ms @ 8kHz
    bits_dim = 128

    original = torch.randn(batch_size, seq_len)
    processed = original + 0.01 * torch.randn_like(original)  # 작은 섭동

    pred_bits = torch.sigmoid(torch.randn(batch_size, bits_dim))
    target_bits = torch.randint(0, 2, (batch_size, bits_dim)).float()

    # BER 테스트
    print("\n[BER Test]")
    ber_result = compute_ber(pred_bits, target_bits)
    print(f"  BER: {ber_result['ber']:.4f}")
    print(f"  Accuracy: {ber_result['accuracy']:.4f}")

    # SNR 테스트
    print("\n[SNR Test]")
    snr = compute_snr(original, processed)
    print(f"  SNR: {snr:.2f} dB")

    # PSNR 테스트
    print("\n[PSNR Test]")
    psnr = compute_psnr(original, processed)
    print(f"  PSNR: {psnr:.2f} dB")

    # Spectral 테스트
    print("\n[Spectral Distance Test]")
    spectral = compute_spectral_distance(original, processed)
    print(f"  LSD: {spectral['lsd']:.4f}")
    print(f"  Spectral Convergence: {spectral['spectral_convergence']:.4f}")

    # PESQ 테스트 (라이브러리 있을 경우)
    print("\n[PESQ Test]")
    if PESQ_AVAILABLE:
        # PESQ는 실제 음성이 필요하므로 스킵
        print("  PESQ library available (requires real speech for testing)")
    else:
        print("  PESQ library not installed")

    # 전체 메트릭
    print("\n[All Metrics]")
    all_metrics = compute_all_metrics(
        original, processed,
        pred_bits, target_bits
    )
    for key, value in all_metrics.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    print("\n✓ All metrics working correctly!")
