"""
CallCops: Loss Functions
=============================

복합 손실 함수 구현:
    L_total = λ_bit * L_BCE + λ_audio * L_Mel + λ_adv * L_GAN

손실 함수 구성:
1. L_BCE: Binary Cross-Entropy for bit accuracy
2. L_Mel: Multi-Resolution Mel-Spectrogram loss
3. L_GAN: Adversarial loss for naturalness
4. L_Codec: Codec robustness loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss
    ==========================

    여러 FFT 크기에서 STFT를 계산하여 시간-주파수 손실 측정.
    다양한 주파수 해상도에서의 음질 보존.

    8kHz 최적화:
    - n_fft: [64, 128, 256] (8ms, 16ms, 32ms 윈도우)
    - hop: n_fft // 4
    """

    def __init__(
        self,
        fft_sizes: List[int] = [64, 128, 256],
        hop_sizes: Optional[List[int]] = None,
        win_sizes: Optional[List[int]] = None,
        window: str = "hann",
        eps: float = 1e-7
    ):
        super().__init__()

        self.fft_sizes = fft_sizes
        self.eps = eps

        if hop_sizes is None:
            hop_sizes = [n // 4 for n in fft_sizes]
        if win_sizes is None:
            win_sizes = fft_sizes

        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

        # Window functions
        self.register_buffer('windows', torch.zeros(len(fft_sizes), max(win_sizes)))

        for i, win_size in enumerate(win_sizes):
            if window == "hann":
                win = torch.hann_window(win_size)
            elif window == "hamming":
                win = torch.hamming_window(win_size)
            else:
                raise ValueError(f"Unknown window: {window}")

            self.windows[i, :win_size] = win

    def stft(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int,
        window: torch.Tensor
    ) -> torch.Tensor:
        """
        STFT 계산

        Args:
            x: [B, T] 오디오

        Returns:
            [B, F, T'] magnitude spectrogram
        """
        # Ensure 2D input
        if x.dim() == 3:
            x = x.squeeze(1)

        # STFT
        spec = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window[:win_size],
            return_complex=True,
            pad_mode='reflect'
        )

        # Magnitude
        mag = torch.abs(spec)

        return mag

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-resolution STFT loss

        Args:
            pred: [B, 1, T] predicted audio
            target: [B, 1, T] target audio

        Returns:
            sc_loss: Spectral convergence loss
            mag_loss: Log magnitude loss
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for i, (fft_size, hop_size, win_size) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_sizes)
        ):
            window = self.windows[i].to(pred.device)

            pred_mag = self.stft(pred, fft_size, hop_size, win_size, window)
            target_mag = self.stft(target, fft_size, hop_size, win_size, window)

            # Spectral Convergence Loss
            # ||S_target - S_pred||_F / ||S_target||_F
            # eps를 더해 0으로 나누기 방지
            sc_loss += torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + self.eps)

            # Log Magnitude Loss
            # ||log(S_target) - log(S_pred)||_1
            # log(0) 방지를 위해 eps 추가
            log_pred = torch.log(pred_mag + self.eps)
            log_target = torch.log(target_mag + self.eps)
            mag_loss += F.l1_loss(log_pred, log_target)

        # Average over resolutions
        num_resolutions = len(self.fft_sizes)
        sc_loss /= num_resolutions
        mag_loss /= num_resolutions

        return sc_loss, mag_loss


class MultiResolutionMelLoss(nn.Module):
    """
    Multi-Resolution Mel-Spectrogram Loss
    =====================================

    Mel 스케일 스펙트로그램 손실.
    인간 청각 특성을 반영한 주파수 가중치.

    8kHz 최적화:
    - Nyquist = 4000Hz
    - n_mels = 40 (전화 대역폭에 적합)
    - fmin = 80Hz, fmax = 3800Hz
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        n_fft_list: List[int] = [64, 128, 256],
        n_mels: int = 40,
        fmin: float = 80.0,
        fmax: Optional[float] = None,
        eps: float = 1e-7
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft_list = n_fft_list
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or (sample_rate / 2 - 100)  # 3900Hz for 8kHz
        self.eps = eps

        # Mel filterbanks for each resolution
        self.mel_bases = nn.ParameterList()

        for n_fft in n_fft_list:
            mel_basis = self._create_mel_filterbank(n_fft)
            self.mel_bases.append(nn.Parameter(mel_basis, requires_grad=False))

    def _create_mel_filterbank(self, n_fft: int) -> torch.Tensor:
        """
        Mel filterbank 생성

        수식:
            mel(f) = 2595 * log10(1 + f/700)
        """
        n_freqs = n_fft // 2 + 1

        # Mel scale conversion
        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(self.fmax)

        # Mel points
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # FFT bin indices
        bin_indices = torch.floor((n_fft + 1) * hz_points / self.sample_rate).long()

        # Create filterbank
        filterbank = torch.zeros(self.n_mels, n_freqs)

        for i in range(self.n_mels):
            left = bin_indices[i]
            center = bin_indices[i + 1]
            right = bin_indices[i + 2]

            # Left slope
            for j in range(left, center):
                if j < n_freqs:
                    filterbank[i, j] = (j - left) / (center - left + 1e-8)

            # Right slope
            for j in range(center, right):
                if j < n_freqs:
                    filterbank[i, j] = (right - j) / (right - center + 1e-8)

        return filterbank

    def _hz_to_mel(self, hz: float) -> float:
        """Hz to Mel scale"""
        return 2595 * math.log10(1 + hz / 700)

    def _mel_to_hz(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel to Hz scale"""
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-resolution Mel loss

        Args:
            pred: [B, 1, T] predicted audio
            target: [B, 1, T] target audio

        Returns:
            mel_loss: Combined mel spectrogram loss
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        total_loss = 0.0

        for i, n_fft in enumerate(self.n_fft_list):
            hop_length = n_fft // 4
            mel_basis = self.mel_bases[i].to(pred.device)

            # STFT
            pred_spec = torch.stft(
                pred, n_fft=n_fft, hop_length=hop_length,
                window=torch.hann_window(n_fft, device=pred.device),
                return_complex=True, pad_mode='reflect'
            )
            target_spec = torch.stft(
                target, n_fft=n_fft, hop_length=hop_length,
                window=torch.hann_window(n_fft, device=target.device),
                return_complex=True, pad_mode='reflect'
            )

            # Magnitude
            pred_mag = torch.abs(pred_spec)
            target_mag = torch.abs(target_spec)

            # Apply mel filterbank
            pred_mel = torch.matmul(mel_basis, pred_mag)
            target_mel = torch.matmul(mel_basis, target_mag)

            # Log mel (eps 추가)
            pred_log_mel = torch.log(pred_mel + self.eps)
            target_log_mel = torch.log(target_mel + self.eps)

            # L1 loss
            total_loss += F.l1_loss(pred_log_mel, target_log_mel)

        return total_loss / len(self.n_fft_list)


class BitAccuracyLoss(nn.Module):
    """
    Bit Accuracy Loss (L_BCE)
    =========================

    워터마크 비트 복원 정확도 손실.
    수치적 안정성을 위해 로짓(Logit) 입력을 받아 BCEWithLogits를 계산합니다.

    수식:
        L_BCE = binary_cross_entropy_with_logits(pred_bits, target_bits)

    추가 기능:
    - Focal loss 옵션 (어려운 샘플에 집중)
    - Label smoothing (과적합 방지)
    """

    def __init__(
        self,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()

        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        pred_bits: torch.Tensor,
        target_bits: torch.Tensor
    ) -> torch.Tensor:
        """
        Bit accuracy loss

        Args:
            pred_bits: [B, N_bits] predicted logits (before sigmoid)
            target_bits: [B, N_bits] target bits (0 or 1)

        Returns:
            loss: BCE or Focal loss
        """
        # Label smoothing
        if self.label_smoothing > 0:
            target_bits = target_bits * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.use_focal:
            # Focal Loss with logits
            # BCE with logits is more stable than sigmoid -> bce
            bce = F.binary_cross_entropy_with_logits(pred_bits, target_bits, reduction='none')
            pred_probs = torch.sigmoid(pred_bits)
            pt = torch.where(target_bits == 1, pred_probs, 1 - pred_probs)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss = (focal_weight * bce).mean()
        else:
            # Standard BCE with logits (Stable)
            loss = F.binary_cross_entropy_with_logits(pred_bits, target_bits)

        return loss


class AdversarialLoss(nn.Module):
    """
    Adversarial Loss (L_GAN)
    ========================

    GAN 기반 자연스러움 손실.
    Discriminator가 원본과 워터마크 오디오를 구분하지 못하도록 학습.

    지원 모드:
    - 'vanilla': Standard GAN loss
    - 'lsgan': Least Squares GAN (더 안정적)
    - 'hinge': Hinge loss (spectral norm과 함께 사용)
    """

    def __init__(self, mode: str = 'lsgan'):
        super().__init__()

        self.mode = mode

        if mode == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif mode == 'lsgan':
            self.criterion = nn.MSELoss()
        elif mode == 'hinge':
            self.criterion = None  # Custom implementation
        else:
            raise ValueError(f"Unknown GAN mode: {mode}")

    def forward(
        self,
        pred: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """
        Adversarial loss computation

        Args:
            pred: Discriminator output
            target_is_real: True for real samples, False for fake

        Returns:
            GAN loss
        """
        if self.mode == 'vanilla':
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return self.criterion(pred, target)

        elif self.mode == 'lsgan':
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return self.criterion(pred, target)

        elif self.mode == 'hinge':
            if target_is_real:
                return F.relu(1 - pred).mean()
            else:
                return F.relu(1 + pred).mean()

    def generator_loss(self, fake_pred: List[torch.Tensor]) -> torch.Tensor:
        """
        Generator loss (make fake look real)

        Args:
            fake_pred: List of discriminator outputs for fake samples
        """
        loss = 0.0
        for pred in fake_pred:
            if self.mode == 'vanilla':
                target = torch.ones_like(pred)
                loss += self.criterion(pred, target)
            elif self.mode == 'lsgan':
                target = torch.ones_like(pred)
                loss += self.criterion(pred, target)
            elif self.mode == 'hinge':
                loss += -pred.mean()

        return loss / len(fake_pred)

    def discriminator_loss(
        self,
        real_pred: List[torch.Tensor],
        fake_pred: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Discriminator loss

        Args:
            real_pred: List of discriminator outputs for real samples
            fake_pred: List of discriminator outputs for fake samples
        """
        loss = 0.0

        for real, fake in zip(real_pred, fake_pred):
            loss += self.forward(real, target_is_real=True)
            loss += self.forward(fake.detach(), target_is_real=False)

        return loss / (2 * len(real_pred))


class DetectionLoss(nn.Module):
    """
    Detection Loss
    ==============

    워터마크 탐지 신뢰도 손실.
    워터마크된 오디오는 1, 원본은 0으로 분류.
    수치적 안정성을 위해 로짓 입력을 사용합니다.
    """

    def __init__(self):
        super().__init__()
        # Using logits version for stability
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        detection_pred: torch.Tensor,
        has_watermark: torch.Tensor
    ) -> torch.Tensor:
        """
        Detection loss

        Args:
            detection_pred: [B, 1] detection logits
            has_watermark: [B, 1] ground truth (1 if watermarked)

        Returns:
            BCE loss
        """
        return self.criterion(detection_pred, has_watermark)


class CallCopsLoss(nn.Module):
    """
    CallCops Complete Loss Function
    =================================

    전체 손실 함수 통합:
        L_total = λ_bit * L_BCE + λ_audio * L_Mel + λ_adv * L_GAN + λ_det * L_Det

    - L_BCE: 비트 정확도 (로짓 입력 사용)
    - L_Mel: 음질 보존 (PESQ >= 4.0 목표)
    - L_GAN: 자연스러움
    - L_Det: 탐지 정확도
    """

    def __init__(
        self,
        lambda_bit: float = 10.0,
        lambda_audio: float = 10.0,
        lambda_adv: float = 0.1,
        lambda_det: float = 0.5,
        lambda_stft: float = 2.0,
        lambda_l1: float = 10.0,   # NEW: Direct waveform L1 loss for SNR
        sample_rate: int = 8000,
        use_focal_loss: bool = False,
        gan_mode: str = 'lsgan'
    ):
        super().__init__()

        self.lambda_bit = lambda_bit
        self.lambda_audio = lambda_audio
        self.lambda_adv = lambda_adv
        self.lambda_det = lambda_det
        self.lambda_stft = lambda_stft
        self.lambda_l1 = lambda_l1  # NEW: Direct waveform L1 loss

        # Individual loss components
        self.bit_loss = BitAccuracyLoss(use_focal=use_focal_loss)
        self.mel_loss = MultiResolutionMelLoss(sample_rate=sample_rate)
        self.stft_loss = MultiResolutionSTFTLoss()
        self.adv_loss = AdversarialLoss(mode=gan_mode)
        self.det_loss = DetectionLoss()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        pred_bits: torch.Tensor,
        target_bits: torch.Tensor,
        detection_pred: torch.Tensor,
        disc_fake: Optional[List[torch.Tensor]] = None,
        disc_real: Optional[List[torch.Tensor]] = None
    ) -> dict:
        """
        Complete loss computation

        Note:
            로그 연산 및 정밀도 안정을 위해 FP32로 강제 실행됩니다.
        """
        # 정밀도 보장을 위해 입력을 FP32로 캐스팅
        pred_audio = pred_audio.float()
        target_audio = target_audio.float()
        pred_bits = pred_bits.float()
        target_bits = target_bits.float()
        detection_pred = detection_pred.float()
        
        if disc_fake is not None:
            disc_fake = [d.float() for d in disc_fake]
        if disc_real is not None:
            disc_real = [d.float() for d in disc_real]

        losses = {}

        # 1. Bit Accuracy Loss (L_BCE) - Stable Logits Version
        losses['bit'] = self.bit_loss(pred_bits, target_bits)

        # 2. Audio Quality Loss (L_Mel + L_STFT)
        losses['mel'] = self.mel_loss(pred_audio, target_audio)

        sc_loss, mag_loss = self.stft_loss(pred_audio, target_audio)
        losses['stft_sc'] = sc_loss
        losses['stft_mag'] = mag_loss
        losses['stft'] = sc_loss + mag_loss
        
        # NEW: Direct L1 waveform loss (directly improves SNR)
        losses['l1'] = F.l1_loss(pred_audio, target_audio)

        # 3. Detection Loss
        has_watermark = torch.ones_like(detection_pred)
        losses['detection'] = self.det_loss(detection_pred, has_watermark)

        # 4. Adversarial Loss (if discriminator outputs provided)
        if disc_fake is not None:
            losses['adv_g'] = self.adv_loss.generator_loss(disc_fake)
        else:
            losses['adv_g'] = torch.tensor(0.0, device=pred_audio.device)

        # 5. Total Generator Loss
        losses['total'] = (
            self.lambda_bit * losses['bit'] +
            self.lambda_audio * losses['mel'] +
            self.lambda_stft * losses['stft'] +
            self.lambda_l1 * losses['l1'] +    # NEW: L1 waveform loss
            self.lambda_det * losses['detection'] +
            self.lambda_adv * losses['adv_g']
        )

        # 6. Discriminator Loss (computed separately if needed)
        if disc_fake is not None and disc_real is not None:
            losses['adv_d'] = self.adv_loss.discriminator_loss(disc_real, disc_fake)

        return losses

    def compute_metrics(
        self,
        pred_logits: torch.Tensor,
        target_bits: torch.Tensor
    ) -> dict:
        """
        Compute evaluation metrics from logits

        Args:
            pred_logits: [B, N_bits] predicted logits
            target_bits: [B, N_bits] target bits

        Returns:
            Dictionary with BER, accuracy
        """
        # Convert logits to probabilities for metric calculation
        pred_probs = torch.sigmoid(pred_logits)
        
        # Round predictions to 0/1
        pred_binary = (pred_probs > 0.5).float()

        # Bit Error Rate
        errors = (pred_binary != target_bits).float()
        ber = errors.mean().item()

        # Accuracy
        accuracy = 1.0 - ber

        return {
            'ber': ber,
            'accuracy': accuracy
        }


if __name__ == "__main__":
    # 테스트 코드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("CallCops Loss Function Test")
    print("=" * 60)

    # 손실 함수 초기화
    loss_fn = CallCopsLoss(
        lambda_bit=10.0,
        lambda_audio=10.0,
        lambda_adv=0.1,
        sample_rate=8000
    ).to(device)

    # 테스트 텐서 (8kHz, 40ms = 320 samples)
    batch_size = 4
    n_bits = 128

    pred_audio = torch.randn(batch_size, 1, 320).to(device)
    target_audio = torch.randn(batch_size, 1, 320).to(device)
    pred_bits = torch.sigmoid(torch.randn(batch_size, n_bits)).to(device)
    target_bits = torch.randint(0, 2, (batch_size, n_bits)).float().to(device)
    detection_pred = torch.sigmoid(torch.randn(batch_size, 1)).to(device)

    # 손실 계산
    losses = loss_fn(
        pred_audio=pred_audio,
        target_audio=target_audio,
        pred_bits=pred_bits,
        target_bits=target_bits,
        detection_pred=detection_pred
    )

    print("\nLoss Components:")
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {value.item():.6f}")

    # 메트릭 계산
    metrics = loss_fn.compute_metrics(pred_bits, target_bits)
    print("\nMetrics:")
    print(f"  BER: {metrics['ber']:.4f} (target < 0.05)")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")


# Legacy alias for backward compatibility
CallShieldLoss = CallCopsLoss
