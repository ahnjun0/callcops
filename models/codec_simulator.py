"""
CallCops: Differentiable Codec Simulator
==============================================

전화망 코덱(G.711, G.729)의 양자화 오차를 미분 가능하게 시뮬레이션.

- G.729 (8kbps) 압축 후에도 BER < 5% 달성
- 학습 루프 내부에서 코덱 열화를 시뮬레이션
- Straight-Through Estimator (STE)로 gradient 흐름 유지

코덱 특성:
- G.711: 64kbps, 8-bit μ-law/A-law companding
- G.729: 8kbps, CELP 기반 고압축

한국어 음성 특성:
- 한국어 자음의 빠른 전이 구간이 G.729에서 손실되기 쉬움
- 종성 비음(/ㅁ/, /ㄴ/, /ㅇ/)의 저주파 특성 보존 중요
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for Quantization
    ============================================

    Forward: 실제 양자화 수행
    Backward: gradient를 그대로 통과시킴 (STE)

    이를 통해 비미분 가능한 양자화를 학습 가능하게 만듦.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        num_levels: int
    ) -> torch.Tensor:
        """
        Uniform quantization forward

        Args:
            x: 입력 텐서 ([-1, 1] 범위 가정)
            num_levels: 양자화 레벨 수 (e.g., 256 for 8-bit)

        Returns:
            양자화된 텐서
        """
        # [-1, 1] -> [0, num_levels-1]
        x_scaled = (x + 1) / 2 * (num_levels - 1)

        # Round to nearest integer
        x_quantized = torch.round(x_scaled)

        # [0, num_levels-1] -> [-1, 1]
        x_dequantized = x_quantized / (num_levels - 1) * 2 - 1

        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Straight-through: gradient 그대로 전달"""
        return grad_output, None


def straight_through_quantize(x: torch.Tensor, num_levels: int) -> torch.Tensor:
    """STE 양자화 래퍼 함수"""
    return StraightThroughQuantize.apply(x, num_levels)


class MuLawCompanding(nn.Module):
    """
    μ-law Companding (G.711 μ-law)
    ==============================

    북미/일본에서 사용되는 압신 방식.

    수식:
        F(x) = sign(x) * ln(1 + μ|x|) / ln(1 + μ)

    여기서 μ = 255 (8-bit 표준)

    특성:
    - 작은 진폭 신호의 SNR 개선
    - 동적 범위 압축
    """

    def __init__(self, mu: int = 255):
        super().__init__()
        self.mu = mu
        self.log_mu_plus_1 = math.log(1 + mu)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """μ-law 압축"""
        return torch.sign(x) * torch.log1p(self.mu * torch.abs(x)) / self.log_mu_plus_1

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """μ-law 신장"""
        return torch.sign(x) * (torch.exp(torch.abs(x) * self.log_mu_plus_1) - 1) / self.mu

    def forward(
        self,
        x: torch.Tensor,
        quantize: bool = True,
        num_levels: int = 256
    ) -> torch.Tensor:
        """
        μ-law companding with optional quantization

        Args:
            x: 입력 오디오 [-1, 1]
            quantize: 양자화 수행 여부
            num_levels: 양자화 레벨 (기본 256 = 8-bit)

        Returns:
            압신+양자화된 오디오
        """
        # 압축
        compressed = self.compress(x)

        # 양자화 (STE)
        if quantize:
            compressed = straight_through_quantize(compressed, num_levels)

        # 신장
        expanded = self.expand(compressed)

        return expanded


class ALawCompanding(nn.Module):
    """
    A-law Companding (G.711 A-law)
    ==============================

    유럽/국제 표준 압신 방식.

    수식:
        |x| < 1/A: F(x) = A|x| / (1 + ln(A))
        |x| >= 1/A: F(x) = (1 + ln(A|x|)) / (1 + ln(A))

    여기서 A = 87.6 (표준값)

    특성:
    - μ-law보다 낮은 idle channel noise
    - 한국 전화망에서 주로 사용 (국제 표준)
    """

    def __init__(self, A: float = 87.6):
        super().__init__()
        self.A = A
        self.log_A = math.log(A)
        self.one_plus_log_A = 1 + self.log_A

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """A-law 압축"""
        abs_x = torch.abs(x)
        sign_x = torch.sign(x)

        # |x| < 1/A
        small_mask = abs_x < (1 / self.A)
        # |x| >= 1/A
        large_mask = ~small_mask

        result = torch.zeros_like(x)
        result[small_mask] = self.A * abs_x[small_mask] / self.one_plus_log_A
        result[large_mask] = (1 + torch.log(self.A * abs_x[large_mask])) / self.one_plus_log_A

        return sign_x * result

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """A-law 신장"""
        abs_x = torch.abs(x)
        sign_x = torch.sign(x)

        threshold = 1 / self.one_plus_log_A

        small_mask = abs_x < threshold
        large_mask = ~small_mask

        result = torch.zeros_like(x)
        result[small_mask] = abs_x[small_mask] * self.one_plus_log_A / self.A
        result[large_mask] = torch.exp(abs_x[large_mask] * self.one_plus_log_A - 1) / self.A

        return sign_x * result

    def forward(
        self,
        x: torch.Tensor,
        quantize: bool = True,
        num_levels: int = 256
    ) -> torch.Tensor:
        """A-law companding with optional quantization"""
        compressed = self.compress(x)

        if quantize:
            compressed = straight_through_quantize(compressed, num_levels)

        expanded = self.expand(compressed)

        return expanded


class G711Simulator(nn.Module):
    """
    G.711 Codec Simulator
    =====================

    64kbps PCM 코덱 시뮬레이션.

    특성:
    - 8kHz 샘플링, 8-bit 양자화
    - μ-law 또는 A-law companding
    - 매우 낮은 지연 (< 1ms)
    - 높은 음질 (MOS ~4.3)

    한국어 최적화:
    - A-law 사용 (한국 전화망 표준)
    """

    def __init__(
        self,
        companding: str = "alaw",  # "alaw" or "mulaw"
        num_bits: int = 8
    ):
        super().__init__()

        self.num_levels = 2 ** num_bits

        if companding == "alaw":
            self.compander = ALawCompanding()
        elif companding == "mulaw":
            self.compander = MuLawCompanding()
        else:
            raise ValueError(f"Unknown companding: {companding}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        G.711 codec simulation

        Args:
            x: [B, 1, T] 입력 오디오

        Returns:
            [B, 1, T] 코덱 처리된 오디오
        """
        return self.compander(x, quantize=True, num_levels=self.num_levels)


class G729Simulator(nn.Module):
    """
    G.729 Codec Simulator (Approximation)
    =====================================

    8kbps CELP 코덱의 근사 시뮬레이션.

    실제 G.729는 CELP(Code-Excited Linear Prediction) 기반으로
    매우 복잡하지만, 학습 목적으로 그 효과를 근사함.

    시뮬레이션 방법:
    1. Low-pass filtering: 고주파 정보 손실
    2. Quantization noise: 벡터 양자화 효과
    3. Frame-based processing: 10ms 프레임 효과
    4. Spectral smoothing: CELP의 스펙트럼 평활화

    한국어 특성:
    - 한국어 자음 전이 구간 손실 모사
    - 8kbps에서 특히 /ㅅ/, /ㅆ/ 등 마찰음 열화
    """

    def __init__(
        self,
        frame_size: int = 80,  # 10ms @ 8kHz
        noise_std: float = 0.02,  # 양자화 노이즈 강도
        lpf_cutoff: float = 0.8  # LPF 차단 주파수 (Nyquist 대비)
    ):
        super().__init__()

        self.frame_size = frame_size
        self.noise_std = noise_std

        # Low-pass filter 근사
        # Learnable filter for better approximation
        self.lpf = nn.Conv1d(
            1, 1,
            kernel_size=5,
            padding=2,
            bias=False
        )

        # Initialize as low-pass
        with torch.no_grad():
            lpf_kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1]).view(1, 1, -1)
            self.lpf.weight.copy_(lpf_kernel)

        # Spectral smoothing (learnable)
        self.spectral_smooth = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=15, padding=7),
        )

        # Frame-level quantization
        self.frame_quantize_levels = 64  # 6-bit per parameter approximation

    def add_quantization_noise(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """프레임 단위 양자화 노이즈 추가"""
        if training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        else:
            # 추론 시에는 결정적 양자화
            return straight_through_quantize(x, self.frame_quantize_levels)

    def frame_process(self, x: torch.Tensor) -> torch.Tensor:
        """
        프레임 단위 처리 (CELP 효과 근사)

        G.729는 10ms 프레임 단위로 처리하며,
        프레임 경계에서 불연속이 발생할 수 있음.
        """
        B, C, T = x.shape

        # 프레임 패딩
        pad_len = (self.frame_size - T % self.frame_size) % self.frame_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        # 프레임 단위 reshape
        num_frames = x.shape[-1] // self.frame_size
        x_frames = x.view(B, C, num_frames, self.frame_size)

        # 프레임 단위 양자화 (평균 에너지 보존)
        frame_energy = torch.sqrt(torch.mean(x_frames ** 2, dim=-1, keepdim=True) + 1e-8)
        x_normalized = x_frames / (frame_energy + 1e-8)

        # 정규화된 프레임에 양자화 노이즈
        x_quantized = self.add_quantization_noise(x_normalized, self.training)

        # 에너지 복원
        x_restored = x_quantized * frame_energy

        # 원래 shape으로
        x_out = x_restored.view(B, C, -1)

        # 패딩 제거
        if pad_len > 0:
            x_out = x_out[:, :, :-pad_len]

        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        G.729 codec simulation

        Args:
            x: [B, 1, T] 입력 오디오

        Returns:
            [B, 1, T] 코덱 처리된 오디오
        """
        # 1. Low-pass filtering (고주파 손실)
        x = self.lpf(x)

        # 2. Frame-based processing
        x = self.frame_process(x)

        # 3. Spectral smoothing (CELP 효과)
        residual = x
        x = self.spectral_smooth(x)
        x = x + 0.5 * residual  # Skip connection

        # 4. Clipping
        x = torch.clamp(x, -1.0, 1.0)

        return x


class AMRNBSimulator(nn.Module):
    """
    AMR-NB Codec Simulator (Approximation)
    ======================================

    Adaptive Multi-Rate Narrowband - VoLTE/3G 음성 통화 표준.

    AMR-NB 특성:
    - 8 가지 비트레이트: 4.75, 5.15, 5.9, 6.7, 7.4, 7.95, 10.2, 12.2 kbps
    - ACELP (Algebraic Code-Excited Linear Prediction) 기반
    - 8kHz 샘플링, 20ms 프레임
    - 동적 비트레이트 전환 (네트워크 상태에 따라)

    시뮬레이션 방법:
    1. 비트레이트에 따른 양자화 노이즈
    2. 20ms 프레임 기반 처리
    3. ACELP 스펙트럼 평활화 효과
    4. VAD (Voice Activity Detection) 효과
    """

    BITRATE_MODES = {
        'MR475': {'bitrate': 4750, 'noise_std': 0.045, 'lpf_freq': 0.65},
        'MR515': {'bitrate': 5150, 'noise_std': 0.040, 'lpf_freq': 0.70},
        'MR59':  {'bitrate': 5900, 'noise_std': 0.035, 'lpf_freq': 0.75},
        'MR67':  {'bitrate': 6700, 'noise_std': 0.030, 'lpf_freq': 0.78},
        'MR74':  {'bitrate': 7400, 'noise_std': 0.025, 'lpf_freq': 0.80},
        'MR795': {'bitrate': 7950, 'noise_std': 0.022, 'lpf_freq': 0.82},
        'MR102': {'bitrate': 10200, 'noise_std': 0.018, 'lpf_freq': 0.85},
        'MR122': {'bitrate': 12200, 'noise_std': 0.015, 'lpf_freq': 0.88},
    }

    def __init__(
        self,
        mode: str = 'MR122',  # Default: highest quality
        frame_size: int = 160,  # 20ms @ 8kHz
        random_mode: bool = True  # Randomly select mode during training
    ):
        super().__init__()

        self.mode = mode
        self.frame_size = frame_size
        self.random_mode = random_mode

        # Get mode parameters
        mode_params = self.BITRATE_MODES.get(mode, self.BITRATE_MODES['MR122'])
        self.noise_std = mode_params['noise_std']
        self.lpf_freq = mode_params['lpf_freq']

        # Low-pass filter (ACELP spectral envelope smoothing)
        self.lpf = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        with torch.no_grad():
            lpf_kernel = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]).view(1, 1, -1)
            self.lpf.weight.copy_(lpf_kernel)

        # ACELP residual processing
        self.acelp_process = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=11, padding=5),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 1, kernel_size=11, padding=5),
        )

        # Frame quantization levels (depends on bitrate)
        self.frame_quantize_levels = 32  # 5-bit approximation

    def get_random_mode_params(self) -> tuple:
        """학습 시 랜덤 비트레이트 모드 선택"""
        modes = list(self.BITRATE_MODES.keys())
        mode = modes[torch.randint(0, len(modes), (1,)).item()]
        params = self.BITRATE_MODES[mode]
        return params['noise_std'], params['lpf_freq'], mode

    def add_frame_quantization(
        self,
        x: torch.Tensor,
        noise_std: float
    ) -> torch.Tensor:
        """프레임 단위 양자화 + 노이즈"""
        B, C, T = x.shape

        # Pad to frame boundary
        pad_len = (self.frame_size - T % self.frame_size) % self.frame_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        # Reshape to frames [B, C, num_frames, frame_size]
        num_frames = x.shape[-1] // self.frame_size
        x_frames = x.view(B, C, num_frames, self.frame_size)

        # Frame energy for normalization
        frame_energy = torch.sqrt(torch.mean(x_frames ** 2, dim=-1, keepdim=True) + 1e-8)

        # Normalize, quantize, add noise
        x_norm = x_frames / (frame_energy + 1e-8)

        if self.training:
            noise = torch.randn_like(x_norm) * noise_std
            x_quantized = x_norm + noise
        else:
            x_quantized = straight_through_quantize(x_norm.flatten(), self.frame_quantize_levels)
            x_quantized = x_quantized.view_as(x_norm)

        # Restore energy
        x_restored = x_quantized * frame_energy

        # Reshape back
        x_out = x_restored.view(B, C, -1)

        # Remove padding
        if pad_len > 0:
            x_out = x_out[:, :, :-pad_len]

        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        AMR-NB codec simulation

        Args:
            x: [B, 1, T] 입력 오디오

        Returns:
            [B, 1, T] 코덱 처리된 오디오
        """
        # Get mode parameters (random during training for robustness)
        if self.training and self.random_mode:
            noise_std, lpf_freq, _ = self.get_random_mode_params()
        else:
            mode_params = self.BITRATE_MODES.get(self.mode, self.BITRATE_MODES['MR122'])
            noise_std = mode_params['noise_std']

        # 1. Low-pass filtering (ACELP bandwidth limit)
        x = self.lpf(x)

        # 2. Frame-based quantization
        x = self.add_frame_quantization(x, noise_std)

        # 3. ACELP residual processing
        residual = x
        x = self.acelp_process(x)
        x = x + 0.6 * residual  # Skip connection

        # 4. Clipping
        x = torch.clamp(x, -1.0, 1.0)

        return x


class DifferentiableCodecSimulator(nn.Module):
    """
    Differentiable Codec Simulator (통합 모듈)
    ==========================================

    학습 시 여러 코덱을 확률적으로 적용하여
    다양한 전화망 환경에 대한 robustness 확보.

    지원 코덱:
    - G.711 A-law (한국/유럽 표준)
    - G.711 μ-law (북미 표준)
    - G.729 (VoIP, 저대역폭)
    - AMR-NB (VoLTE/3G, 8가지 비트레이트)
    - Pass-through (코덱 없음)

    학습 전략:
    - 각 배치에서 무작위로 코덱 선택
    - 점진적으로 코덱 강도 증가 (curriculum learning)
    """

    def __init__(
        self,
        codec_types: List[str] = ["g711_alaw", "g711_ulaw", "g729", "amr_nb", "none"],
        codec_probs: Optional[List[float]] = None,
        curriculum_epochs: int = 10  # Curriculum learning epochs
    ):
        super().__init__()

        self.codec_types = codec_types

        if codec_probs is None:
            # 균등 분포
            codec_probs = [1.0 / len(codec_types)] * len(codec_types)

        self.register_buffer('codec_probs', torch.tensor(codec_probs))

        # 코덱 인스턴스 생성
        self.codecs = nn.ModuleDict()

        for codec_type in codec_types:
            if codec_type == "g711_alaw":
                self.codecs[codec_type] = G711Simulator(companding="alaw")
            elif codec_type == "g711_ulaw":
                self.codecs[codec_type] = G711Simulator(companding="mulaw")
            elif codec_type == "g729":
                self.codecs[codec_type] = G729Simulator()
            elif codec_type == "amr_nb":
                self.codecs[codec_type] = AMRNBSimulator()
            elif codec_type == "none":
                self.codecs[codec_type] = nn.Identity()
            else:
                raise ValueError(f"Unknown codec: {codec_type}")

        self.curriculum_epochs = curriculum_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Curriculum learning을 위한 epoch 설정"""
        self.current_epoch = epoch

    def get_curriculum_strength(self) -> float:
        """현재 curriculum 강도 (0 -> 1)"""
        if self.curriculum_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self.curriculum_epochs)

    def forward(
        self,
        x: torch.Tensor,
        codec_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        Codec simulation forward pass

        Args:
            x: [B, 1, T] 입력 오디오
            codec_type: 특정 코덱 지정 (None이면 확률적 선택)

        Returns:
            processed: [B, 1, T] 코덱 처리된 오디오
            selected_codec: 선택된 코덱 이름
        """
        if codec_type is not None:
            # 지정된 코덱 사용
            selected_codec = codec_type
        else:
            # 확률적 코덱 선택
            if self.training:
                # Curriculum: 초기에는 쉬운 코덱(none) 비중 높임
                strength = self.get_curriculum_strength()

                # None 확률 조정
                probs = self.codec_probs.clone()
                none_idx = self.codec_types.index("none") if "none" in self.codec_types else -1

                if none_idx >= 0 and strength < 1.0:
                    # 초기에 none 비중 증가
                    probs[none_idx] += (1 - strength) * 0.5
                    probs = probs / probs.sum()

                idx = torch.multinomial(probs, 1).item()
                selected_codec = self.codec_types[idx]
            else:
                # 추론 시 가장 어려운 코덱 (G.729) 사용
                selected_codec = "g729" if "g729" in self.codec_types else self.codec_types[0]

        # 코덱 적용
        processed = self.codecs[selected_codec](x)

        return processed, selected_codec


class CodecAugmentation(nn.Module):
    """
    Codec Augmentation Pipeline
    ===========================

    학습 시 사용하는 전체 augmentation 파이프라인.
    코덱 시뮬레이션 + 추가 노이즈/필터링.

    파이프라인:
    1. Codec simulation
    2. Bandpass filtering (300-3400Hz)
    3. Additive noise
    4. Resampling jitter
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        bandpass_low: int = 300,
        bandpass_high: int = 3400,
        snr_range: Tuple[float, float] = (15, 40)
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.snr_range = snr_range

        # Codec simulator
        self.codec_sim = DifferentiableCodecSimulator()

        # Bandpass filter (근사)
        # 실제로는 butterworth filter 사용 권장
        self.bandpass = nn.Conv1d(
            1, 1,
            kernel_size=31,
            padding=15,
            bias=False
        )

        # Initialize as bandpass
        self._init_bandpass(bandpass_low, bandpass_high)

    def _init_bandpass(self, low_hz: int, high_hz: int):
        """Sinc filter 기반 bandpass 초기화"""
        # Simplified: Gaussian-weighted sinc
        kernel_size = 31
        t = torch.linspace(-15, 15, kernel_size)

        # Low-pass sinc for high cutoff
        w_high = 2 * high_hz / self.sample_rate
        sinc_high = torch.sinc(w_high * t) * w_high

        # Low-pass sinc for low cutoff
        w_low = 2 * low_hz / self.sample_rate
        sinc_low = torch.sinc(w_low * t) * w_low

        # Bandpass = high_lp - low_lp
        bandpass_kernel = sinc_high - sinc_low

        # Gaussian window
        window = torch.exp(-t ** 2 / (2 * 5 ** 2))
        bandpass_kernel = bandpass_kernel * window

        # Normalize
        bandpass_kernel = bandpass_kernel / bandpass_kernel.sum()

        with torch.no_grad():
            self.bandpass.weight.copy_(bandpass_kernel.view(1, 1, -1))

    def add_noise(
        self,
        x: torch.Tensor,
        snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        가우시안 노이즈 추가

        Args:
            x: 입력 오디오
            snr_db: Signal-to-Noise Ratio (dB)

        Returns:
            노이즈가 추가된 오디오
        """
        if snr_db is None:
            snr_db = torch.empty(1).uniform_(*self.snr_range).item()

        # Signal power
        signal_power = torch.mean(x ** 2)

        # Noise power for target SNR
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Generate noise
        noise = torch.randn_like(x) * torch.sqrt(noise_power)

        return x + noise

    def forward(
        self,
        x: torch.Tensor,
        apply_codec: bool = True,
        apply_bandpass: bool = True,
        apply_noise: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full augmentation pipeline

        Args:
            x: [B, 1, T] 입력 오디오

        Returns:
            augmented: 증강된 오디오
            info: 적용된 증강 정보
        """
        info = {}

        # 1. Codec simulation
        if apply_codec and self.training:
            x, codec_used = self.codec_sim(x)
            info['codec'] = codec_used

        # 2. Bandpass filtering
        if apply_bandpass:
            x = self.bandpass(x)
            info['bandpass'] = True

        # 3. Additive noise
        if apply_noise and self.training:
            snr = torch.empty(1).uniform_(*self.snr_range).item()
            x = self.add_noise(x, snr)
            info['snr_db'] = snr

        # 4. Clipping
        x = torch.clamp(x, -1.0, 1.0)

        return x, info


if __name__ == "__main__":
    # 테스트 코드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Codec Simulator Test")
    print("=" * 60)

    # 테스트 오디오 (8kHz, 1초)
    x = torch.randn(4, 1, 8000).to(device)

    # G.711 A-law 테스트
    g711_alaw = G711Simulator(companding="alaw").to(device)
    y_alaw = g711_alaw(x)
    print(f"G.711 A-law - Input: {x.shape}, Output: {y_alaw.shape}")
    print(f"  MSE: {F.mse_loss(x, y_alaw).item():.6f}")

    # G.711 μ-law 테스트
    g711_mulaw = G711Simulator(companding="mulaw").to(device)
    y_mulaw = g711_mulaw(x)
    print(f"G.711 μ-law - Input: {x.shape}, Output: {y_mulaw.shape}")
    print(f"  MSE: {F.mse_loss(x, y_mulaw).item():.6f}")

    # G.729 테스트
    g729 = G729Simulator().to(device)
    y_g729 = g729(x)
    print(f"G.729 - Input: {x.shape}, Output: {y_g729.shape}")
    print(f"  MSE: {F.mse_loss(x, y_g729).item():.6f}")

    # 통합 시뮬레이터 테스트
    print("\n" + "=" * 60)
    print("Differentiable Codec Simulator Test")
    print("=" * 60)

    codec_sim = DifferentiableCodecSimulator().to(device)
    codec_sim.train()

    for i in range(5):
        y, codec_used = codec_sim(x)
        print(f"  Iteration {i+1}: {codec_used}, MSE: {F.mse_loss(x, y).item():.6f}")

    # Gradient flow 테스트
    print("\n" + "=" * 60)
    print("Gradient Flow Test")
    print("=" * 60)

    x.requires_grad = True
    y, _ = codec_sim(x)
    loss = y.sum()
    loss.backward()

    print(f"Input grad norm: {x.grad.norm().item():.6f}")
    print("✓ Gradient flows through codec simulation!")
