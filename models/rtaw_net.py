"""
CallCops: Real-Time Audio Watermarking Network
====================================================

1. Causal Architecture: 모든 Conv1d는 미래 프레임을 참조하지 않음
2. Attention-based Embedding: 마스킹 임계치 기반 비트 삽입
3. Multi-Resolution Discriminator: 자연스러운 오디오 생성

한국어 8kHz 상담 데이터 최적화:
- 40ms 프레임 (320 samples) 단위 처리
- 전화망 대역폭 (300-3400Hz) 집중
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from .attention import MaskingAwareAttention, TemporalAttention


class CausalConv1d(nn.Module):
    """
    Causal Convolution 1D Layer
    ===========================

    실시간 처리를 위한 인과적 컨볼루션.
    미래 샘플을 참조하지 않도록 좌측에만 패딩을 적용.

    수식:
        padding = (kernel_size - 1) * dilation
        output[t] = f(input[t-k+1:t+1])  # 과거 + 현재만 참조
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()

        # Causal padding: 좌측에만 패딩 적용
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # Weight normalization for stable training
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] 오디오 텐서
        Returns:
            [B, C_out, T] Causally convolved 텐서
        """
        # 좌측 패딩 후 우측 자르기 (미래 참조 방지)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    Residual Block with Causal Convolutions
    =======================================

    Skip Connection을 통한 gradient flow 개선.
    Dilation을 통해 receptive field 확장.

    구조:
        input -> CausalConv -> PReLU -> CausalConv -> + -> output
              |___________________________________|
                         (skip connection)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)

        self.activation = nn.PReLU(channels)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization for stability
        self.norm1 = nn.GroupNorm(1, channels)  # Instance norm equivalent
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Skip connection with residual learning"""
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

        # Residual connection
        return x + residual


class RTAWEncoder(nn.Module):
    """
    RTAW Encoder: 워터마크 삽입 네트워크
    =====================================

    1. Causal Conv1d: 실시간 스트리밍을 위한 인과적 구조
    2. Attention Module: 마스킹 임계치 기반 비트 삽입 위치 결정
    3. Skip Connections: Gradient vanishing 방지

    입력:
        - audio: [B, 1, T] 원본 오디오 (8kHz, 40ms = 320 samples)
        - bits: [B, N_bits] 삽입할 워터마크 비트 시퀀스

    출력:
        - watermarked_audio: [B, 1, T] 워터마크가 삽입된 오디오
        - attention_weights: [B, T] 비트 삽입 강도 맵

    한국어 최적화:
        - 8kHz 샘플링에 맞춘 커널 크기
        - 300-3400Hz 전화망 대역폭 집중
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: List[int] = [32, 64, 128, 256],
        kernel_size: int = 7,
        num_residual_blocks: int = 4,
        dilation_base: int = 2,
        attention_heads: int = 4,
        bits_dim: int = 128,  # 128-bit Cyclic Payload
        dropout: float = 0.1
    ):
        super().__init__()

        self.bits_dim = bits_dim

        # ============================================
        # 1. Audio Encoder: Downsampling Path
        # ============================================
        # 오디오를 잠재 공간으로 인코딩

        encoder_layers = []
        prev_ch = in_channels

        for i, ch in enumerate(hidden_channels):
            encoder_layers.extend([
                CausalConv1d(prev_ch, ch, kernel_size),
                nn.GroupNorm(1, ch),
                nn.PReLU(ch),
            ])
            prev_ch = ch

        self.audio_encoder = nn.Sequential(*encoder_layers)

        # ============================================
        # 2. Bit Embedding: 워터마크 비트 -> 잠재 벡터
        # ============================================
        # 128-bit payload를 hidden dimension으로 확장

        self.bit_embedding = nn.Sequential(
            nn.Linear(bits_dim, hidden_channels[-1]),
            nn.PReLU(),
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.PReLU(),
        )

        # ============================================
        # 3. Attention Module: 마스킹 임계치 분석
        # ============================================
        # 사람 귀가 인지하기 어려운 고에너지 구간 탐지

        self.masking_attention = MaskingAwareAttention(
            embed_dim=hidden_channels[-1],
            num_heads=attention_heads,
            dropout=dropout
        )

        # ============================================
        # 4. Fusion Layer: Audio + Watermark 결합
        # ============================================

        self.fusion = nn.Sequential(
            CausalConv1d(hidden_channels[-1] * 2, hidden_channels[-1], kernel_size=3),
            nn.GroupNorm(1, hidden_channels[-1]),
            nn.PReLU(hidden_channels[-1]),
        )

        # ============================================
        # 5. Residual Blocks: 정밀한 섭동 학습
        # ============================================

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                channels=hidden_channels[-1],
                kernel_size=kernel_size,
                dilation=dilation_base ** (i % 4),  # 1, 2, 4, 8 cycling
                dropout=dropout
            )
            for i in range(num_residual_blocks)
        ])

        # ============================================
        # 6. Audio Decoder: Upsampling Path
        # ============================================
        # 잠재 공간에서 오디오로 복원

        decoder_layers = []
        reversed_channels = list(reversed(hidden_channels))

        for i, ch in enumerate(reversed_channels[1:]):
            decoder_layers.extend([
                CausalConv1d(reversed_channels[i], ch, kernel_size),
                nn.GroupNorm(1, ch),
                nn.PReLU(ch),
            ])

        # Final layer: 단일 채널 오디오 출력
        decoder_layers.append(
            CausalConv1d(hidden_channels[0], in_channels, kernel_size)
        )

        self.audio_decoder = nn.Sequential(*decoder_layers)

        # ============================================
        # 7. Perturbation Scaling: 섭동 강도 제어
        # ============================================
        # 작은 섭동만 허용하여 음질 보존

        self.perturbation_scale = nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        audio: torch.Tensor,
        bits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        워터마크 삽입 Forward Pass

        Args:
            audio: [B, 1, T] 원본 오디오
            bits: [B, bits_dim] 워터마크 비트 (0/1 또는 -1/1)

        Returns:
            watermarked: [B, 1, T] 워터마크된 오디오
            attention_weights: [B, T] 삽입 강도 맵
        """
        B, _, T = audio.shape

        # 1. 오디오 인코딩
        audio_features = self.audio_encoder(audio)  # [B, C, T']
        T_feat = audio_features.shape[-1]

        # 2. 비트 임베딩 및 시간축 확장
        bit_embed = self.bit_embedding(bits)  # [B, C]
        bit_embed = bit_embed.unsqueeze(-1).expand(-1, -1, T_feat)  # [B, C, T']

        # 3. Attention 기반 삽입 위치 결정
        # 마스킹 임계치가 높은(= 변화를 감지하기 어려운) 구간에 집중
        audio_attended, attention_weights = self.masking_attention(
            audio_features.transpose(1, 2)  # [B, T', C]
        )
        audio_attended = audio_attended.transpose(1, 2)  # [B, C, T']

        # 4. Audio + Watermark Fusion
        fused = torch.cat([audio_attended, bit_embed], dim=1)  # [B, 2C, T']
        fused = self.fusion(fused)  # [B, C, T']

        # 5. Residual Processing
        for block in self.residual_blocks:
            fused = block(fused)

        # 6. 섭동 디코딩
        perturbation = self.audio_decoder(fused)  # [B, 1, T]

        # T 길이 맞추기 (causal padding으로 인한 길이 변화 보정)
        if perturbation.shape[-1] != T:
            perturbation = F.interpolate(perturbation, size=T, mode='linear')

        # 7. Scaled perturbation 적용
        # 작은 섭동으로 음질 보존 (PESQ >= 4.0 목표)
        perturbation = torch.tanh(perturbation) * self.perturbation_scale

        # 8. 원본 + 섭동 = 워터마크된 오디오
        watermarked = audio + perturbation

        # Clipping to valid audio range [-1, 1]
        watermarked = torch.clamp(watermarked, -1.0, 1.0)

        return watermarked, attention_weights


class RTAWDecoder(nn.Module):
    """
    RTAW Decoder: 워터마크 추출 네트워크
    =====================================

    오디오에서 비트 확률값과 탐지 신뢰도를 추출하는 Binary Classifier.

    출력:
    - bit_probs: [B, N_bits] 각 비트의 1일 확률
    - detection_confidence: [B, 1] 워터마크 탐지 신뢰도
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: List[int] = [32, 64, 128, 256],
        kernel_size: int = 5,
        num_residual_blocks: int = 4,
        bits_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.bits_dim = bits_dim

        # ============================================
        # 1. Feature Extractor: Causal Conv Stack
        # ============================================

        encoder_layers = []
        prev_ch = in_channels

        for ch in hidden_channels:
            encoder_layers.extend([
                CausalConv1d(prev_ch, ch, kernel_size),
                nn.GroupNorm(1, ch),
                nn.PReLU(ch),
            ])
            prev_ch = ch

        self.feature_extractor = nn.Sequential(*encoder_layers)

        # ============================================
        # 2. Residual Blocks
        # ============================================

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                channels=hidden_channels[-1],
                kernel_size=kernel_size,
                dilation=2 ** (i % 4),
                dropout=dropout
            )
            for i in range(num_residual_blocks)
        ])

        # ============================================
        # 3. Temporal Attention: 시간축 집약
        # ============================================

        self.temporal_attention = TemporalAttention(
            embed_dim=hidden_channels[-1],
            dropout=dropout
        )

        # ============================================
        # 4. Bit Classifier: Binary Classification
        # ============================================

        self.bit_classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], bits_dim),
            nn.Sigmoid()  # [0, 1] 확률 출력
        )

        # ============================================
        # 5. Detection Head: 워터마크 존재 여부
        # ============================================

        self.detection_head = nn.Sequential(
            nn.Linear(hidden_channels[-1], 64),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        워터마크 추출 Forward Pass

        Args:
            audio: [B, 1, T] 워터마크된 (또는 원본) 오디오

        Returns:
            bit_probs: [B, bits_dim] 각 비트의 확률
            detection_conf: [B, 1] 탐지 신뢰도
        """
        # 1. Feature extraction
        features = self.feature_extractor(audio)  # [B, C, T']

        # 2. Residual processing
        for block in self.residual_blocks:
            features = block(features)

        # 3. Temporal aggregation via attention
        features = features.transpose(1, 2)  # [B, T', C]
        aggregated = self.temporal_attention(features)  # [B, C]

        # 4. Bit classification
        bit_probs = self.bit_classifier(aggregated)  # [B, bits_dim]

        # 5. Detection confidence
        detection_conf = self.detection_head(aggregated)  # [B, 1]

        return bit_probs, detection_conf


class MultiResolutionDiscriminator(nn.Module):
    """
    Multi-Resolution Discriminator
    ==============================

    오디오의 자연스러움을 판별하는 GAN Discriminator.
    여러 해상도에서 분석하여 다양한 주파수 대역의 artifacts 탐지.

    구조:
    - 원본 해상도 판별기
    - 2x 다운샘플링 판별기
    - 4x 다운샘플링 판별기

    이를 통해:
    - 고주파 artifacts (aliasing, clicking)
    - 저주파 artifacts (envelope distortion)
    모두 탐지 가능
    """

    def __init__(
        self,
        resolutions: List[int] = [1, 2, 4],  # Downsampling factors
        channels: List[int] = [32, 64, 128, 256],
        kernel_size: int = 5
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            self._build_discriminator(channels, kernel_size)
            for _ in resolutions
        ])

        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=res, stride=res) if res > 1 else nn.Identity()
            for res in resolutions
        ])

    def _build_discriminator(
        self,
        channels: List[int],
        kernel_size: int
    ) -> nn.Module:
        """단일 해상도 판별기 구축"""
        layers = []
        prev_ch = 1

        for i, ch in enumerate(channels):
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size, stride=2, padding=kernel_size // 2),
                nn.GroupNorm(1, ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            prev_ch = ch

        # Final classification
        layers.append(nn.Conv1d(prev_ch, 1, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    def forward(
        self,
        audio: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Multi-resolution discrimination

        Args:
            audio: [B, 1, T]

        Returns:
            List of discrimination scores at each resolution
        """
        outputs = []

        for downsampler, discriminator in zip(self.downsamplers, self.discriminators):
            x = downsampler(audio)
            out = discriminator(x)
            outputs.append(out)

        return outputs


class RTAWNet(nn.Module):
    """
    RTAW Complete Network
    =====================

    Encoder + Decoder + Discriminator를 통합한 전체 네트워크.
    End-to-end 학습 및 추론 지원.

    학습 모드:
    - Encoder: 워터마크 삽입
    - Decoder: 워터마크 추출
    - Discriminator: 자연스러움 판별
    - Codec Simulator: 전화망 robustness

    추론 모드:
    - embed(): 워터마크 삽입만
    - extract(): 워터마크 추출만
    """

    def __init__(
        self,
        encoder_config: Optional[dict] = None,
        decoder_config: Optional[dict] = None,
        discriminator_config: Optional[dict] = None,
        bits_dim: int = 128
    ):
        super().__init__()

        encoder_config = encoder_config or {}
        decoder_config = decoder_config or {}
        discriminator_config = discriminator_config or {}

        self.encoder = RTAWEncoder(bits_dim=bits_dim, **encoder_config)
        self.decoder = RTAWDecoder(bits_dim=bits_dim, **decoder_config)
        self.discriminator = MultiResolutionDiscriminator(**discriminator_config)

        self.bits_dim = bits_dim

    def embed(
        self,
        audio: torch.Tensor,
        bits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        워터마크 삽입 (추론용)

        Args:
            audio: [B, 1, T] 원본 오디오
            bits: [B, bits_dim] 워터마크 비트

        Returns:
            watermarked: [B, 1, T] 워터마크된 오디오
            attention: [B, T] 삽입 강도 맵
        """
        return self.encoder(audio, bits)

    def extract(
        self,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        워터마크 추출 (추론용)

        Args:
            audio: [B, 1, T] 워터마크된 오디오

        Returns:
            bit_probs: [B, bits_dim] 비트 확률
            detection: [B, 1] 탐지 신뢰도
        """
        return self.decoder(audio)

    def forward(
        self,
        audio: torch.Tensor,
        bits: torch.Tensor,
        return_discriminator: bool = True
    ) -> dict:
        """
        전체 Forward Pass (학습용)

        Args:
            audio: [B, 1, T] 원본 오디오
            bits: [B, bits_dim] 워터마크 비트
            return_discriminator: Discriminator 출력 포함 여부

        Returns:
            dict containing:
                - watermarked: 워터마크된 오디오
                - attention: 삽입 강도 맵
                - bit_probs: 추출된 비트 확률
                - detection: 탐지 신뢰도
                - disc_real: 원본 판별 점수 (optional)
                - disc_fake: 워터마크 판별 점수 (optional)
        """
        # 1. Embed watermark
        watermarked, attention = self.encoder(audio, bits)

        # 2. Extract watermark
        bit_probs, detection = self.decoder(watermarked)

        result = {
            'watermarked': watermarked,
            'attention': attention,
            'bit_probs': bit_probs,
            'detection': detection,
        }

        # 3. Discriminate (for GAN loss)
        if return_discriminator:
            with torch.no_grad():
                disc_real = self.discriminator(audio)
            disc_fake = self.discriminator(watermarked)

            result['disc_real'] = disc_real
            result['disc_fake'] = disc_fake

        return result

    def get_causal_receptive_field(self) -> int:
        """
        Causal receptive field 계산 (지연 시간 추정용)

        Returns:
            총 receptive field (samples)
        """
        # 대략적인 계산 (실제 구조에 따라 조정 필요)
        kernel_size = 7
        num_layers = 8
        dilation_base = 2

        receptive_field = 0
        for i in range(num_layers):
            dilation = dilation_base ** (i % 4)
            receptive_field += (kernel_size - 1) * dilation

        return receptive_field

    def estimate_latency_ms(self, sample_rate: int = 8000) -> float:
        """
        추정 지연 시간 (ms)

        Args:
            sample_rate: 샘플링 레이트

        Returns:
            지연 시간 (ms)
        """
        receptive_field = self.get_causal_receptive_field()
        return (receptive_field / sample_rate) * 1000


if __name__ == "__main__":
    # 테스트 코드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 초기화
    model = RTAWNet(bits_dim=128).to(device)

    # 테스트 입력 (8kHz, 40ms = 320 samples, batch=4)
    batch_size = 4
    audio = torch.randn(batch_size, 1, 320).to(device)
    bits = torch.randint(0, 2, (batch_size, 128)).float().to(device)

    # Forward pass
    output = model(audio, bits)

    print("=" * 60)
    print("CallCops Model Test")
    print("=" * 60)
    print(f"Input audio shape: {audio.shape}")
    print(f"Input bits shape: {bits.shape}")
    print(f"Watermarked audio shape: {output['watermarked'].shape}")
    print(f"Bit probs shape: {output['bit_probs'].shape}")
    print(f"Detection shape: {output['detection'].shape}")
    print(f"Estimated latency: {model.estimate_latency_ms():.2f} ms")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
