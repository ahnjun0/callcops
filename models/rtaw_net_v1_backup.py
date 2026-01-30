"""
CallCops: Real-Time Audio Watermarking Network
==============================================

경량 딥러닝 아키텍처 for 8kHz 전화망 오디오 워터마킹.

설계 목표:
- 실시간 추론을 위한 경량 구조
- Residual Connections으로 gradient vanishing 방지
- 가변 길이 오디오 지원 (AdaptiveAvgPool1d)
- 8kHz 샘플레이트 최적화

아키텍처:
    Encoder: Audio [B,1,T] + Message [B,128] → Watermarked [B,1,T]
    Decoder: Watermarked [B,1,T] → Predicted Bits [B,128]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """
    기본 Convolutional Block

    Conv1d → BatchNorm → LeakyReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual Block with Skip Connection

    구조:
        input ─→ Conv → BN → ReLU → Conv → BN ─→ (+) → ReLU → output
              └────────────────────────────────┘
                        (skip connection)

    Gradient vanishing 방지 및 깊은 네트워크 학습 안정화
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection
        out = out + residual
        out = self.act(out)

        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    채널별 중요도 학습을 통한 feature recalibration.
    경량 attention 메커니즘.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # Changed from ReLU to prevent gradient clipping
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        b, c, _ = x.shape

        # Squeeze: Global average pooling
        y = self.squeeze(x).view(b, c)

        # Excitation: FC layers
        y = self.excitation(y).view(b, c, 1)

        # Scale
        return x * y


class CrossModalFusionBlock(nn.Module):
    """
    Cross-Modal Fusion Block for Audio-Message Interaction (Memory Efficient)
    ==========================================================================
    
    O(T) 복잡도의 Linear Attention으로 메모리 효율적 구현.
    Audio [B, C, T]가 Global Message Vector [B, C, 1]에 attend.
    
    구조:
    1. Message → Global Vector Projection [B, C, 1]
    2. Linear Attention: Audio가 Global Message를 query (O(T) not O(T²))
    3. Bit Pattern Modulation: 128비트 패턴 주입
    4. Gated Fusion: 두 modality 결합
    
    Memory: [B, T, 1] attention map instead of [B, T, T]
    """
    
    def __init__(self, audio_channels: int, message_dim: int = 128):
        super().__init__()
        
        self.audio_channels = audio_channels
        self.message_dim = message_dim
        self.head_dim = audio_channels // 4
        
        # Message를 Global Vector로 projection [B, 128] → [B, C]
        self.message_proj = nn.Sequential(
            nn.Linear(message_dim, audio_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(audio_channels, audio_channels),
        )
        
        # Bit-wise temporal pattern generator
        # 각 비트가 고유한 temporal pattern을 생성하도록 학습
        self.bit_pattern_conv = nn.Conv1d(
            message_dim, audio_channels,
            kernel_size=1, bias=False
        )
        
        # Linear Attention: Audio queries Global Message (O(T) complexity)
        # Query: from audio features [B, C, T] → [B, head_dim, T]
        self.query_proj = nn.Conv1d(audio_channels, self.head_dim, kernel_size=1)
        
        # Key & Value: from GLOBAL message vector [B, C] → [B, head_dim] / [B, C]
        # NOT from expanded [B, C, T] - this is the key difference!
        self.key_proj = nn.Linear(audio_channels, self.head_dim)
        self.value_proj = nn.Linear(audio_channels, audio_channels)
        
        # Per-position modulation (content-aware gating)
        self.position_gate = nn.Sequential(
            nn.Conv1d(audio_channels, audio_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Conv1d(audio_channels * 2, audio_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Conv1d(audio_channels * 2, audio_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(audio_channels)
        
    def forward(
        self,
        audio_feat: torch.Tensor,  # [B, C, T]
        message: torch.Tensor       # [B, 128]
    ) -> torch.Tensor:
        B, C, T = audio_feat.shape
        
        # 1. Global message projection (single vector, NOT expanded)
        msg_global = self.message_proj(message)  # [B, C]
        
        # 2. Bit-wise temporal patterns
        msg_bits = message.unsqueeze(-1).expand(-1, -1, T)  # [B, 128, T]
        bit_patterns = self.bit_pattern_conv(msg_bits)  # [B, C, T]
        
        # 3. Linear Attention: Audio attends to GLOBAL message (O(T) complexity)
        # Query from audio: [B, head_dim, T]
        Q = self.query_proj(audio_feat)  # [B, head_dim, T]
        
        # Key & Value from GLOBAL message vector (shape [B, C], NOT [B, C, T])
        K = self.key_proj(msg_global)    # [B, head_dim]
        V = self.value_proj(msg_global)  # [B, C]
        
        # Attention scores: each audio position attends to the single global message
        # Q: [B, head_dim, T] → permute → [B, T, head_dim]
        # K: [B, head_dim] → unsqueeze → [B, head_dim, 1]
        # Result: [B, T, 1] - each of T positions has ONE attention weight
        attn = torch.bmm(Q.permute(0, 2, 1), K.unsqueeze(-1))  # [B, T, 1]
        attn = attn / (self.head_dim ** 0.5)
        attn = torch.sigmoid(attn)  # Sigmoid for single-target attention (not softmax)
        
        # Attended value: broadcast V [B, C] across T positions, modulated by attn [B, T, 1]
        # V: [B, C] → [B, C, 1] → [B, C, T] via broadcast
        # attn: [B, T, 1] → [B, 1, T] for channel-wise multiplication
        attended = V.unsqueeze(-1) * attn.permute(0, 2, 1)  # [B, C, T]
        
        # 4. Content-aware position gating
        pos_gate = self.position_gate(audio_feat)  # [B, C, T]
        modulated = pos_gate * (attended + bit_patterns)
        
        # 5. Gated fusion
        combined = torch.cat([audio_feat, modulated], dim=1)  # [B, 2C, T]
        gate = self.gate(combined)  # [B, C, T]
        
        fused = gate * audio_feat + (1 - gate) * modulated
        
        # 6. Output projection
        output = self.output_conv(torch.cat([fused, audio_feat], dim=1))
        output = self.norm(output)
        
        return output


class TemporalBitExtractor(nn.Module):
    """
    Temporal-Aware Bit Extractor
    ============================
    
    AdaptiveAvgPool1d(1)을 대체하여 temporal 정보를 보존하면서
    128-bit를 추출. 각 비트가 특정 temporal region에 대응.
    
    구조:
    1. Temporal segmentation: T → 128 segments
    2. Per-segment feature extraction
    3. Bit-wise classification
    """
    
    def __init__(self, in_channels: int, num_bits: int = 128):
        super().__init__()
        
        self.num_bits = num_bits
        
        # Adaptive pooling to fixed temporal resolution
        # 128보다 큰 temporal resolution 유지 (정보 보존)
        self.temporal_pool = nn.AdaptiveAvgPool1d(num_bits * 2)  # 256 time steps
        
        # Temporal feature refinement
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Reduce temporal dimension: 256 → 128 with learned weights
        self.bit_pool = nn.Conv1d(
            in_channels // 2, in_channels // 2,
            kernel_size=2, stride=2, groups=in_channels // 2
        )
        
        # Per-bit classifier
        # 각 temporal position이 하나의 비트에 대응
        self.bit_classifier = nn.Conv1d(
            in_channels // 2, 1,
            kernel_size=1
        )
        
        # Global context for bit correlation
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels // 2, num_bits),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] feature map (variable length)
            
        Returns:
            logits: [B, 128] bit logits
        """
        # 1. Fixed temporal resolution
        x = self.temporal_pool(x)  # [B, C, 256]
        
        # 2. Temporal feature refinement
        x = self.temporal_conv(x)  # [B, C//2, 256]
        
        # 3. Reduce to 128 time steps
        x = self.bit_pool(x)  # [B, C//2, 128]
        
        # 4. Per-position bit prediction
        local_logits = self.bit_classifier(x).squeeze(1)  # [B, 128]
        
        # 5. Global context (bit correlation)
        global_logits = self.global_context(x)  # [B, 128]
        
        # 6. Combine local and global
        logits = local_logits + 0.5 * global_logits
        
        return logits


# =============================================================================
# Encoder (Embedding Network)
# =============================================================================

class Encoder(nn.Module):
    """
    Watermark Embedding Network
    ===========================

    오디오에 128-bit 메시지를 삽입하는 인코더.

    구조:
    1. Audio Feature Extraction: Conv1d 스택으로 오디오 특징 추출
    2. Message Projection: 128-bit → high-dim 벡터로 확장
    3. Feature Fusion: 오디오 특징 + 메시지 특징 결합
    4. Residual Refinement: Residual blocks로 정밀 조정
    5. Output: tanh 활성화로 [-1, 1] 범위 출력

    입력:
        audio: [B, 1, T] - 8kHz 오디오
        message: [B, 128] - 워터마크 비트

    출력:
        watermarked: [B, 1, T] - 워터마크 삽입된 오디오
    """

    def __init__(
        self,
        message_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 4
    ):
        super().__init__()

        self.message_dim = message_dim
        self.hidden_channels = hidden_channels

        # =============================================
        # 1. Audio Encoder (Downsampling Path)
        # =============================================
        # Conv1d로 오디오 특징 추출

        encoder_layers = []
        in_ch = 1

        for out_ch in hidden_channels:
            encoder_layers.append(ConvBlock(in_ch, out_ch, kernel_size=7, padding=3))
            in_ch = out_ch

        self.audio_encoder = nn.Sequential(*encoder_layers)

        # =============================================
        # 2. Cross-Modal Fusion (Replaces simple message projection)
        # =============================================
        # 강화된 Audio-Message 상호작용
        
        self.cross_modal_fusion = CrossModalFusionBlock(
            audio_channels=hidden_channels[-1],
            message_dim=message_dim
        )
        
        # Post-fusion refinement
        self.fusion_refine = nn.Sequential(
            ConvBlock(hidden_channels[-1], hidden_channels[-1], kernel_size=3, padding=1),
            SEBlock(hidden_channels[-1]),
        )

        # =============================================
        # 4. Residual Refinement Blocks
        # =============================================
        # Dilated convolutions으로 넓은 receptive field

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                channels=hidden_channels[-1],
                kernel_size=3,
                dilation=2 ** (i % 4)  # 1, 2, 4, 8 cycling
            )
            for i in range(num_residual_blocks)
        ])

        # =============================================
        # 5. Audio Decoder (Upsampling Path)
        # =============================================
        # 특징 → 오디오 복원

        decoder_layers = []
        reversed_channels = list(reversed(hidden_channels))

        for i, out_ch in enumerate(reversed_channels[1:]):
            in_ch = reversed_channels[i]
            decoder_layers.append(ConvBlock(in_ch, out_ch, kernel_size=7, padding=3))

        self.audio_decoder = nn.Sequential(*decoder_layers)

        # =============================================
        # 6. Final Output Layer
        # =============================================
        # tanh로 [-1, 1] 범위 보장

        self.output_conv = nn.Conv1d(hidden_channels[0], 1, kernel_size=7, padding=3)

        # Perturbation scale (학습 가능, but CLAMPED to prevent vanishing)
        # alpha_min=0.01 ensures encoder MUST inject some signal
        # alpha_max=0.3 prevents excessive distortion
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # Will be transformed
        self.alpha_min = 0.01
        self.alpha_max = 0.3

    def forward(
        self,
        audio: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        """
        워터마크 삽입

        Args:
            audio: [B, 1, T] 원본 오디오
            message: [B, 128] 워터마크 비트 (0/1 또는 float)

        Returns:
            watermarked: [B, 1, T] 워터마크된 오디오
        """
        B, _, T = audio.shape

        # 1. Audio feature extraction
        audio_feat = self.audio_encoder(audio)  # [B, C, T]

        # 2. Cross-modal fusion (replaces simple expand)
        # 강화된 message-audio 상호작용
        fused = self.cross_modal_fusion(audio_feat, message)  # [B, C, T]
        
        # 3. Post-fusion refinement
        fused = self.fusion_refine(fused)  # [B, C, T]

        # 4. Residual refinement
        for block in self.residual_blocks:
            fused = block(fused)

        # 5. Decode to audio
        decoded = self.audio_decoder(fused)  # [B, C0, T]

        # 6. Final output
        perturbation = self.output_conv(decoded)  # [B, 1, T]

        # 길이 맞추기
        if perturbation.shape[-1] != T:
            perturbation = F.interpolate(perturbation, size=T, mode='linear', align_corners=False)

        # Compute clamped alpha (CRITICAL: prevents vanishing to zero)
        # sigmoid maps (-inf, inf) → (0, 1), then scale to [alpha_min, alpha_max]
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_raw)
        
        # tanh로 범위 제한 + clamped scale
        # Small scaling factor (0.1) on tanh output for finer control
        perturbation = 0.1 * torch.tanh(perturbation) * alpha

        # 원본 + 섭동
        watermarked = audio + perturbation
        watermarked = torch.clamp(watermarked, -1.0, 1.0)

        return watermarked


# =============================================================================
# Decoder (Extraction Network)
# =============================================================================

class Decoder(nn.Module):
    """
    Watermark Extraction Network
    ============================

    워터마크된 오디오에서 128-bit 메시지를 추출하는 디코더.

    구조:
    1. Feature Extraction: Conv1d 스택 (점진적 채널 확장)
    2. Residual Blocks: 깊은 특징 학습
    3. Global Pooling: AdaptiveAvgPool1d로 가변 길이 지원
    4. Classifier: Linear + Sigmoid로 128-bit 출력

    입력:
        audio: [B, 1, T] - 워터마크된 오디오 (가변 길이)

    출력:
        message: [B, 128] - 추출된 비트 확률 (0~1)
    """

    def __init__(
        self,
        message_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 4
    ):
        super().__init__()

        self.message_dim = message_dim

        # =============================================
        # 1. Feature Extractor
        # =============================================
        # Progressive channel expansion

        encoder_layers = []
        in_ch = 1

        for out_ch in hidden_channels:
            encoder_layers.append(
                ConvBlock(in_ch, out_ch, kernel_size=5, stride=2, padding=2)
            )
            in_ch = out_ch

        self.feature_extractor = nn.Sequential(*encoder_layers)

        # =============================================
        # 2. Residual Blocks
        # =============================================

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                channels=hidden_channels[-1],
                kernel_size=3,
                dilation=2 ** (i % 4)
            )
            for i in range(num_residual_blocks)
        ])

        # SE Block for channel attention
        self.se_block = SEBlock(hidden_channels[-1])

        # =============================================
        # 3. Temporal Bit Extractor (Replaces global pooling)
        # =============================================
        # AdaptiveAvgPool1d(1) 대신 temporal 정보 보존하는 구조
        
        self.bit_extractor = TemporalBitExtractor(
            in_channels=hidden_channels[-1],
            num_bits=message_dim
        )
        
        # Legacy classifier removed - TemporalBitExtractor handles classification
        # =============================================
        # 4. Message Classifier (REMOVED - integrated into TemporalBitExtractor)
        # =============================================
        # Keeping a simple refinement layer for compatibility
        self.classifier_refine = nn.Sequential(
            nn.Linear(message_dim, message_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(message_dim * 2, message_dim),
            # nn.Sigmoid() 제거: 수치적 안정성을 위해 로짓(Logit)을 직접 출력하고 
            # 손실 함수에서 binary_cross_entropy_with_logits를 사용함.
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        워터마크 추출 (로짓 출력)

        Args:
            audio: [B, 1, T] 워터마크된 오디오 (가변 길이)

        Returns:
            logits: [B, 128] 추출된 비트의 로짓 값
        """
        # 1. Feature extraction
        features = self.feature_extractor(audio)  # [B, C, T']

        # 2. Residual blocks
        for block in self.residual_blocks:
            features = block(features)

        # 3. Channel attention
        features = self.se_block(features)

        # 4. Temporal bit extraction (replaces global pooling)
        # 각 비트가 특정 temporal region에서 추출됨
        bit_logits = self.bit_extractor(features)  # [B, 128]

        # 5. Refinement (residual connection)
        logits = bit_logits + self.classifier_refine(bit_logits)  # [B, 128]

        return logits


# =============================================================================
# Discriminator (for GAN training)
# =============================================================================

class Discriminator(nn.Module):
    """
    Multi-Scale Discriminator
    =========================

    워터마크된 오디오의 자연스러움을 판별.
    여러 스케일에서 분석하여 다양한 주파수 대역의 artifacts 탐지.
    """

    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        channels: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            self._build_discriminator(channels)
            for _ in scales
        ])

        self.pools = nn.ModuleList([
            nn.AvgPool1d(scale, stride=scale) if scale > 1 else nn.Identity()
            for scale in scales
        ])

    def _build_discriminator(self, channels: List[int]) -> nn.Module:
        layers = []
        in_ch = 1

        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch

        layers.append(nn.Conv1d(in_ch, 1, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            List of discrimination scores at each scale
        """
        outputs = []

        for pool, disc in zip(self.pools, self.discriminators):
            x = pool(audio)
            out = disc(x)
            outputs.append(out)

        return outputs


# =============================================================================
# CallCopsNet (Wrapper Class)
# =============================================================================

class CallCopsNet(nn.Module):
    """
    CallCops Complete Network
    =========================

    Encoder + Decoder + Discriminator 통합 네트워크.

    Features:
    - embed(): 워터마크 삽입
    - extract(): 워터마크 추출
    - forward(): End-to-end 학습용

    Usage:
        model = CallCopsNet(message_dim=128)

        # Embedding
        watermarked = model.embed(audio, bits)

        # Extraction
        extracted_bits = model.extract(watermarked)

        # Training
        outputs = model(audio, bits)
    """

    def __init__(
        self,
        message_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 4,
        use_discriminator: bool = True
    ):
        super().__init__()

        self.message_dim = message_dim

        # Encoder (Embedding Network)
        self.encoder = Encoder(
            message_dim=message_dim,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks
        )

        # Decoder (Extraction Network)
        self.decoder = Decoder(
            message_dim=message_dim,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks
        )

        # Discriminator (optional, for GAN training)
        self.discriminator = None
        if use_discriminator:
            self.discriminator = Discriminator()

    def embed(
        self,
        audio: torch.Tensor,
        message: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        워터마크 삽입

        Args:
            audio: [B, 1, T] 원본 오디오
            message: [B, 128] 워터마크 비트

        Returns:
            watermarked: [B, 1, T] 워터마크된 오디오
            attention: None (placeholder for compatibility)
        """
        watermarked = self.encoder(audio, message)
        return watermarked, None

    def extract(
        self,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        워터마크 추출 (추론용)

        Args:
            audio: [B, 1, T] 워터마크된 오디오

        Returns:
            probs: [B, 128] 추출된 비트 확률 (0~1)
            detection: [B, 1] 워터마크 탐지 신뢰도 (0~1)
        """
        logits = self.decoder(audio)
        probs = torch.sigmoid(logits)
        
        # Detection: 0.5에서 얼마나 떨어져 있는지로 판단 (0 or 1에 가까우면 신뢰도 높음)
        # 워터마크가 있으면 신뢰도가 1에 가깝고, 없으면(랜덤하면) 0에 가깝게 정규화
        detection = torch.abs(probs - 0.5).mean(dim=1, keepdim=True) * 2
        return probs, detection

    def forward(
        self,
        audio: torch.Tensor,
        message: torch.Tensor,
        return_discriminator: bool = True
    ) -> dict:
        """
        End-to-end Forward Pass (학습용)

        Args:
            audio: [B, 1, T] 원본 오디오
            message: [B, 128] 워터마크 비트
            return_discriminator: Discriminator 출력 포함 여부

        Returns:
            dict:
                - watermarked: 워터마크된 오디오
                - extracted: 추출된 비트 확률
                - disc_real: 원본 판별 점수 (optional)
                - disc_fake: 워터마크 판별 점수 (optional)
        """
        # 1. Embed watermark
        watermarked = self.encoder(audio, message)

        # 2. Extract watermark
        extracted = self.decoder(watermarked)

        result = {
            'watermarked': watermarked,
            'extracted': extracted,
        }

        # 3. Discriminate (for GAN loss)
        if return_discriminator and self.discriminator is not None:
            with torch.no_grad():
                disc_real = self.discriminator(audio)
            disc_fake = self.discriminator(watermarked)

            result['disc_real'] = disc_real
            result['disc_fake'] = disc_fake

        return result

    def count_parameters(self) -> dict:
        """모델 파라미터 수 계산"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters()) if self.discriminator else 0

        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'discriminator': disc_params,
            'total': encoder_params + decoder_params + disc_params
        }


# =============================================================================
# Legacy Compatibility (기존 코드 호환성)
# =============================================================================

# 기존 RTAWNet 호환성 유지
class RTAWNet(CallCopsNet):
    """RTAWNet alias for backward compatibility"""

    def __init__(self, bits_dim: int = 128, **kwargs):
        super().__init__(message_dim=bits_dim, **kwargs)
        self.bits_dim = bits_dim

RTAWEncoder = Encoder
RTAWDecoder = Decoder
MultiResolutionDiscriminator = Discriminator


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CallCops Network Architecture Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model initialization
    model = CallCopsNet(
        message_dim=128,
        hidden_channels=[32, 64, 128, 256],
        num_residual_blocks=4
    ).to(device)

    # Count parameters
    params = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Discriminator: {params['discriminator']:,}")
    print(f"  Total: {params['total']:,}")

    # Test inputs
    batch_size = 4
    audio_length = 8000  # 1 second @ 8kHz

    audio = torch.randn(batch_size, 1, audio_length).to(device)
    message = torch.randint(0, 2, (batch_size, 128)).float().to(device)

    print(f"\nInput Shapes:")
    print(f"  Audio: {audio.shape}")
    print(f"  Message: {message.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(audio, message)

    print(f"\nOutput Shapes:")
    print(f"  Watermarked: {outputs['watermarked'].shape}")
    print(f"  Extracted: {outputs['extracted'].shape}")

    # Test embed/extract API
    print(f"\n[Embed/Extract API Test]")
    with torch.no_grad():
        wm, attention = model.embed(audio, message)
        ext, detection = model.extract(wm)

    print(f"  Embed returns: watermarked={wm.shape}, attention={attention}")
    print(f"  Extract returns: bits={ext.shape}, detection={detection.shape}")

    # Test variable length
    print(f"\n[Variable Length Test]")
    for length in [320, 1600, 8000, 16000]:
        test_audio = torch.randn(1, 1, length).to(device)
        test_msg = torch.randint(0, 2, (1, 128)).float().to(device)

        with torch.no_grad():
            wm, _ = model.embed(test_audio, test_msg)
            ext, det = model.extract(wm)

        print(f"  Length {length:5d}: watermarked={wm.shape}, extracted={ext.shape}, detection={det.shape}")

    # SNR estimation
    original = audio[0, 0].cpu()
    watermarked = outputs['watermarked'][0, 0].cpu()
    perturbation = watermarked - original

    snr = 10 * torch.log10(
        torch.mean(original ** 2) / (torch.mean(perturbation ** 2) + 1e-10)
    )

    print(f"\nQuality Metrics:")
    print(f"  Perturbation range: [{perturbation.min():.4f}, {perturbation.max():.4f}]")
    print(f"  Estimated SNR: {snr:.1f} dB")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
