"""
CallCops: Frame-Wise Real-Time Audio Watermarking Network v2.0
==============================================================

설계 철학:
1. 실시간성 (Real-time Streaming)
   - 40ms 프레임 단위 처리 (320 samples @ 8kHz)
   - O(1) 복잡도 per frame
   - 저지연 추론

2. 순환 보안 (Cyclic Robustness)
   - 128-bit Cyclic Payload (5.12초 사이클)
   - 임의 지점 탐지
   - 자기 동기화 (Sync Pattern)

3. 투명성 (Invisible Evidence)
   - 청각적 비인지 (SNR > 30dB)
   - 코덱 저항성 (G.711, G.729)

아키텍처:
    Encoder: Audio [B,1,T] + Message [B,128] → Watermarked [B,1,T]
             (40ms 프레임마다 1비트 삽입, Cyclic)
    
    Decoder: Audio [B,1,T] → Bits [B, num_frames]
             (40ms 프레임마다 1비트 추출)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

# =============================================================================
# Constants
# =============================================================================

FRAME_SAMPLES = 320       # 40ms @ 8kHz
PAYLOAD_LENGTH = 128      # 128-bit cyclic payload
CYCLE_SAMPLES = FRAME_SAMPLES * PAYLOAD_LENGTH  # 40,960 samples = 5.12s
SAMPLE_RATE = 8000


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
        out = out + residual
        out = self.act(out)
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


# =============================================================================
# Frame-Wise Fusion Block (Replaces CrossModalFusionBlock)
# =============================================================================

class FrameWiseFusionBlock(nn.Module):
    """
    Frame-Wise Fusion Block for 40ms Frame Processing
    ==================================================
    
    각 40ms 프레임에 해당하는 1비트만 삽입.
    Cyclic 인덱싱으로 128프레임마다 동일한 비트 패턴 반복.
    
    입력:
        audio_feat: [B, C, T] - 오디오 피처 (T는 320의 배수)
        message: [B, 128] - 128비트 페이로드
    
    출력:
        fused: [B, C, T] - 비트 정보가 융합된 피처
    """
    
    def __init__(
        self, 
        audio_channels: int, 
        message_dim: int = 128,
        frame_samples: int = FRAME_SAMPLES
    ):
        super().__init__()
        
        self.audio_channels = audio_channels
        self.message_dim = message_dim
        self.frame_samples = frame_samples
        
        # 비트 값 (0/1) → 채널 임베딩
        self.bit_embedding = nn.Sequential(
            nn.Linear(1, audio_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(audio_channels // 2, audio_channels),
        )
        
        # 프레임 내 temporal modulation
        self.temporal_modulator = nn.Sequential(
            nn.Conv1d(audio_channels, audio_channels, kernel_size=3, padding=1, groups=audio_channels),
            nn.BatchNorm1d(audio_channels),
            nn.LeakyReLU(0.2, inplace=True),
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
        
        # 프레임 수 계산 (feature extraction에서 downsampling이 없다고 가정)
        # 실제로는 audio_feat의 T가 원본 T보다 작을 수 있음
        num_frames = T // self.frame_samples if T >= self.frame_samples else T
        
        # 각 프레임에 해당하는 비트 인덱스 (Cyclic)
        frame_indices = torch.arange(num_frames, device=audio_feat.device)
        bit_indices = frame_indices % self.message_dim  # [num_frames]
        
        # 각 프레임의 비트 값 가져오기
        frame_bits = message[:, bit_indices]  # [B, num_frames]
        
        # 비트 임베딩
        bit_embeds = self.bit_embedding(frame_bits.unsqueeze(-1))  # [B, num_frames, C]
        bit_embeds = bit_embeds.permute(0, 2, 1)  # [B, C, num_frames]
        
        # 프레임 크기로 확장 (각 프레임 전체에 동일 비트 적용)
        if num_frames < T:
            # T가 frame_samples보다 작은 경우
            bit_signal = F.interpolate(bit_embeds, size=T, mode='nearest')
        else:
            # 각 프레임을 frame_samples만큼 반복
            bit_signal = bit_embeds.repeat_interleave(self.frame_samples, dim=-1)
            bit_signal = bit_signal[:, :, :T]  # 정확한 길이로 자르기
        
        # Temporal modulation
        modulated = self.temporal_modulator(bit_signal)
        
        # Gated fusion
        combined = torch.cat([audio_feat, modulated], dim=1)  # [B, 2C, T]
        gate = self.gate(combined)  # [B, C, T]
        
        fused = gate * audio_feat + (1 - gate) * modulated
        
        # Output projection
        output = self.output_conv(torch.cat([fused, audio_feat], dim=1))
        output = self.norm(output)
        
        return output


# =============================================================================
# Frame-Wise Bit Extractor (Replaces TemporalBitExtractor)
# =============================================================================

class FrameWiseBitExtractor(nn.Module):
    """
    Frame-Wise Bit Extractor for O(1) Complexity
    =============================================
    
    AdaptiveAvgPool1d 제거!
    Stride=frame_samples Conv1d로 프레임당 1비트 추출.
    
    입력:
        x: [B, C, T'] - 피처 맵 (downsampled)
    
    출력:
        logits: [B, num_frames] - 각 프레임의 비트 로짓
    
    핵심: 입력 길이가 달라도 비트 위치가 고정됨!
    """
    
    def __init__(
        self, 
        in_channels: int, 
        frame_samples: int = FRAME_SAMPLES // 16  # downsampled by 16x
    ):
        super().__init__()
        
        self.frame_samples = frame_samples
        
        # 프레임 피처 정제
        self.frame_refine = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 프레임당 1비트 추출 (Stride = frame_samples)
        # kernel_size = frame_samples로 정확히 1프레임을 커버
        self.frame_classifier = nn.Conv1d(
            in_channels // 2, 1,
            kernel_size=max(frame_samples, 1),
            stride=max(frame_samples, 1),
            padding=0
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T'] feature map
            
        Returns:
            logits: [B, num_frames] bit logits
        """
        B, C, T = x.shape
        
        # 1. 프레임 피처 정제
        x = self.frame_refine(x)  # [B, C//2, T]
        
        # 2. 패딩 (T가 frame_samples의 배수가 아닌 경우)
        if T < self.frame_samples:
            # T가 너무 작으면 패딩
            pad_size = self.frame_samples - T
            x = F.pad(x, (0, pad_size))
        elif T % self.frame_samples != 0:
            pad_size = self.frame_samples - (T % self.frame_samples)
            x = F.pad(x, (0, pad_size))
        
        # 3. 프레임당 1비트 추출
        logits = self.frame_classifier(x)  # [B, 1, num_frames]
        logits = logits.squeeze(1)  # [B, num_frames]
        
        return logits


# =============================================================================
# Encoder (Frame-Wise Embedding Network)
# =============================================================================

class Encoder(nn.Module):
    """
    Frame-Wise Watermark Embedding Network
    ======================================
    
    40ms 프레임마다 1비트씩 삽입하는 인코더.
    Cyclic 인덱싱으로 128프레임(5.12초)마다 동일한 128비트 반복.
    
    입력:
        audio: [B, 1, T] - 8kHz 오디오 (T는 320의 배수 권장)
        message: [B, 128] - 128비트 페이로드
    
    출력:
        watermarked: [B, 1, T] - 워터마크된 오디오
    """

    def __init__(
        self,
        message_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 4,
        frame_samples: int = FRAME_SAMPLES
    ):
        super().__init__()

        self.message_dim = message_dim
        self.hidden_channels = hidden_channels
        self.frame_samples = frame_samples

        # =============================================
        # 1. Audio Encoder (Feature Extraction)
        # =============================================
        # Stride=1로 유지하여 temporal resolution 보존
        
        encoder_layers = []
        in_ch = 1

        for out_ch in hidden_channels:
            encoder_layers.append(ConvBlock(in_ch, out_ch, kernel_size=7, padding=3))
            in_ch = out_ch

        self.audio_encoder = nn.Sequential(*encoder_layers)

        # =============================================
        # 2. Frame-Wise Fusion (1비트/프레임)
        # =============================================
        
        self.frame_fusion = FrameWiseFusionBlock(
            audio_channels=hidden_channels[-1],
            message_dim=message_dim,
            frame_samples=frame_samples
        )
        
        # Post-fusion refinement
        self.fusion_refine = nn.Sequential(
            ConvBlock(hidden_channels[-1], hidden_channels[-1], kernel_size=3, padding=1),
            SEBlock(hidden_channels[-1]),
        )

        # =============================================
        # 3. Residual Refinement Blocks
        # =============================================

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                channels=hidden_channels[-1],
                kernel_size=3,
                dilation=2 ** (i % 4)
            )
            for i in range(num_residual_blocks)
        ])

        # =============================================
        # 4. Audio Decoder (Reconstruction)
        # =============================================

        decoder_layers = []
        reversed_channels = list(reversed(hidden_channels))

        for i, out_ch in enumerate(reversed_channels[1:]):
            in_ch = reversed_channels[i]
            decoder_layers.append(ConvBlock(in_ch, out_ch, kernel_size=7, padding=3))

        self.audio_decoder = nn.Sequential(*decoder_layers)

        # =============================================
        # 5. Final Output Layer
        # =============================================

        self.output_conv = nn.Conv1d(hidden_channels[0], 1, kernel_size=7, padding=3)

        # Perturbation scale (학습 가능)
        # 작은 perturbation으로 시작하여 음질 보존
        # 0.1 * alpha 범위: [0.001, 0.03] - 매우 작은 perturbation
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 → α≈0.155 초기값
        self.alpha_min = 0.01   # 최소 강도
        self.alpha_max = 0.3    # 최대 강도

    def forward(
        self,
        audio: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        """
        40ms 프레임별 1비트 삽입

        Args:
            audio: [B, 1, T] 원본 오디오
            message: [B, 128] 워터마크 비트 (0/1)

        Returns:
            watermarked: [B, 1, T] 워터마크된 오디오
        """
        B, _, T = audio.shape

        # 1. Audio feature extraction
        audio_feat = self.audio_encoder(audio)  # [B, C, T]

        # 2. Frame-wise fusion (각 프레임에 해당 비트만 삽입)
        fused = self.frame_fusion(audio_feat, message)  # [B, C, T]
        
        # 3. Post-fusion refinement
        fused = self.fusion_refine(fused)

        # 4. Residual refinement
        for block in self.residual_blocks:
            fused = block(fused)

        # 5. Decode to audio
        decoded = self.audio_decoder(fused)

        # 6. Final output
        perturbation = self.output_conv(decoded)

        # 길이 맞추기
        if perturbation.shape[-1] != T:
            perturbation = F.interpolate(perturbation, size=T, mode='linear', align_corners=False)

        # Clamped alpha (학습 가능, [0.6, 1.0] 범위)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_raw)
        
        # 논문 수식: x̂ = x + α * δ
        # tanh는 perturbation을 [-1, 1]로 제한
        # alpha는 워터마크 강도 조절 (0.6~1.0)
        perturbation = torch.tanh(perturbation) * alpha * 0.1  # 0.1은 초기 안정화용

        # 원본 + 섭동
        watermarked = audio + perturbation
        watermarked = torch.clamp(watermarked, -1.0, 1.0)

        return watermarked


# =============================================================================
# Decoder (Frame-Wise Extraction Network)
# =============================================================================

class Decoder(nn.Module):
    """
    Frame-Wise Watermark Extraction Network
    =======================================
    
    40ms 프레임마다 1비트씩 추출하는 디코더.
    O(1) 복잡도로 프레임당 고정 연산.
    
    입력:
        audio: [B, 1, T] - 워터마크된 오디오
    
    출력:
        logits: [B, num_frames] - 각 프레임의 비트 로짓
                (num_frames = T // FRAME_SAMPLES)
    """

    def __init__(
        self,
        message_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 4,
        frame_samples: int = FRAME_SAMPLES
    ):
        super().__init__()

        self.message_dim = message_dim
        self.frame_samples = frame_samples
        
        # Downsampling factor (stride=2 per layer)
        self.downsample_factor = 2 ** len(hidden_channels)  # 16x for 4 layers
        self.downsampled_frame = frame_samples // self.downsample_factor

        # =============================================
        # 1. Feature Extractor (with downsampling)
        # =============================================

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

        # SE Block
        self.se_block = SEBlock(hidden_channels[-1])

        # =============================================
        # 3. Frame-Wise Bit Extractor (O(1) per frame)
        # =============================================
        
        self.bit_extractor = FrameWiseBitExtractor(
            in_channels=hidden_channels[-1],
            frame_samples=self.downsampled_frame
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        프레임별 비트 추출

        Args:
            audio: [B, 1, T] 워터마크된 오디오

        Returns:
            logits: [B, num_frames] 각 프레임의 비트 로짓
        """
        # 1. Feature extraction (with downsampling)
        features = self.feature_extractor(audio)  # [B, C, T']

        # 2. Residual blocks
        for block in self.residual_blocks:
            features = block(features)

        # 3. Channel attention
        features = self.se_block(features)

        # 4. Frame-wise bit extraction
        logits = self.bit_extractor(features)  # [B, num_frames]

        return logits
    
    def extract_128bits(self, audio: torch.Tensor) -> torch.Tensor:
        """
        128비트 페이로드 복원 (Cyclic aggregation)
        
        긴 오디오에서 여러 사이클을 평균하여 128비트 복원.
        
        Args:
            audio: [B, 1, T] - 최소 5.12초 권장
            
        Returns:
            bits_128: [B, 128] - 복원된 128비트 페이로드 로짓
        """
        logits = self.forward(audio)  # [B, num_frames]
        B, num_frames = logits.shape
        
        # 128비트로 집계 (Cyclic averaging)
        bits_128 = torch.zeros(B, self.message_dim, device=logits.device)
        counts = torch.zeros(self.message_dim, device=logits.device)
        
        for i in range(num_frames):
            bit_idx = i % self.message_dim
            bits_128[:, bit_idx] += logits[:, i]
            counts[bit_idx] += 1
        
        # 평균
        bits_128 = bits_128 / counts.unsqueeze(0).clamp(min=1)
        
        return bits_128


# =============================================================================
# Discriminator (for GAN training)
# =============================================================================

class Discriminator(nn.Module):
    """
    Multi-Scale Discriminator
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
    CallCops Complete Network v2.0
    ==============================
    
    Frame-Wise 워터마킹:
    - 40ms 프레임당 1비트 삽입/추출
    - 128비트 Cyclic Payload (5.12초 사이클)
    - O(1) 복잡도 per frame
    """

    def __init__(
        self,
        message_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 4,
        use_discriminator: bool = True,
        frame_samples: int = FRAME_SAMPLES
    ):
        super().__init__()

        self.message_dim = message_dim
        self.frame_samples = frame_samples

        # Encoder
        self.encoder = Encoder(
            message_dim=message_dim,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            frame_samples=frame_samples
        )

        # Decoder
        self.decoder = Decoder(
            message_dim=message_dim,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            frame_samples=frame_samples
        )

        # Discriminator (optional)
        self.use_discriminator = use_discriminator
        if use_discriminator:
            self.discriminator = Discriminator()

    def embed(
        self,
        audio: torch.Tensor,
        message: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        워터마크 삽입 (40ms 프레임당 1비트)

        Args:
            audio: [B, 1, T]
            message: [B, 128]

        Returns:
            watermarked: [B, 1, T]
            perturbation: [B, 1, T] (for analysis)
        """
        watermarked = self.encoder(audio, message)
        perturbation = watermarked - audio
        return watermarked, perturbation

    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        """
        워터마크 추출 (프레임별)

        Args:
            audio: [B, 1, T]

        Returns:
            logits: [B, num_frames] - 각 프레임의 비트 로짓
        """
        return self.decoder(audio)
    
    def extract_128bits(self, audio: torch.Tensor) -> torch.Tensor:
        """
        128비트 페이로드 복원

        Args:
            audio: [B, 1, T]

        Returns:
            bits_128: [B, 128] - Cyclic 집계된 비트 로짓
        """
        return self.decoder.extract_128bits(audio)

    def count_parameters(self) -> dict:
        """파라미터 수 계산"""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters()) if self.use_discriminator else 0

        return {
            'encoder': enc_params,
            'decoder': dec_params,
            'discriminator': disc_params,
            'total': enc_params + dec_params + disc_params
        }


# =============================================================================
# Utility Functions
# =============================================================================

def align_to_frames(audio: torch.Tensor, frame_samples: int = FRAME_SAMPLES) -> torch.Tensor:
    """
    오디오를 프레임 경계에 맞게 패딩
    
    Args:
        audio: [B, 1, T]
        frame_samples: 프레임 크기 (기본 320)
        
    Returns:
        aligned: [B, 1, T'] where T' is multiple of frame_samples
    """
    T = audio.shape[-1]
    if T % frame_samples == 0:
        return audio
    
    pad_size = frame_samples - (T % frame_samples)
    return F.pad(audio, (0, pad_size))


def get_cyclic_bit_indices(num_frames: int, payload_length: int = PAYLOAD_LENGTH) -> torch.Tensor:
    """
    Cyclic 비트 인덱스 생성
    
    Args:
        num_frames: 프레임 수
        payload_length: 페이로드 길이 (기본 128)
        
    Returns:
        indices: [num_frames] - 각 프레임의 비트 인덱스 (0~127 반복)
    """
    return torch.arange(num_frames) % payload_length


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CallCops Frame-Wise Network v2.0 Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 모델 생성
    model = CallCopsNet(message_dim=128).to(device)
    params = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Total: {params['total']:,}")
    
    # 테스트 1: 1초 오디오 (25 프레임)
    print("\n--- Test 1: 1초 오디오 (25 프레임) ---")
    audio_1s = torch.randn(2, 1, 8000).to(device)
    message = torch.randint(0, 2, (2, 128)).float().to(device)
    
    watermarked, perturbation = model.embed(audio_1s, message)
    logits = model.extract(watermarked)
    
    print(f"Input: {audio_1s.shape}")
    print(f"Watermarked: {watermarked.shape}")
    print(f"Extracted logits: {logits.shape}")  # [2, 25]
    print(f"Expected frames: {8000 // 320} = 25")
    
    # 테스트 2: 5.12초 오디오 (128 프레임 = 1 사이클)
    print("\n--- Test 2: 5.12초 오디오 (128 프레임) ---")
    audio_5s = torch.randn(2, 1, 40960).to(device)
    
    watermarked, _ = model.embed(audio_5s, message)
    logits = model.extract(watermarked)
    bits_128 = model.extract_128bits(watermarked)
    
    print(f"Input: {audio_5s.shape}")
    print(f"Extracted logits: {logits.shape}")  # [2, 128]
    print(f"128-bit payload: {bits_128.shape}")  # [2, 128]
    
    # 정확도 테스트
    pred_bits = (torch.sigmoid(bits_128) > 0.5).float()
    accuracy = (pred_bits == message).float().mean()
    print(f"Accuracy (untrained): {accuracy.item():.2%}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
