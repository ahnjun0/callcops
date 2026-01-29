"""
CallCops: Attention Modules
================================

마스킹 임계치(Masking Threshold) 기반 Attention 구현.

- 사람의 청각 마스킹 효과를 활용
- 고에너지 구간(마스킹 임계치가 높은 구간)에 워터마크 집중
- 이를 통해 인지적 투명성(Perceptual Transparency) 달성

한국어 음성 특성:
- 한국어 자음/모음 구조의 에너지 분포 고려
- 초성, 중성, 종성의 마스킹 특성 차이 활용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MaskingAwareAttention(nn.Module):
    """
    Masking-Aware Multi-Head Attention
    ==================================

    청각 마스킹 효과를 모방한 Attention 메커니즘.

    원리:
    1. 음성 신호의 에너지 분포 분석
    2. 고에너지 구간 = 마스킹 임계치 높음 = 변화 감지 어려움
    3. 이 구간에 워터마크 비트를 집중적으로 삽입

    수식:
        Masking_Score(t) = σ(Energy(t) / mean(Energy))
        Attention(Q, K, V) = softmax(QK^T / √d + Masking_Bias) V

    한국어 최적화:
        - 8kHz 샘플링에서 자음 마찰음(ㅅ, ㅆ, ㅎ 등)의
          고주파 에너지 활용
        - 모음의 포만트(Formant) 구간 활용
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        energy_window: int = 5,  # 에너지 계산 윈도우
        masking_strength: float = 1.0  # 마스킹 영향력
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.energy_window = energy_window
        self.masking_strength = masking_strength

        # Q, K, V projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Masking threshold estimator
        # 신호 특성에서 마스킹 임계치를 추정
        self.masking_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

        # Learnable masking bias
        self.masking_bias_scale = nn.Parameter(torch.ones(1))

    def compute_energy_mask(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        시간 축의 에너지 기반 마스킹 맵 계산

        Args:
            x: [B, T, C] 특징 텐서

        Returns:
            energy_mask: [B, T, 1] 정규화된 에너지 맵
        """
        # 채널 방향 에너지 계산
        energy = torch.sum(x ** 2, dim=-1, keepdim=True)  # [B, T, 1]

        # 이동 평균으로 smoothing
        if self.energy_window > 1:
            kernel = torch.ones(1, 1, self.energy_window, device=x.device) / self.energy_window
            energy_smoothed = F.conv1d(
                energy.transpose(1, 2),
                kernel,
                padding=self.energy_window // 2
            ).transpose(1, 2)
        else:
            energy_smoothed = energy

        # 정규화 (0-1 범위)
        energy_min = energy_smoothed.min(dim=1, keepdim=True)[0]
        energy_max = energy_smoothed.max(dim=1, keepdim=True)[0]
        energy_norm = (energy_smoothed - energy_min) / (energy_max - energy_min + 1e-8)

        return energy_norm

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masking-aware attention forward pass

        Args:
            x: [B, T, C] 입력 특징
            mask: [B, T] optional attention mask

        Returns:
            output: [B, T, C] attended 특징
            attention_weights: [B, T] 마스킹 기반 가중치
        """
        B, T, C = x.shape

        # 1. Q, K, V projection
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, T, head_dim]

        # 2. Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        # 3. 에너지 기반 마스킹 bias
        energy_mask = self.compute_energy_mask(x)  # [B, T, 1]
        masking_bias = self.masking_estimator(x)   # [B, T, 1] learned mask

        # 결합된 마스킹 점수
        combined_mask = (energy_mask + masking_bias) / 2  # [B, T, 1]

        # Attention score에 마스킹 bias 추가
        # 고에너지/고마스킹 구간에 더 집중하도록
        mask_bias = combined_mask.transpose(1, 2).unsqueeze(1)  # [B, 1, 1, T]
        mask_bias = mask_bias * self.masking_bias_scale * self.masking_strength

        attn_scores = attn_scores + mask_bias

        # 4. Causal masking (optional, for streaming)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )

        # 5. Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. Apply attention
        output = torch.matmul(attn_weights, V)  # [B, H, T, head_dim]
        output = output.transpose(1, 2).contiguous().view(B, T, C)

        # 7. Output projection
        output = self.out_proj(output)

        # 반환할 attention weights: 시간축 마스킹 점수
        return output, combined_mask.squeeze(-1)


class TemporalAttention(nn.Module):
    """
    Temporal Attention for Sequence Aggregation
    ============================================

    시간 축을 따라 특징을 집약하는 Attention.
    Decoder에서 전체 시퀀스를 단일 벡터로 요약할 때 사용.

    수식:
        score(t) = v^T tanh(W_h * h_t + b)
        α = softmax(scores)
        output = Σ α_t * h_t
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        hidden_dim = hidden_dim or embed_dim

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Temporal aggregation via attention

        Args:
            x: [B, T, C] 시퀀스 특징
            mask: [B, T] optional mask

        Returns:
            aggregated: [B, C] 집약된 특징 벡터
        """
        # Attention scores
        scores = self.attention(x).squeeze(-1)  # [B, T]

        # Masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        weights = F.softmax(scores, dim=-1)  # [B, T]
        weights = self.dropout(weights)

        # Weighted sum
        aggregated = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, C]

        return aggregated


class SpectralMaskingAttention(nn.Module):
    """
    Spectral Masking Attention (선택적 모듈)
    ========================================

    주파수 도메인에서의 마스킹 효과를 활용.
    STFT를 통해 주파수별 에너지를 분석하고,
    인접 주파수 성분의 마스킹 효과를 모델링.

    한국어 음성 특성 활용:
    - 기본 주파수 (F0): 남성 ~120Hz, 여성 ~220Hz
    - 포만트: F1(300-800Hz), F2(800-2500Hz)
    - 8kHz 샘플링에서 Nyquist = 4kHz

    Note: 이 모듈은 실시간성이 떨어지므로 선택적 사용
    """

    def __init__(
        self,
        n_fft: int = 256,  # 8kHz에서 32ms 윈도우
        hop_length: int = 64,  # 8ms hop
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq = n_fft // 2 + 1

        # 주파수 임베딩
        self.freq_embedding = nn.Embedding(self.n_freq, embed_dim)

        # Cross-frequency attention
        self.cross_freq_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 마스킹 임계치 추정
        self.masking_threshold = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spectral masking attention

        Args:
            x: [B, C, T] 시간 도메인 특징

        Returns:
            masked_features: [B, C, T] 마스킹 적용된 특징
            spectral_mask: [B, T, n_freq] 주파수별 마스킹 맵
        """
        B, C, T = x.shape

        # 간단한 "pseudo-spectral" 분석
        # 실제로는 learnable 주파수 분해를 사용

        # 시간축을 주파수처럼 취급하는 트릭 (경량화)
        x_reshaped = x.transpose(1, 2)  # [B, T, C]

        # 주파수 위치 임베딩 추가
        freq_pos = torch.arange(min(C, self.n_freq), device=x.device)
        freq_embed = self.freq_embedding(freq_pos)  # [n_freq, embed_dim]

        # 특징과 결합
        if C > self.n_freq:
            x_freq = x_reshaped[:, :, :self.n_freq]
        else:
            x_freq = F.pad(x_reshaped, (0, self.n_freq - C))

        # Cross-frequency attention
        x_attended, attn_weights = self.cross_freq_attn(
            x_freq, x_freq, x_freq
        )

        # 마스킹 임계치 추정
        spectral_mask = self.masking_threshold(x_attended)  # [B, T, 1]

        # 원래 특징에 마스킹 적용
        if C > self.n_freq:
            mask_expanded = F.pad(spectral_mask.squeeze(-1), (0, C - self.n_freq))
        else:
            mask_expanded = spectral_mask.squeeze(-1)[:, :, :C]

        masked_features = x * mask_expanded.transpose(1, 2)

        return masked_features, spectral_mask.squeeze(-1)


if __name__ == "__main__":
    # 테스트 코드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MaskingAwareAttention 테스트
    print("=" * 60)
    print("Masking-Aware Attention Test")
    print("=" * 60)

    attn = MaskingAwareAttention(embed_dim=256, num_heads=4).to(device)

    # [B, T, C] 입력
    x = torch.randn(4, 100, 256).to(device)
    output, weights = attn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min():.3f}, {weights.max():.3f}]")

    # TemporalAttention 테스트
    print("\n" + "=" * 60)
    print("Temporal Attention Test")
    print("=" * 60)

    temp_attn = TemporalAttention(embed_dim=256).to(device)
    aggregated = temp_attn(x)

    print(f"Input shape: {x.shape}")
    print(f"Aggregated shape: {aggregated.shape}")
