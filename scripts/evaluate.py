"""
CallCops: Evaluation Script
================================

PESQ, BER 등 품질 메트릭 평가.

평가 항목:
1. PESQ (Perceptual Evaluation of Speech Quality)
2. BER (Bit Error Rate)
3. Detection Accuracy
4. Codec Robustness (G.711, G.729)
5. 지연 시간 측정
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import yaml

# PESQ 라이브러리 (선택적)
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("Warning: pesq not installed. PESQ evaluation will be skipped.")
    print("Install with: pip install pesq")

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from models import RTAWNet, DifferentiableCodecSimulator
from scripts.dataset import create_dataloader, PayloadGenerator


class Evaluator:
    """
    CallCops 평가기
    ====================

    품질 목표:
    - PESQ >= 4.0
    - BER < 5% (G.729 압축 후)
    - 지연 < 200ms
    """

    def __init__(
        self,
        model: RTAWNet,
        device: torch.device,
        sample_rate: int = 8000
    ):
        self.model = model
        self.device = device
        self.sample_rate = sample_rate

        # 코덱 시뮬레이터
        self.codec_sim = DifferentiableCodecSimulator(
            codec_types=['g711_alaw', 'g711_ulaw', 'g729', 'none']
        ).to(device)
        self.codec_sim.eval()

        # 페이로드 생성기
        self.payload_gen = PayloadGenerator()

    @torch.no_grad()
    def evaluate_ber(
        self,
        dataloader,
        codec_type: str = 'none'
    ) -> Dict[str, float]:
        """
        BER (Bit Error Rate) 평가

        Args:
            dataloader: 테스트 DataLoader
            codec_type: 적용할 코덱

        Returns:
            BER 및 관련 메트릭
        """
        self.model.eval()

        total_bits = 0
        error_bits = 0
        correct_detections = 0
        total_samples = 0

        for batch in tqdm(dataloader, desc=f"BER Eval ({codec_type})"):
            audio = batch['audio'].to(self.device)
            bits = batch['bits'].to(self.device)

            # 워터마크 삽입
            watermarked, _ = self.model.embed(audio, bits)

            # 코덱 적용
            if codec_type != 'none':
                watermarked, _ = self.codec_sim(watermarked, codec_type=codec_type)

            # 워터마크 추출
            bit_probs, detection = self.model.extract(watermarked)

            # 비트 오류 계산
            pred_bits = (bit_probs > 0.5).float()
            errors = (pred_bits != bits).sum().item()

            error_bits += errors
            total_bits += bits.numel()

            # 탐지 정확도
            detected = (detection > 0.5).float()
            correct_detections += detected.sum().item()
            total_samples += detection.numel()

        ber = error_bits / total_bits
        detection_acc = correct_detections / total_samples

        return {
            'ber': ber,
            'accuracy': 1 - ber,
            'detection_accuracy': detection_acc,
            'total_bits': total_bits,
            'error_bits': error_bits
        }

    @torch.no_grad()
    def evaluate_pesq(
        self,
        dataloader,
        max_samples: int = 100
    ) -> Dict[str, float]:
        """
        PESQ 평가

        Args:
            dataloader: 테스트 DataLoader
            max_samples: 최대 평가 샘플 수

        Returns:
            PESQ 점수
        """
        if not PESQ_AVAILABLE:
            return {'pesq': -1, 'error': 'pesq not installed'}

        self.model.eval()

        pesq_scores = []
        num_samples = 0

        for batch in tqdm(dataloader, desc="PESQ Eval"):
            if num_samples >= max_samples:
                break

            audio = batch['audio'].to(self.device)
            bits = batch['bits'].to(self.device)

            # 워터마크 삽입
            watermarked, _ = self.model.embed(audio, bits)

            # CPU로 이동 및 numpy 변환
            original_np = audio.squeeze().cpu().numpy()
            watermarked_np = watermarked.squeeze().cpu().numpy()

            # 배치의 각 샘플에 대해 PESQ 계산
            for i in range(original_np.shape[0] if original_np.ndim > 1 else 1):
                try:
                    if original_np.ndim > 1:
                        orig = original_np[i]
                        wm = watermarked_np[i]
                    else:
                        orig = original_np
                        wm = watermarked_np

                    # PESQ 계산 (8kHz narrowband)
                    score = pesq(self.sample_rate, orig, wm, 'nb')
                    pesq_scores.append(score)
                    num_samples += 1

                    if num_samples >= max_samples:
                        break
                except Exception as e:
                    print(f"PESQ error: {e}")
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
            return {'pesq': -1, 'error': 'No valid samples'}

    def measure_latency(
        self,
        num_iterations: int = 100,
        frame_samples: int = 320  # 40ms @ 8kHz
    ) -> Dict[str, float]:
        """
        지연 시간 측정

        Args:
            num_iterations: 측정 반복 횟수
            frame_samples: 프레임 샘플 수

        Returns:
            지연 시간 통계
        """
        self.model.eval()

        # 테스트 입력
        audio = torch.randn(1, 1, frame_samples).to(self.device)
        bits = torch.randint(0, 2, (1, 128)).float().to(self.device)

        # Warmup
        for _ in range(10):
            self.model.embed(audio, bits)
            self.model.extract(audio)

        # 동기화 (CUDA)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Embed 지연 측정
        embed_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.model.embed(audio, bits)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            embed_times.append((time.perf_counter() - start) * 1000)  # ms

        # Extract 지연 측정
        extract_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.model.extract(audio)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            extract_times.append((time.perf_counter() - start) * 1000)  # ms

        return {
            'embed_mean_ms': np.mean(embed_times),
            'embed_std_ms': np.std(embed_times),
            'embed_p95_ms': np.percentile(embed_times, 95),
            'extract_mean_ms': np.mean(extract_times),
            'extract_std_ms': np.std(extract_times),
            'extract_p95_ms': np.percentile(extract_times, 95),
            'total_mean_ms': np.mean(embed_times) + np.mean(extract_times),
            'frame_ms': frame_samples / self.sample_rate * 1000
        }

    def full_evaluation(
        self,
        dataloader,
        output_path: Optional[Path] = None
    ) -> Dict[str, Dict]:
        """
        전체 평가 수행

        Args:
            dataloader: 테스트 DataLoader
            output_path: 결과 저장 경로

        Returns:
            전체 평가 결과
        """
        results = {}

        print("=" * 60)
        print("CallCops Full Evaluation")
        print("=" * 60)

        # 1. BER (코덱 없음)
        print("\n[1/5] BER (No Codec)")
        results['ber_none'] = self.evaluate_ber(dataloader, codec_type='none')
        print(f"  BER: {results['ber_none']['ber']:.4f}")

        # 2. BER (G.711 A-law)
        print("\n[2/5] BER (G.711 A-law)")
        results['ber_g711_alaw'] = self.evaluate_ber(dataloader, codec_type='g711_alaw')
        print(f"  BER: {results['ber_g711_alaw']['ber']:.4f}")

        # 3. BER (G.711 μ-law)
        print("\n[3/5] BER (G.711 μ-law)")
        results['ber_g711_ulaw'] = self.evaluate_ber(dataloader, codec_type='g711_ulaw')
        print(f"  BER: {results['ber_g711_ulaw']['ber']:.4f}")

        # 4. BER (G.729)
        print("\n[4/5] BER (G.729)")
        results['ber_g729'] = self.evaluate_ber(dataloader, codec_type='g729')
        print(f"  BER: {results['ber_g729']['ber']:.4f}")

        # 5. PESQ
        print("\n[5/5] PESQ")
        results['pesq'] = self.evaluate_pesq(dataloader)
        if 'pesq_mean' in results['pesq']:
            print(f"  PESQ: {results['pesq']['pesq_mean']:.2f} ± {results['pesq']['pesq_std']:.2f}")

        # 지연 시간
        print("\n[Latency Measurement]")
        results['latency'] = self.measure_latency()
        print(f"  Embed: {results['latency']['embed_mean_ms']:.2f} ms")
        print(f"  Extract: {results['latency']['extract_mean_ms']:.2f} ms")
        print(f"  Total: {results['latency']['total_mean_ms']:.2f} ms")

        # 품질 목표 체크
        print("\n" + "=" * 60)
        print("Quality Target Check")
        print("=" * 60)

        pesq_target = 4.0
        ber_target = 0.05
        latency_target = 200

        pesq_pass = results['pesq'].get('pesq_mean', 0) >= pesq_target
        ber_pass = results['ber_g729']['ber'] < ber_target
        latency_pass = results['latency']['total_mean_ms'] < latency_target

        print(f"  PESQ >= {pesq_target}: {'✓ PASS' if pesq_pass else '✗ FAIL'}")
        print(f"  BER (G.729) < {ber_target}: {'✓ PASS' if ber_pass else '✗ FAIL'}")
        print(f"  Latency < {latency_target}ms: {'✓ PASS' if latency_pass else '✗ FAIL'}")

        results['quality_check'] = {
            'pesq_pass': pesq_pass,
            'ber_pass': ber_pass,
            'latency_pass': latency_pass,
            'all_pass': pesq_pass and ber_pass and latency_pass
        }

        # 결과 저장
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # YAML로 저장
            def convert_to_serializable(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                return obj

            with open(output_path, 'w') as f:
                yaml.dump(convert_to_serializable(results), f, default_flow_style=False)

            print(f"\nResults saved to: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="CallCops Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output', type=str, default='results/evaluation.yaml',
                        help='Output path for results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')

    args = parser.parse_args()

    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})

    # 모델 초기화 및 로드
    model = RTAWNet(
        bits_dim=config.get('watermark', {}).get('payload_length', 128)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # DataLoader 생성
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        mode='test',
        sample_rate=config.get('audio', {}).get('sample_rate', 8000),
        frame_ms=config.get('audio', {}).get('frame_ms', 40)
    )

    # 평가 수행
    evaluator = Evaluator(model, device)
    results = evaluator.full_evaluation(dataloader, output_path=args.output)


if __name__ == "__main__":
    main()
