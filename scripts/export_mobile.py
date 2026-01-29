"""
CallCops: Mobile Export Script
===================================

Android Lite Interpreter 최적화를 위한 모델 변환.

지원 포맷:
1. TorchScript (JIT Trace/Script)
2. TorchScript Lite (.ptl)
3. Dynamic Quantization (INT8)
4. ONNX Export

안드로이드 요구사항:
- 모델 크기 < 10MB
- 추론 시간 < 50ms (40ms 프레임 기준)
- 메모리 사용량 < 50MB
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.quantization as quant

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from models import RTAWNet, RTAWEncoder, RTAWDecoder


class EncoderWrapper(nn.Module):
    """
    Encoder 래퍼 (모바일 최적화)
    ============================

    워터마크 삽입만 수행하는 경량 모듈.
    Android에서 실시간 워터마킹에 사용.
    """

    def __init__(self, encoder: RTAWEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        audio: torch.Tensor,
        bits: torch.Tensor
    ) -> torch.Tensor:
        """
        워터마크 삽입

        Args:
            audio: [1, 1, 320] 40ms 오디오 프레임
            bits: [1, 128] 워터마크 비트

        Returns:
            [1, 1, 320] 워터마크된 오디오
        """
        watermarked, _ = self.encoder(audio, bits)
        return watermarked


class DecoderWrapper(nn.Module):
    """
    Decoder 래퍼 (모바일 최적화)
    ============================

    워터마크 추출만 수행하는 경량 모듈.
    Android에서 인증에 사용.
    """

    def __init__(self, decoder: RTAWDecoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        워터마크 추출

        Args:
            audio: [1, 1, 320] 워터마크된 오디오

        Returns:
            bit_probs: [1, 128] 비트 확률
            detection: [1, 1] 탐지 신뢰도
        """
        return self.decoder(audio)


class MobileCallCops(nn.Module):
    """
    모바일 통합 모듈
    ================

    Encoder + Decoder를 하나로 통합.
    Android에서 단일 모델로 로드 가능.
    """

    def __init__(self, model: RTAWNet):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder

    def embed(
        self,
        audio: torch.Tensor,
        bits: torch.Tensor
    ) -> torch.Tensor:
        """워터마크 삽입"""
        watermarked, _ = self.encoder(audio, bits)
        return watermarked

    def extract(
        self,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """워터마크 추출"""
        return self.decoder(audio)

    def forward(
        self,
        audio: torch.Tensor,
        bits: torch.Tensor,
        mode: str = "embed"
    ) -> torch.Tensor:
        """
        통합 forward (JIT 호환)

        mode: "embed" 또는 "extract"
        """
        if mode == "embed":
            return self.embed(audio, bits)
        else:
            bit_probs, detection = self.extract(audio)
            # 단일 텐서로 결합
            return torch.cat([bit_probs, detection], dim=-1)


def export_torchscript(
    model: nn.Module,
    output_path: Path,
    example_inputs: tuple,
    use_script: bool = False
):
    """
    TorchScript 변환

    Args:
        model: PyTorch 모델
        output_path: 저장 경로
        example_inputs: 예제 입력 (trace용)
        use_script: True면 torch.jit.script, False면 trace
    """
    model.eval()

    if use_script:
        # Script mode (동적 제어 흐름 지원)
        scripted = torch.jit.script(model)
    else:
        # Trace mode (더 최적화됨)
        scripted = torch.jit.trace(model, example_inputs)

    # 최적화
    scripted = torch.jit.optimize_for_inference(scripted)

    # 저장
    scripted.save(str(output_path))

    print(f"TorchScript model saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def export_lite(
    torchscript_path: Path,
    output_path: Path
):
    """
    TorchScript Lite 변환 (.ptl)

    Android Lite Interpreter용 경량 포맷.
    """
    # Lite 모델 생성
    scripted = torch.jit.load(str(torchscript_path))

    # Mobile 최적화
    optimized = torch.utils.mobile_optimizer.optimize_for_mobile(
        scripted,
        preserved_methods=['embed', 'extract'] if hasattr(scripted, 'embed') else None
    )

    # Lite 포맷으로 저장
    optimized._save_for_lite_interpreter(str(output_path))

    print(f"TorchScript Lite model saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def apply_dynamic_quantization(
    model: nn.Module
) -> nn.Module:
    """
    동적 양자화 적용 (INT8)

    Conv1d, Linear 레이어를 INT8로 양자화.
    모델 크기 감소 및 CPU 추론 가속.
    """
    # 양자화 대상 레이어
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv1d},
        dtype=torch.qint8
    )

    return quantized


def export_onnx(
    model: nn.Module,
    output_path: Path,
    example_inputs: tuple,
    input_names: list,
    output_names: list,
    dynamic_axes: dict = None
):
    """
    ONNX 포맷으로 변환

    TensorRT, NNAPI 등 다양한 런타임 지원.
    """
    model.eval()

    torch.onnx.export(
        model,
        example_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True
    )

    print(f"ONNX model saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def benchmark_model(
    model: nn.Module,
    device: torch.device,
    num_iterations: int = 100,
    frame_samples: int = 320
):
    """
    모델 벤치마크

    추론 시간 및 메모리 사용량 측정.
    """
    import time

    model.eval()
    model.to(device)

    # 예제 입력
    audio = torch.randn(1, 1, frame_samples).to(device)
    bits = torch.randint(0, 2, (1, 128)).float().to(device)

    # Warmup
    for _ in range(10):
        if hasattr(model, 'embed'):
            model.embed(audio, bits)
        else:
            model(audio, bits)

    # 벤치마크
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()

        if hasattr(model, 'embed'):
            model.embed(audio, bits)
        else:
            model(audio, bits)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)

    import numpy as np

    print(f"\nBenchmark Results ({num_iterations} iterations):")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Std: {np.std(times):.2f} ms")
    print(f"  Min: {np.min(times):.2f} ms")
    print(f"  Max: {np.max(times):.2f} ms")
    print(f"  P95: {np.percentile(times, 95):.2f} ms")

    # 메모리 사용량 (CUDA)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

        if hasattr(model, 'embed'):
            model.embed(audio, bits)
        else:
            model(audio, bits)

        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  Peak Memory: {peak_memory:.2f} MB")


def count_parameters(model: nn.Module) -> dict:
    """모델 파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'total_mb': total * 4 / 1024 / 1024  # float32 기준
    }


def main():
    parser = argparse.ArgumentParser(description="CallCops Mobile Export")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='exported',
                        help='Output directory for exported models')
    parser.add_argument('--formats', type=str, nargs='+',
                        default=['torchscript', 'lite', 'quantized', 'onnx'],
                        help='Export formats')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark after export')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for benchmarking')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 체크포인트 로드
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})

    # 모델 초기화
    model = RTAWNet(
        bits_dim=config.get('watermark', {}).get('payload_length', 128)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 모델 정보
    params = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Estimated size (float32): {params['total_mb']:.2f} MB")

    # 예제 입력 (40ms @ 8kHz)
    frame_samples = 320
    example_audio = torch.randn(1, 1, frame_samples)
    example_bits = torch.randint(0, 2, (1, 128)).float()

    # 모바일 래퍼 생성
    mobile_model = MobileCallCops(model)
    encoder_model = EncoderWrapper(model.encoder)
    decoder_model = DecoderWrapper(model.decoder)

    print(f"\nExporting to: {output_dir}")

    # ==================
    # TorchScript Export
    # ==================
    if 'torchscript' in args.formats:
        print("\n[1] TorchScript Export")

        # Encoder
        ts_encoder_path = output_dir / "encoder.pt"
        export_torchscript(
            encoder_model,
            ts_encoder_path,
            (example_audio, example_bits)
        )

        # Decoder
        ts_decoder_path = output_dir / "decoder.pt"
        export_torchscript(
            decoder_model,
            ts_decoder_path,
            (example_audio,)
        )

    # ==================
    # TorchScript Lite Export
    # ==================
    if 'lite' in args.formats:
        print("\n[2] TorchScript Lite Export")

        # Encoder Lite
        if (output_dir / "encoder.pt").exists():
            export_lite(
                output_dir / "encoder.pt",
                output_dir / "encoder.ptl"
            )

        # Decoder Lite
        if (output_dir / "decoder.pt").exists():
            export_lite(
                output_dir / "decoder.pt",
                output_dir / "decoder.ptl"
            )

    # ==================
    # Quantized Export
    # ==================
    if 'quantized' in args.formats:
        print("\n[3] Quantized Model Export")

        # Encoder 양자화
        quantized_encoder = apply_dynamic_quantization(encoder_model)
        torch.jit.save(
            torch.jit.trace(quantized_encoder, (example_audio, example_bits)),
            str(output_dir / "encoder_quantized.pt")
        )
        print(f"Quantized encoder saved: {output_dir / 'encoder_quantized.pt'}")

        # Decoder 양자화
        quantized_decoder = apply_dynamic_quantization(decoder_model)
        torch.jit.save(
            torch.jit.trace(quantized_decoder, (example_audio,)),
            str(output_dir / "decoder_quantized.pt")
        )
        print(f"Quantized decoder saved: {output_dir / 'decoder_quantized.pt'}")

    # ==================
    # ONNX Export
    # ==================
    if 'onnx' in args.formats:
        print("\n[4] ONNX Export")

        # Encoder ONNX
        export_onnx(
            encoder_model,
            output_dir / "encoder.onnx",
            (example_audio, example_bits),
            input_names=['audio', 'bits'],
            output_names=['watermarked'],
            dynamic_axes={
                'audio': {0: 'batch', 2: 'time'},
                'watermarked': {0: 'batch', 2: 'time'}
            }
        )

        # Decoder ONNX
        export_onnx(
            decoder_model,
            output_dir / "decoder.onnx",
            (example_audio,),
            input_names=['audio'],
            output_names=['bit_probs', 'detection'],
            dynamic_axes={
                'audio': {0: 'batch', 2: 'time'}
            }
        )

    # ==================
    # Benchmark
    # ==================
    if args.benchmark:
        print("\n[5] Benchmarking")

        device = torch.device(args.device)

        print("\nOriginal Model:")
        benchmark_model(encoder_model, device)

        if 'quantized' in args.formats:
            print("\nQuantized Model:")
            quantized_encoder = apply_dynamic_quantization(encoder_model)
            benchmark_model(quantized_encoder, torch.device('cpu'))

    # ==================
    # Summary
    # ==================
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)

    for f in output_dir.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {size_mb:.2f} MB")

    print("\nAndroid Integration:")
    print("  1. Copy .ptl files to assets/")
    print("  2. Load with LiteModuleLoader.load()")
    print("  3. Run inference with module.forward()")

    print("\nExample Android Code:")
    print("""
    // Load model
    Module encoder = LiteModuleLoader.load(
        assetFilePath(context, "encoder.ptl")
    );

    // Prepare input
    float[] audioData = new float[320];  // 40ms @ 8kHz
    float[] bitsData = new float[128];   // Watermark bits

    Tensor audioTensor = Tensor.fromBlob(audioData, new long[]{1, 1, 320});
    Tensor bitsTensor = Tensor.fromBlob(bitsData, new long[]{1, 128});

    // Run inference
    Tensor output = encoder.forward(IValue.from(audioTensor), IValue.from(bitsTensor))
                           .toTensor();
    """)


if __name__ == "__main__":
    main()
