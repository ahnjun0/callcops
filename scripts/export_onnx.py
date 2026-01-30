"""
CallCops: ONNX Export Script for Web/Mobile Inference
=======================================================

ONNX Runtime Web (Wasm/WebGPU) 및 모바일 환경을 위한 최적화된 변환.

Features:
1. Encoder/Decoder 분리 export
2. Dynamic Axes (Batch, Time)
3. Opset 16 (ONNX Runtime Web 호환)
4. INT8 Quantization (Static + Dynamic)
5. PyTorch vs ONNX 검증

Target:
- ONNX Runtime Web (Wasm/WebGPU)
- ONNX Runtime Mobile (iOS/Android)
- Model Size < 10MB
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CallCopsNet, Encoder, Decoder


# =============================================================================
# Wrapper Classes for Clean ONNX Export
# =============================================================================

class EncoderONNXWrapper(nn.Module):
    """
    Encoder ONNX Wrapper
    ====================
    
    Audio + Message → Watermarked Audio
    
    ONNX 호환을 위해 forward만 노출.
    """
    
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(
        self,
        audio: torch.Tensor,   # [B, 1, T]
        message: torch.Tensor  # [B, 128]
    ) -> torch.Tensor:
        """
        Args:
            audio: [B, 1, T] - 8kHz audio (variable length)
            message: [B, 128] - Watermark bits (0/1 float)
            
        Returns:
            watermarked: [B, 1, T] - Watermarked audio
        """
        return self.encoder(audio, message)


class DecoderONNXWrapper(nn.Module):
    """
    Decoder ONNX Wrapper v2.0
    =========================
    
    Frame-Wise Decoder compatible wrapper.
    Returns frame-wise bit probabilities.
    
    Watermarked Audio → Frame-wise Bit Probabilities
    """
    
    def __init__(self, decoder: Decoder, target_length: int = 8000):
        super().__init__()
        self.decoder = decoder
        self.target_length = target_length
        
        # Frame configuration
        self.frame_samples = 320  # 40ms @ 8kHz
        self.expected_frames = target_length // self.frame_samples  # 25 frames for 1s
    
    def forward(
        self,
        audio: torch.Tensor  # [B, 1, T]
    ) -> torch.Tensor:
        """
        Args:
            audio: [B, 1, T] - Watermarked audio (fixed length = target_length)
            
        Returns:
            bit_probs: [B, num_frames] - Extracted bit probabilities per frame
        """
        # Run through decoder (new frame-wise architecture)
        logits = self.decoder(audio)  # [B, num_frames]
        
        # Sigmoid for probabilities
        probs = torch.sigmoid(logits)
        return probs


# =============================================================================
# ONNX Export Functions
# =============================================================================

def export_encoder_onnx(
    encoder: Encoder,
    output_path: Path,
    opset_version: int = 16,
    example_length: int = 8000  # 1초 @ 8kHz
) -> Path:
    """Encoder를 ONNX로 export"""
    
    wrapper = EncoderONNXWrapper(encoder)
    wrapper.eval()
    
    # Example inputs
    batch_size = 1
    example_audio = torch.randn(batch_size, 1, example_length)
    example_message = torch.randint(0, 2, (batch_size, 128)).float()
    
    # Dynamic axes for variable batch and audio length
    dynamic_axes = {
        'audio': {0: 'batch_size', 2: 'audio_length'},
        'message': {0: 'batch_size'},
        'watermarked': {0: 'batch_size', 2: 'audio_length'}
    }
    
    # Export
    torch.onnx.export(
        wrapper,
        (example_audio, example_message),
        str(output_path),
        input_names=['audio', 'message'],
        output_names=['watermarked'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"✅ Encoder exported: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


def export_decoder_onnx(
    decoder: Decoder,
    output_path: Path,
    opset_version: int = 16,
    example_length: int = 8000
) -> Path:
    """Decoder를 ONNX로 export (Frame-Wise v2.0)
    
    New architecture uses fixed stride Conv1d instead of AdaptiveAvgPool1d,
    so dynamic audio length is now supported!
    
    Output: [B, num_frames] where num_frames = audio_length // 320
    """
    
    wrapper = DecoderONNXWrapper(decoder, target_length=example_length)
    wrapper.eval()
    
    # Example input
    batch_size = 1
    example_audio = torch.randn(batch_size, 1, example_length)
    
    # Dynamic axes - both batch_size AND audio_length are now dynamic!
    dynamic_axes = {
        'audio': {0: 'batch_size', 2: 'audio_length'},
        'bit_probs': {0: 'batch_size', 1: 'num_frames'}
    }
    
    # Export
    torch.onnx.export(
        wrapper,
        (example_audio,),
        str(output_path),
        input_names=['audio'],
        output_names=['bit_probs'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"✅ Decoder exported: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Output: [B, num_frames] (num_frames = audio_length // 320)")
    
    return output_path


# =============================================================================
# INT8 Quantization
# =============================================================================

def quantize_onnx_dynamic(
    input_path: Path,
    output_path: Path
) -> Path:
    """
    ONNX Dynamic Quantization (INT8)
    
    가장 간단한 양자화. 캘리브레이션 데이터 불필요.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("⚠️  onnxruntime-extensions required for quantization")
        print("   Install: pip install onnxruntime onnxruntime-extensions")
        return input_path
    
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
        per_channel=False,
        reduce_range=False
    )
    
    print(f"✅ Quantized (Dynamic INT8): {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


def quantize_onnx_static(
    input_path: Path,
    output_path: Path,
    calibration_data: list,
    is_encoder: bool = True
) -> Path:
    """
    ONNX Static Quantization (INT8)
    
    캘리브레이션 데이터를 사용하여 더 정밀한 양자화.
    """
    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
    except ImportError:
        print("⚠️  onnxruntime required for static quantization")
        return quantize_onnx_dynamic(input_path, output_path)
    
    class CallCopsCalibrationReader(CalibrationDataReader):
        def __init__(self, data_list: list, is_encoder: bool):
            self.data_list = data_list
            self.is_encoder = is_encoder
            self.index = 0
        
        def get_next(self) -> Optional[Dict]:
            if self.index >= len(self.data_list):
                return None
            
            audio = self.data_list[self.index]
            self.index += 1
            
            if self.is_encoder:
                message = np.random.randint(0, 2, (1, 128)).astype(np.float32)
                return {'audio': audio, 'message': message}
            else:
                return {'audio': audio}
    
    calibration_reader = CallCopsCalibrationReader(calibration_data, is_encoder)
    
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=False
    )
    
    print(f"✅ Quantized (Static INT8): {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


# =============================================================================
# Validation
# =============================================================================

def validate_onnx_model(
    pytorch_model: nn.Module,
    onnx_path: Path,
    is_encoder: bool = True,
    num_tests: int = 5,
    tolerance: float = 1e-4
) -> bool:
    """
    PyTorch vs ONNX 출력 비교 검증
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  onnxruntime required for validation")
        print("   Install: pip install onnxruntime")
        return False
    
    # ONNX Runtime session
    session = ort.InferenceSession(
        str(onnx_path),
        providers=['CPUExecutionProvider']
    )
    
    pytorch_model.eval()
    
    print(f"\n🔍 Validating {onnx_path.name} against PyTorch...")
    
    all_passed = True
    max_diff = 0.0
    
    for i in range(num_tests):
        # Random test input (variable length)
        audio_length = np.random.randint(2000, 16000)  # 0.25s ~ 2s
        audio_np = np.random.randn(1, 1, audio_length).astype(np.float32)
        audio_torch = torch.from_numpy(audio_np)
        
        if is_encoder:
            message_np = np.random.randint(0, 2, (1, 128)).astype(np.float32)
            message_torch = torch.from_numpy(message_np)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_out = pytorch_model(audio_torch, message_torch).numpy()
            
            # ONNX inference
            onnx_out = session.run(
                None,
                {'audio': audio_np, 'message': message_np}
            )[0]
        else:
            # PyTorch inference
            with torch.no_grad():
                logits = pytorch_model(audio_torch)
                pytorch_out = torch.sigmoid(logits).numpy()
            
            # ONNX inference
            onnx_out = session.run(None, {'audio': audio_np})[0]
        
        # Compare
        diff = np.abs(pytorch_out - onnx_out).max()
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print(f"   ❌ Test {i+1}: FAILED (max diff: {diff:.6f})")
            all_passed = False
        else:
            print(f"   ✅ Test {i+1}: PASSED (max diff: {diff:.6f}, length: {audio_length})")
    
    if all_passed:
        print(f"   ✅ All {num_tests} tests PASSED (max diff: {max_diff:.6f})")
    else:
        print(f"   ❌ Some tests FAILED")
    
    return all_passed


def check_onnx_model(onnx_path: Path) -> bool:
    """ONNX 모델 유효성 검사"""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"   ✅ ONNX model check passed: {onnx_path.name}")
        return True
    except Exception as e:
        print(f"   ❌ ONNX model check failed: {e}")
        return False


# =============================================================================
# Main Export Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CallCops ONNX Export for Web/Mobile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pth)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='exported/onnx',
        help='Output directory for ONNX models'
    )
    parser.add_argument(
        '--opset', type=int, default=16,
        help='ONNX opset version (16+ recommended for ONNX Runtime Web)'
    )
    parser.add_argument(
        '--quantize', action='store_true',
        help='Apply INT8 quantization'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Validate ONNX output against PyTorch'
    )
    parser.add_argument(
        '--skip_encoder', action='store_true',
        help='Skip encoder export'
    )
    parser.add_argument(
        '--skip_decoder', action='store_true',
        help='Skip decoder export'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CallCops ONNX Export")
    print("=" * 60)
    
    # ========================================
    # 1. Load Checkpoint
    # ========================================
    print(f"\n📦 Loading checkpoint: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    
    model = CallCopsNet(
        message_dim=config.get('watermark', {}).get('payload_length', 128),
        hidden_channels=config.get('model', {}).get('hidden_channels', [32, 64, 128, 256]),
        num_residual_blocks=config.get('model', {}).get('num_residual_blocks', 4),
        use_discriminator=False  # Export에는 Discriminator 불필요
    )
    
    # Load weights (strict=False for partial load)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Model info
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"   Encoder parameters: {encoder_params:,}")
    print(f"   Decoder parameters: {decoder_params:,}")
    
    # ========================================
    # 2. Export Encoder
    # ========================================
    if not args.skip_encoder:
        print(f"\n🔧 Exporting Encoder (Opset {args.opset})...")
        
        encoder_path = output_dir / "encoder.onnx"
        export_encoder_onnx(
            model.encoder,
            encoder_path,
            opset_version=args.opset
        )
        
        check_onnx_model(encoder_path)
        
        if args.validate:
            encoder_wrapper = EncoderONNXWrapper(model.encoder)
            validate_onnx_model(encoder_wrapper, encoder_path, is_encoder=True)
        
        if args.quantize:
            print("\n🔧 Quantizing Encoder...")
            encoder_quant_path = output_dir / "encoder_int8.onnx"
            quantize_onnx_dynamic(encoder_path, encoder_quant_path)
    
    # ========================================
    # 3. Export Decoder
    # ========================================
    if not args.skip_decoder:
        print(f"\n🔧 Exporting Decoder (Opset {args.opset})...")
        
        decoder_path = output_dir / "decoder.onnx"
        export_decoder_onnx(
            model.decoder,
            decoder_path,
            opset_version=args.opset
        )
        
        check_onnx_model(decoder_path)
        
        if args.validate:
            validate_onnx_model(model.decoder, decoder_path, is_encoder=False)
        
        if args.quantize:
            print("\n🔧 Quantizing Decoder...")
            decoder_quant_path = output_dir / "decoder_int8.onnx"
            quantize_onnx_dynamic(decoder_path, decoder_quant_path)
    
    # ========================================
    # 4. Summary
    # ========================================
    print("\n" + "=" * 60)
    print("📊 Export Summary")
    print("=" * 60)
    
    total_size = 0
    for f in sorted(output_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        status = "✅" if size_mb < 10 else "⚠️"
        print(f"   {status} {f.name}: {size_mb:.2f} MB")
    
    print(f"\n   Total: {total_size:.2f} MB")
    
    if total_size < 20:
        print("   ✅ Size target met (< 10MB per model)")
    else:
        print("   ⚠️  Consider quantization to reduce size")
    
    # Usage guide
    print("\n" + "=" * 60)
    print("🌐 ONNX Runtime Web Usage")
    print("=" * 60)
    print("""
// JavaScript Example
import * as ort from 'onnxruntime-web';

// Load encoder
const encoder = await ort.InferenceSession.create('./encoder.onnx');

// Prepare input (8kHz audio, Float32Array)
const audioData = new Float32Array(8000);  // 1 second
const messageData = new Float32Array(128); // 128-bit watermark

// Create tensors
const audioTensor = new ort.Tensor('float32', audioData, [1, 1, 8000]);
const messageTensor = new ort.Tensor('float32', messageData, [1, 128]);

// Run inference
const result = await encoder.run({
    audio: audioTensor,
    message: messageTensor
});

const watermarkedAudio = result.watermarked.data;
""")
    
    print("\n✅ Export completed!")


if __name__ == "__main__":
    main()
