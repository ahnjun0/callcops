"""
CallCops: Training Script
==============================

Real-Time Audio Watermarking 모델 학습 스크립트.

학습 전략:
1. Generator (Encoder + Decoder) 학습: Bit Loss + Audio Loss + Adversarial Loss
2. Discriminator 학습: Real vs Fake 판별
3. Codec Augmentation (Optional): G.711/G.729 robustness 강화

품질 목표:
- PESQ >= 4.0
- BER < 5%

Usage:
    python scripts/train.py --epochs 100 --batch_size 64
    python scripts/train.py --config configs/default.yaml --resume checkpoints/latest.pth
"""

import os
import sys
import argparse
import traceback
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CallCopsNet, CallCopsLoss, DifferentiableCodecSimulator
from scripts.dataset import create_train_val_loaders, create_dataloader
from utils.messenger import CallCopsMessenger


# =============================================================================
# Utility Functions
# =============================================================================

def compute_snr(original: torch.Tensor, watermarked: torch.Tensor) -> float:
    """
    Signal-to-Noise Ratio 계산

    SNR = 10 * log10(mean(x²) / mean((x-x')²))
    Note: sum 대신 mean을 사용하여 긴 오디오/큰 배치에서의 오버플로우 방지

    Args:
        original: [B, 1, T] 원본 오디오
        watermarked: [B, 1, T] 워터마크된 오디오

    Returns:
        SNR in dB
    """
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - watermarked) ** 2)

    if noise_power < 1e-10:
        return 100.0  # 거의 동일한 경우

    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


def compute_ber(pred_logits: torch.Tensor, target_bits: torch.Tensor) -> float:
    """
    Bit Error Rate 계산 (프레임 단위 호환)

    Args:
        pred_logits: [B, num_frames] 예측된 로짓 (프레임별)
        target_bits: [B, 128] 또는 [B, num_frames] 목표 비트

    Returns:
        BER (0~1)
    """
    B, num_frames = pred_logits.shape
    
    # target_bits를 프레임 수에 맞게 확장 (Cyclic)
    if target_bits.shape[1] != num_frames:
        # target_bits: [B, 128] -> [B, num_frames]
        frame_indices = torch.arange(num_frames, device=target_bits.device) % target_bits.shape[1]
        target_bits_expanded = target_bits[:, frame_indices]
    else:
        target_bits_expanded = target_bits
    
    pred_bits = (torch.sigmoid(pred_logits) > 0.5).float()
    errors = (pred_bits != target_bits_expanded).float()
    return errors.mean().item()


def get_frame_target_bits(bits: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    128비트 페이로드를 프레임 수에 맞게 Cyclic 확장
    
    Args:
        bits: [B, 128] 원본 페이로드
        num_frames: 타겟 프레임 수
        
    Returns:
        frame_bits: [B, num_frames] 프레임별 비트
    """
    frame_indices = torch.arange(num_frames, device=bits.device) % bits.shape[1]
    return bits[:, frame_indices]


# =============================================================================
# Trainer Class
# =============================================================================

class CallCopsTrainer:
    """
    CallCops 모델 트레이너
    ======================

    GAN 기반 학습 루프:
    1. Discriminator Update: Real vs Fake 판별
    2. Generator Update: Bit Loss + Audio Loss + Adversarial Loss
    """

    def __init__(
        self,
        model: CallCopsNet,
        loss_fn: CallCopsLoss,
        opt_g: optim.Optimizer,
        opt_d: optim.Optimizer,
        device: torch.device,
        codec_sim: Optional[DifferentiableCodecSimulator] = None,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        messenger: Optional[CallCopsMessenger] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.device = device
        self.codec_sim = codec_sim
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.messenger = messenger

        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        # 학습 상태
        self.current_epoch = 0
        self.global_step = 0
        self.best_ber = 1.0
        self.best_loss = float('inf')
        
        # 학습 이력 (Plot팅용)
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ber': [], 'val_ber': []
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        단일 학습 스텝 (GAN Training with AMP)

        1. Discriminator Update: Maximize log(D(real)) + log(1 - D(fake))
        2. Generator Update: Minimize total loss (bit + audio + adv)
        """
        audio = batch['audio'].to(self.device)
        bits = batch['bits'].to(self.device)

        # ========================================
        # 1. Discriminator Update
        # ========================================
        self.opt_d.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
            with torch.no_grad():
                # Generator forward (no grad for D update)
                watermarked, _ = self.model.embed(audio, bits)

            # Discriminator forward
            disc_real = self.model.discriminator(audio)
            disc_fake = self.model.discriminator(watermarked.detach())

            # Discriminator loss
            d_loss = self.loss_fn.adv_loss.discriminator_loss(disc_real, disc_fake)

        # Backward & Step
        if self.use_amp:
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.opt_d)
            torch.nn.utils.clip_grad_norm_(
                self.model.discriminator.parameters(),
                self.grad_clip
            )
            self.scaler.step(self.opt_d)
            # scaler.update()는 Generator까지 다 끝난 후 한 번만 호출
        else:
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.discriminator.parameters(),
                self.grad_clip
            )
            self.opt_d.step()

        # ========================================
        # 2. Generator Update
        # ========================================
        self.opt_g.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
            # Generator forward
            watermarked, _ = self.model.embed(audio, bits)

            # Codec simulation (optional, for robustness)
            if self.codec_sim is not None:
                watermarked_degraded, codec_used = self.codec_sim(watermarked)
            else:
                watermarked_degraded = watermarked
                codec_used = 'none'

            # Decoder forward (returns frame-wise logits)
            pred_logits = self.model.decoder(watermarked_degraded)  # [B, num_frames]

            # 프레임 수에 맞게 타겟 비트 확장 (Cyclic)
            num_frames = pred_logits.shape[1]
            target_bits_expanded = get_frame_target_bits(bits, num_frames)

            # Detection logits: 비트 로짓의 절대값 평균
            detection_logits = torch.abs(pred_logits).mean(dim=1, keepdim=True)

            # Discriminator forward (for generator loss)
            disc_fake = self.model.discriminator(watermarked)

            # Generator losses (프레임 단위 비트 비교)
            losses = self.loss_fn(
                pred_audio=watermarked,
                target_audio=audio,
                pred_bits=pred_logits,  # [B, num_frames] logits
                target_bits=target_bits_expanded,  # [B, num_frames] expanded
                detection_pred=detection_logits,
                disc_fake=disc_fake
            )
            
            g_loss = losses['total']

        # Backward & Step
        if self.use_amp:
            self.scaler.scale(g_loss).backward()
            self.scaler.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.parameters()) +
                list(self.model.decoder.parameters()),
                self.grad_clip
            )
            self.scaler.step(self.opt_g)
            
            # Update scaler once per iteration
            self.scaler.update()
        else:
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.parameters()) +
                list(self.model.decoder.parameters()),
                self.grad_clip
            )
            self.opt_g.step()

        # ========================================
        # 3. Metrics
        # ========================================
        with torch.no_grad():
            # CPU로 이동하여 계산 (메모리 누수 방지)
            ber = compute_ber(pred_logits.detach().float(), bits.detach().float())
            snr = compute_snr(audio.detach().float(), watermarked.detach().float())

        self.global_step += 1

        return {
            'loss_total': losses['total'].item(),
            'loss_bit': losses['bit'].item(),
            'loss_mel': losses['mel'].item(),
            'loss_stft': losses['stft'].item(),
            'loss_adv_g': losses['adv_g'].item(),
            'loss_adv_d': d_loss.item(),
            'ber': ber,
            'snr': snr,
            'codec': codec_used
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증 루프"""
        self.model.eval()

        total_metrics = {
            'loss': 0.0,
            'ber': 0.0,
            'snr': 0.0,
            'accuracy': 0.0
        }
        num_batches = 0

        for batch in val_loader:
            audio = batch['audio'].to(self.device)
            bits = batch['bits'].to(self.device)

            # Forward
            watermarked, _ = self.model.embed(audio, bits)
            pred_logits = self.model.decoder(watermarked)  # [B, num_frames]
            
            # 프레임 수에 맞게 타겟 비트 확장 (Cyclic)
            num_frames = pred_logits.shape[1]
            target_bits_expanded = get_frame_target_bits(bits, num_frames)
            
            detection_logits = torch.abs(pred_logits).mean(dim=1, keepdim=True)

            # Losses (프레임 단위 비교)
            losses = self.loss_fn(
                pred_audio=watermarked,
                target_audio=audio,
                pred_bits=pred_logits,
                target_bits=target_bits_expanded,
                detection_pred=detection_logits
            )

            # Metrics
            ber = compute_ber(pred_logits, bits)
            snr = compute_snr(audio, watermarked)

            total_metrics['loss'] += losses['total'].item()
            total_metrics['ber'] += ber
            total_metrics['snr'] += snr
            total_metrics['accuracy'] += (1.0 - ber)

            num_batches += 1

        # Average
        for key in total_metrics:
            total_metrics[key] /= max(num_batches, 1)

        self.model.train()
        return total_metrics

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """에포크 학습"""
        self.model.train()
        self.current_epoch = epoch

        # Codec curriculum (optional)
        if self.codec_sim is not None:
            self.codec_sim.set_epoch(epoch)

        epoch_metrics = {
            'loss_total': 0.0,
            'loss_bit': 0.0,
            'ber': 0.0,
            'snr': 0.0
        }

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{total_epochs}",
            leave=True
        )

        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)

            # 누적
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]

            # Progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss_total']:.4f}",
                'ber': f"{metrics['ber']:.4f}",
                'snr': f"{metrics['snr']:.1f}dB"
            })

        # Average
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        return epoch_metrics

    def save_checkpoint(
        self,
        path: Path,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
        is_latest: bool = False
    ):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'best_ber': self.best_ber,
            'best_loss': self.best_loss,
            'metrics': metrics,
            'history': self.history
        }

        if config is not None:
            checkpoint['config'] = config

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Save latest link (or copy)
        if is_latest:
            latest_path = path.parent / "latest.pth"
            try:
                # 덮어쓰기 위해 기존 파일 삭제
                if latest_path.exists():
                    latest_path.unlink()
                # 복사가 더 안정적일 수 있음 (특히 파일시스템에 따라)
                shutil.copy(path, latest_path)
            except Exception as e:
                print(f"Warning: Failed to create latest.pth: {e}")

    def load_checkpoint(self, path: Path):
        """체크포인트 로드"""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
        self.opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_ber = checkpoint.get('best_ber', 1.0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"Best BER: {self.best_ber:.4f}")

        return checkpoint.get('config')


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    save_dir: Path,
    device: torch.device,
    resume_path: Optional[Path] = None
):
    """
    메인 학습 함수
    """
    # 0. 메신저 초기화
    messenger = CallCopsMessenger()
    messenger.send_message(f"🚀 **CallCops Training Started**\nDevice: {device}\nSave Dir: `{save_dir}`")

    try:
        # ========================================
        # 1. Model 초기화
        # ========================================
        training_config = config.get('training', {})
        model_config = config.get('model', {})

        model = CallCopsNet(
            message_dim=config.get('watermark', {}).get('payload_length', 128),
            hidden_channels=model_config.get('hidden_channels', [32, 64, 128, 256]),
            num_residual_blocks=model_config.get('num_residual_blocks', 4),
            use_discriminator=True
        ).to(device)

        # 파라미터 수 출력
        params = model.count_parameters()
        print(f"\nModel Parameters:")
        print(f"  Encoder: {params['encoder']:,}")
        print(f"  Decoder: {params['decoder']:,}")
        print(f"  Discriminator: {params['discriminator']:,}")
        print(f"  Total: {params['total']:,}")

        # ========================================
        # 2. Loss Function
        # ========================================
        loss_fn = CallCopsLoss(
            lambda_bit=training_config.get('lambda_bit', 10.0),
            lambda_audio=training_config.get('lambda_audio', 10.0),
            lambda_adv=training_config.get('lambda_adv', 0.1),
            lambda_det=training_config.get('lambda_det', 0.5),
            lambda_stft=training_config.get('lambda_stft', 2.0),
            lambda_l1=training_config.get('lambda_l1', 10.0),  # NEW: Direct L1 loss for SNR
            sample_rate=config.get('audio', {}).get('sample_rate', 8000)
        ).to(device)

        # ========================================
        # 3. Optimizers (LR: 2e-4)
        # ========================================
        lr = training_config.get('learning_rate', 2e-4)
        betas = tuple(training_config.get('adam_betas', [0.5, 0.9]))

        opt_g = optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=lr,
            betas=betas
        )

        opt_d = optim.Adam(
            model.discriminator.parameters(),
            lr=lr,
            betas=betas
        )

        # ========================================
        # 4. Codec Simulator (Optional)
        # ========================================
        codec_sim = None
        codec_config = config.get('codec', {})
        if codec_config.get('enabled', False):
            codec_sim = DifferentiableCodecSimulator(
                codec_types=codec_config.get('types', ['g711_alaw', 'g729', 'none']),
                curriculum_epochs=training_config.get('curriculum_epochs', 10)
            ).to(device)

        # ========================================
        # 5. Trainer
        # ========================================
        trainer = CallCopsTrainer(
            model=model,
            loss_fn=loss_fn,
            opt_g=opt_g,
            opt_d=opt_d,
            device=device,
            codec_sim=codec_sim,
            grad_clip=training_config.get('grad_clip', 1.0),
            use_amp=training_config.get('use_amp', True),  # AMP Enabled by default
            messenger=messenger
        )

        # 체크포인트 복원
        if resume_path and resume_path.exists():
            trainer.load_checkpoint(resume_path)
            messenger.send_message(f"🔄 **Resumed Training** from epoch {trainer.current_epoch}")

        # ========================================
        # 6. Training Loop
        # ========================================
        num_epochs = training_config.get('epochs', 100)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("CallCops Training Started")
        print("=" * 60)
        print(f"  Device: {device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {config.get('training', {}).get('batch_size', 64)}")
        print(f"  Learning rate: {lr}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Codec simulation: {'Enabled' if codec_sim else 'Disabled'}")
        print("=" * 60 + "\n")

        for epoch in range(trainer.current_epoch, num_epochs):
            # Train
            train_metrics = trainer.train_epoch(train_loader, epoch, num_epochs)

            # Validate
            val_metrics = trainer.validate(val_loader)

            # Update History
            trainer.history['train_loss'].append(train_metrics['loss_total'])
            trainer.history['val_loss'].append(val_metrics['loss'])
            trainer.history['train_ber'].append(train_metrics['ber'])
            trainer.history['val_ber'].append(val_metrics['ber'])

            # Summary
            summary_text = (
                f"✅ **Epoch {epoch+1}/{num_epochs}**\n"
                f"📉 Train Loss: `{train_metrics['loss_total']:.4f}`\n"
                f"📉 Val Loss: `{val_metrics['loss']:.4f}`\n"
                f"🎯 **Val BER**: `{val_metrics['ber']:.4f}`\n"
                f"🔊 Val SNR: `{val_metrics['snr']:.1f}dB`"
            )
            
            print(f"\n{summary_text.replace('**', '').replace('`', '')}")

            # Send Notification
            if messenger:
                messenger.send_message(f"{summary_text}\n\n{messenger.get_system_info()}")

                # Send Plot every 10 epochs
                if (epoch + 1) % 10 == 0:
                    messenger.send_plot(trainer.history, title=f"Training Status (Epoch {epoch+1})")

            # Save Latest Checkpoint (매 에포크마다)
            trainer.save_checkpoint(
                save_dir / f"checkpoint_epoch{epoch+1}.pth",
                val_metrics,
                config,
                is_latest=True
            )

            # 1. Best BER Model
            is_best_ber = val_metrics['ber'] < trainer.best_ber
            if is_best_ber:
                trainer.best_ber = val_metrics['ber']
                trainer.save_checkpoint(
                    save_dir / "best_ber_model.pth",
                    val_metrics,
                    config
                )
                print(f"  ★ New best BER model! BER: {trainer.best_ber:.4f}")
                if messenger:
                    messenger.send_message(f"🏆 **New Best BER!** `{trainer.best_ber:.4f}`")

            # 2. Best Loss Model
            is_best_loss = val_metrics['loss'] < trainer.best_loss
            if is_best_loss:
                trainer.best_loss = val_metrics['loss']
                trainer.save_checkpoint(
                    save_dir / "best_loss_model.pth",
                    val_metrics,
                    config
                )
                print(f"  ★ New best Loss model! Loss: {trainer.best_loss:.4f}")

            # 주기적 영구 저장 (10 에포크마다 별도 파일로 남김)
            if (epoch + 1) % 10 == 0:
                print(f"  Creating permanent checkpoint for epoch {epoch+1}...")
                # save_checkpoint에서 이미 저장했으므로 별도 작업 불필요

            print()

        # 최종 저장
        trainer.save_checkpoint(
            save_dir / "final_model.pth",
            val_metrics,
            config
        )

        print("=" * 60)
        print("Training Completed!")
        print(f"Best BER: {trainer.best_ber:.4f}")
        print(f"Checkpoints saved to: {save_dir}")
        print("=" * 60)
        
        if messenger:
            messenger.send_message(f"🎉 **Training Completed**\nBest BER: `{trainer.best_ber:.4f}`")

    except Exception as e:
        error_msg = f"❌ **Training Crashed!**\n\n```\n{traceback.format_exc()[-1000:]}\n```"
        print(traceback.format_exc())
        if messenger:
            messenger.send_message(error_msg)
        sys.exit(1)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CallCops Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 데이터 경로
    parser.add_argument(
        '--train_dir', type=str, default='data/raw/training',
        help='Training data directory'
    )
    parser.add_argument(
        '--val_dir', type=str, default='data/raw/validation',
        help='Validation data directory'
    )

    # 학습 설정
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size (Default: 64 for RTX 3090)'
    )
    parser.add_argument(
        '--lr', type=float, default=2e-4,
        help='Learning rate'
    )

    # 모델 설정
    parser.add_argument(
        '--message_dim', type=int, default=128,
        help='Watermark message dimension (bits)'
    )

    # 경로 설정
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file (overrides CLI args)'
    )
    parser.add_argument(
        '--save_dir', type=str, default='checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )

    # 디바이스 설정
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )

    # 기타
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    # 디버깅 및 안정성
    parser.add_argument(
        '--no_amp', action='store_true',
        help='Disable Mixed Precision (AMP) training (Recommended if loss=nan occurs)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable Anomaly Detection for debugging NaNs'
    )

    args = parser.parse_args()

    # ========================================
    # 디버그 모드 설정
    # ========================================
    if args.debug:
        print("\n⚠️ DEBUG MODE ENABLED: Anomaly Detection is ON (This will slow down training)")
        torch.autograd.set_detect_anomaly(True)

    # ========================================
    # Config 로드 또는 생성
    # ========================================
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {args.config}")
    else:
        # CLI 인자로 config 생성
        config = {
            'audio': {
                'sample_rate': 8000,
                'frame_ms': 40
            },
            'watermark': {
                'payload_length': args.message_dim
            },
            'model': {
                'hidden_channels': [32, 64, 128, 256],
                'num_residual_blocks': 4
            },
            'training': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'adam_betas': [0.5, 0.9],
                'grad_clip': 1.0,
                'lambda_bit': 10.0,
                'lambda_audio': 10.0,
                'lambda_adv': 0.1,
                'lambda_det': 0.5,
                'lambda_stft': 2.0,
                'use_amp': not args.no_amp  # CLI 인자로 제어
            },
            'codec': {
                'enabled': False,
                'types': ['g711_alaw', 'g729', 'none']
            }
        }

    # CLI 인자로 오버라이드
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    config['training']['use_amp'] = not args.no_amp

    # ========================================
    # 디바이스 설정
    # ========================================
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Mixed Precision: {'Enabled' if config['training']['use_amp'] else 'Disabled'}")

    # ========================================
    # 자동 Resume 확인
    # ========================================
    resume_path = args.resume
    if resume_path is None:
        latest_ckpt = Path("checkpoints/latest.pth")
        if latest_ckpt.exists():
            print(f"Found latest checkpoint: {latest_ckpt}")
            resume_path = str(latest_ckpt)

    # ========================================
    # 랜덤 시드
    # ========================================
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # ========================================
    # DataLoader 생성
    # ========================================
    print(f"\nLoading data...")

    train_loader, val_loader = create_train_val_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers,
        sample_rate=config['audio']['sample_rate'],
        pin_memory=True  # RTX 3090 최적화
    )

    # ========================================
    # 저장 디렉토리 설정
    # ========================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / timestamp

    # ========================================
    # 학습 시작
    # ========================================
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=save_dir,
        device=device,
        resume_path=Path(resume_path) if resume_path else None
    )


if __name__ == "__main__":
    main()