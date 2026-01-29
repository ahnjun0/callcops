"""
CallCops: Training Script
==============================

RTAW 모델 학습 스크립트.

학습 전략:
1. Generator (Encoder + Decoder) 학습
2. Discriminator 학습
3. Codec Augmentation을 통한 robustness 강화
4. Curriculum Learning: 점진적 난이도 증가

품질 목표:
- PESQ >= 4.0
- BER < 5% (G.729 압축 후)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from models import RTAWNet, CallCopsLoss, DifferentiableCodecSimulator
from scripts.dataset import create_dataloader


class Trainer:
    """
    CallCops 모델 트레이너
    ===========================

    학습 루프 및 검증 관리.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.config = config
        self.device = device

        # 모델 초기화
        self.model = RTAWNet(
            bits_dim=config['watermark']['payload_length'],
            encoder_config=config.get('model', {}).get('encoder', {}),
            decoder_config=config.get('model', {}).get('decoder', {}),
            discriminator_config=config.get('model', {}).get('discriminator', {})
        ).to(device)

        # 코덱 시뮬레이터
        self.codec_sim = DifferentiableCodecSimulator(
            codec_types=config.get('codec', {}).get('types', ['g711_alaw', 'g729', 'none']),
            curriculum_epochs=config.get('training', {}).get('curriculum_epochs', 10)
        ).to(device)

        # 손실 함수
        training_config = config.get('training', {})
        self.criterion = CallCopsLoss(
            lambda_bit=training_config.get('lambda_bit', 1.0),
            lambda_audio=training_config.get('lambda_audio', 10.0),
            lambda_adv=training_config.get('lambda_adv', 0.1),
            sample_rate=config.get('audio', {}).get('sample_rate', 8000)
        ).to(device)

        # Optimizer
        lr = training_config.get('learning_rate', 0.0001)
        betas = training_config.get('adam_betas', [0.5, 0.9])

        # Generator optimizer (Encoder + Decoder)
        self.optim_g = optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()),
            lr=lr,
            betas=tuple(betas)
        )

        # Discriminator optimizer
        self.optim_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=lr,
            betas=tuple(betas)
        )

        # Scheduler
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optim_g,
            T_max=training_config.get('epochs', 100)
        )
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optim_d,
            T_max=training_config.get('epochs', 100)
        )

        # Gradient clipping
        self.grad_clip = training_config.get('grad_clip', 1.0)

        # 학습 상태
        self.current_epoch = 0
        self.global_step = 0
        self.best_ber = 1.0

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """단일 학습 스텝"""
        audio = batch['audio'].to(self.device)
        bits = batch['bits'].to(self.device)

        # ==================
        # Generator 학습
        # ==================
        self.optim_g.zero_grad()

        # Forward pass
        watermarked, attention = self.model.embed(audio, bits)

        # 코덱 시뮬레이션 (robustness 강화)
        watermarked_codec, codec_used = self.codec_sim(watermarked)

        # 비트 추출
        bit_probs, detection = self.model.extract(watermarked_codec)

        # Discriminator forward (for GAN loss)
        with torch.no_grad():
            disc_real = self.model.discriminator(audio)
        disc_fake = self.model.discriminator(watermarked)

        # 손실 계산
        losses = self.criterion(
            pred_audio=watermarked,
            target_audio=audio,
            pred_bits=bit_probs,
            target_bits=bits,
            detection_pred=detection,
            disc_fake=disc_fake,
            disc_real=disc_real
        )

        # Generator backward
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()),
            self.grad_clip
        )

        self.optim_g.step()

        # ==================
        # Discriminator 학습
        # ==================
        self.optim_d.zero_grad()

        # Discriminator forward (detached watermarked)
        disc_real = self.model.discriminator(audio)
        disc_fake = self.model.discriminator(watermarked.detach())

        # Discriminator loss
        d_loss = self.criterion.adv_loss.discriminator_loss(disc_real, disc_fake)
        d_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.discriminator.parameters(),
            self.grad_clip
        )

        self.optim_d.step()

        # 메트릭
        metrics = self.criterion.compute_metrics(bit_probs, bits)

        return {
            'loss_total': losses['total'].item(),
            'loss_bit': losses['bit'].item(),
            'loss_mel': losses['mel'].item(),
            'loss_adv_g': losses['adv_g'].item(),
            'loss_adv_d': d_loss.item(),
            'ber': metrics['ber'],
            'accuracy': metrics['accuracy'],
            'codec': codec_used
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """검증 루프"""
        self.model.eval()

        total_metrics = {
            'loss': 0.0,
            'ber': 0.0,
            'ber_g711': 0.0,
            'ber_g729': 0.0,
            'accuracy': 0.0
        }
        num_batches = 0

        for batch in val_loader:
            audio = batch['audio'].to(self.device)
            bits = batch['bits'].to(self.device)

            # Forward pass
            watermarked, _ = self.model.embed(audio, bits)

            # 코덱 없이 추출
            bit_probs, detection = self.model.extract(watermarked)
            metrics = self.criterion.compute_metrics(bit_probs, bits)
            total_metrics['ber'] += metrics['ber']
            total_metrics['accuracy'] += metrics['accuracy']

            # G.711 A-law 후 추출
            watermarked_g711, _ = self.codec_sim(watermarked, codec_type='g711_alaw')
            bit_probs_g711, _ = self.model.extract(watermarked_g711)
            metrics_g711 = self.criterion.compute_metrics(bit_probs_g711, bits)
            total_metrics['ber_g711'] += metrics_g711['ber']

            # G.729 후 추출
            watermarked_g729, _ = self.codec_sim(watermarked, codec_type='g729')
            bit_probs_g729, _ = self.model.extract(watermarked_g729)
            metrics_g729 = self.criterion.compute_metrics(bit_probs_g729, bits)
            total_metrics['ber_g729'] += metrics_g729['ber']

            # 손실
            losses = self.criterion(
                pred_audio=watermarked,
                target_audio=audio,
                pred_bits=bit_probs,
                target_bits=bits,
                detection_pred=detection
            )
            total_metrics['loss'] += losses['total'].item()

            num_batches += 1

        # 평균
        for key in total_metrics:
            total_metrics[key] /= num_batches

        self.model.train()

        return total_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Path,
        log_dir: Optional[Path] = None
    ):
        """
        전체 학습 루프

        Args:
            train_loader: 학습 DataLoader
            val_loader: 검증 DataLoader
            num_epochs: 에포크 수
            save_dir: 체크포인트 저장 디렉토리
            log_dir: TensorBoard 로그 디렉토리
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        writer = None
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir)

        print("=" * 60)
        print("CallCops Training Started")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("=" * 60)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Curriculum learning: 코덱 난이도 조절
            self.codec_sim.set_epoch(epoch)

            # 학습 루프
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            epoch_metrics = {
                'loss_total': 0.0,
                'loss_bit': 0.0,
                'ber': 0.0
            }

            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch)

                # 누적
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key] += metrics[key]

                self.global_step += 1

                # Progress bar 업데이트
                pbar.set_postfix({
                    'loss': f"{metrics['loss_total']:.4f}",
                    'ber': f"{metrics['ber']:.4f}",
                    'codec': metrics['codec']
                })

                # TensorBoard 로깅
                if writer and self.global_step % 100 == 0:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f'train/{key}', value, self.global_step)

            # 에포크 평균
            num_batches = len(train_loader)
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

            # 검증
            val_metrics = self.validate(val_loader)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {epoch_metrics['loss_total']:.4f}")
            print(f"  Train BER: {epoch_metrics['ber']:.4f}")
            print(f"  Val BER (no codec): {val_metrics['ber']:.4f}")
            print(f"  Val BER (G.711): {val_metrics['ber_g711']:.4f}")
            print(f"  Val BER (G.729): {val_metrics['ber_g729']:.4f}")

            # TensorBoard 검증 로깅
            if writer:
                for key, value in val_metrics.items():
                    writer.add_scalar(f'val/{key}', value, epoch)

            # Scheduler step
            self.scheduler_g.step()
            self.scheduler_d.step()

            # 체크포인트 저장
            is_best = val_metrics['ber_g729'] < self.best_ber
            if is_best:
                self.best_ber = val_metrics['ber_g729']

            self.save_checkpoint(
                save_dir / f"checkpoint_epoch{epoch+1}.pt",
                val_metrics
            )

            if is_best:
                self.save_checkpoint(
                    save_dir / "best_model.pt",
                    val_metrics
                )
                print(f"  ★ New best model! BER (G.729): {self.best_ber:.4f}")

        if writer:
            writer.close()

        print("\n" + "=" * 60)
        print("Training Completed!")
        print(f"Best BER (G.729): {self.best_ber:.4f}")
        print("=" * 60)

    def save_checkpoint(
        self,
        path: Path,
        metrics: Dict[str, float]
    ):
        """체크포인트 저장"""
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optim_g_state_dict': self.optim_g.state_dict(),
            'optim_d_state_dict': self.optim_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_ber': self.best_ber,
            'metrics': metrics,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: Path):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim_g.load_state_dict(checkpoint['optim_g_state_dict'])
        self.optim_d.load_state_dict(checkpoint['optim_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_ber = checkpoint['best_ber']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"Best BER: {self.best_ber:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CallCops Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/raw/training',
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='data/raw/validation',
                        help='Path to validation data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')

    args = parser.parse_args()

    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # CLI 인자로 설정 오버라이드
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # DataLoader 생성
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        mode='train',
        sample_rate=config['audio']['sample_rate'],
        frame_ms=config['audio']['frame_ms']
    )

    val_dir = args.val_dir or args.data_dir
    val_loader = create_dataloader(
        data_dir=val_dir,
        batch_size=config['training']['batch_size'],
        mode='val',
        sample_rate=config['audio']['sample_rate'],
        frame_ms=config['audio']['frame_ms']
    )

    # 트레이너 초기화
    trainer = Trainer(config, device)

    # 체크포인트 복원
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # 저장 디렉토리 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / timestamp
    log_dir = Path(args.log_dir) / timestamp

    # 학습 시작
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        save_dir=save_dir,
        log_dir=log_dir
    )


if __name__ == "__main__":
    main()
