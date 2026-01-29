"""
CallCops: Messenger Module
==========================

텔레그램 봇을 이용한 학습 상태 모니터링 및 알림 모듈.
.env 파일에서 보안 정보를 로드하여 사용합니다.
"""

import os
import io
import platform
import subprocess
import requests
import matplotlib
import matplotlib.pyplot as plt
import GPUtil
from dotenv import load_dotenv
from typing import Dict, Any

# Headless 서버를 위한 백엔드 설정
matplotlib.use('Agg')

class CallCopsMessenger:
    def __init__(self, env_path: str = ".env"):
        """
        초기화: .env 파일에서 토큰 로드
        """
        load_dotenv(env_path)
        
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        
        if not self.enabled:
            print("[Messenger] Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in .env")
            print("[Messenger] Notifications are disabled.")
        else:
            print(f"[Messenger] Initialized for Chat ID: {self.chat_id}")

    def send_message(self, text: str):
        """텍스트 메시지 전송"""
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"[Messenger] Failed to send message: {e}")

    def send_plot(self, history: Dict[str, list], title: str = "Training Progress"):
        """
        학습 이력(Loss, BER) 그래프 생성 및 전송
        
        Args:
            history: {'train_loss': [], 'val_loss': [], 'val_ber': [], ...}
        """
        if not self.enabled:
            return

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 1. Loss Plot
            if 'train_loss' in history:
                ax1.plot(history['train_loss'], label='Train Loss')
            if 'val_loss' in history:
                ax1.plot(history['val_loss'], label='Val Loss')
            ax1.set_title("Loss Curve")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            # 2. BER Plot
            if 'train_ber' in history:
                ax1.plot(history['train_ber'], label='Train BER', linestyle='--')
            if 'val_ber' in history:
                ax2.plot(history['val_ber'], label='Val BER', color='orange')
            
            # Target line
            ax2.axhline(y=0.05, color='r', linestyle=':', label='Target (5%)')
            
            ax2.set_title("Bit Error Rate (BER)")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("BER")
            ax2.set_ylim(0, 0.5)  # BER은 0~0.5 범위가 중요
            ax2.legend()
            ax2.grid(True)

            plt.suptitle(title)
            plt.tight_layout()

            # 메모리 버퍼에 저장
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)

            # 전송
            url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
            files = {'photo': ('plot.png', buf, 'image/png')}
            data = {'chat_id': self.chat_id}
            
            requests.post(url, data=data, files=files, timeout=20)
            
        except Exception as e:
            print(f"[Messenger] Failed to send plot: {e}")

    def get_system_info(self) -> str:
        """현재 시스템 리소스 상태 반환"""
        info = []
        
        # 1. CPU Load (Simple approximation via loadavg)
        if hasattr(os, 'getloadavg'):
            load = os.getloadavg()
            info.append(f"🖥️ *CPU Load*: {load[0]:.2f} / {load[1]:.2f} / {load[2]:.2f}")
        
        # 2. Memory (using psutil if available, else skip)
        try:
            import psutil
            mem = psutil.virtual_memory()
            info.append(f"💾 *RAM*: {mem.percent}% ({mem.used / 1e9:.1f}/{mem.total / 1e9:.1f} GB)")
        except ImportError:
            pass

        # 3. GPU (using GPUtil)
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                info.append(f"🚀 *GPU {i} ({gpu.name})*: {gpu.load*100:.0f}% Load, {gpu.memoryUsed}/{gpu.memoryTotal} MB VRAM")
        except Exception:
            info.append("🚀 *GPU*: Info unavailable")

        return "\n".join(info)

if __name__ == "__main__":
    # Test
    messenger = CallCopsMessenger()
    print(messenger.get_system_info())
    if messenger.enabled:
        messenger.send_message("🔔 CallCops Messenger Test: System Online")
