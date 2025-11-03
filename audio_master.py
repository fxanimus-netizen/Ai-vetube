# -*- coding: utf-8 -*-
"""
masters/audio_master.py — Аудио-мастер VTuber системы

Управляет:
- Распознаванием речи (STT)
- Синтезом речи (TTS)
- Управлением аудиоустройствами
- Обработкой ошибок аудио

Версия: 1.0 (2025-11-03)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import BaseMaster

# Импорты аудио-модулей
from audio.stt import WhisperSTT
from audio.tts import TTS

# Импорты конфигурации
from core.config import VTuberConfig, load_config

logger = logging.getLogger("MasterAudio")


class MasterAudio(BaseMaster):
    """
    Аудио-мастер — управление STT и TTS.
    
    Возможности:
    - Распознавание речи (STT)
    - Синтез речи (TTS)
    - Управление аудиоустройствами
    - Обработка ошибок аудио
    
    API:
    - listen() — слушать пользователя (с timeout)
    - speak() — озвучить текст с эмоцией
    - stop_speaking() — остановить воспроизведение
    - is_speaking() — проверить, идёт ли воспроизведение
    """
    
    def __init__(
        self,
        config: Optional[VTuberConfig] = None,
        stt_device: Optional[str] = None,
        tts_device: Optional[str] = None,
    ):
        super().__init__("Audio")
        
        # Конфигурация
        self.config = config or VTuberConfig.load()
        
        # Устройства (определяются автоматически, если не заданы)
        self._stt_device = stt_device
        self._tts_device = tts_device
        
        # Подсистемы (инициализируем при start)
        self.stt: Optional[WhisperSTT] = None
        self.tts: Optional[TTS] = None
        
        # Статистика
        self._listen_count = 0
        self._speak_count = 0
        self._error_count = 0
    
    async def _start_internal(self) -> None:
        """Запуск STT и TTS"""
        # 1. Определяем устройства
        self._stt_device, self._tts_device = self._detect_devices()
        
        # 2. Инициализация STT
        try:
            cfg = load_config()
            stt_cfg = cfg.get("stt", {})
            
            self.stt = WhisperSTT(
                model_size=stt_cfg.get("model_size", "small"),
                device=self._stt_device,
                compute_type=stt_cfg.get("compute_type", "float16"),
                beam_size=stt_cfg.get("beam_size", 1),
                sample_rate=stt_cfg.get("sample_rate", 16000),
                chunk_sec=stt_cfg.get("chunk_sec", 0.5),
                vad_backend=stt_cfg.get("vad_backend", "silero_onnx"),
                silero_model_path=stt_cfg.get("silero_model_path"),
                speech_threshold=stt_cfg.get("speech_threshold", 0.55),
                silence_threshold=stt_cfg.get("silence_threshold", 0.45),
                min_speech_ms=stt_cfg.get("min_speech_ms", 200),
                min_silence_ms=stt_cfg.get("min_silence_ms", 500),
            )
            self.logger.info(f"✅ STT инициализирован (device={self._stt_device})")
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации STT: {e}", exc_info=True)
            raise
        
        # 3. Инициализация TTS
        try:
            self.tts = TTS(
                model=self.config.tts_model,
                speaker="female",
                style="soft",
                language="ru",
                device=self._tts_device,
                config=self.config,
            )
            self.logger.info(f"✅ TTS инициализирован (device={self._tts_device})")
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации TTS: {e}", exc_info=True)
            raise
        
        self.logger.info("🎵 MasterAudio полностью инициализирован")
    
    async def _stop_internal(self) -> None:
        """Закрытие STT и TTS"""
        # Статистика
        self.logger.info(
            f"📊 Статистика аудио: listen={self._listen_count}, "
            f"speak={self._speak_count}, errors={self._error_count}"
        )
        
        # Останавливаем TTS
        if self.tts:
            try:
                await self.tts.stop()
                await self.tts.aclose()
                self.logger.info("✅ TTS закрыт")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка закрытия TTS: {e}")
        
        # STT не требует явного закрытия (но на всякий случай)
        if self.stt and hasattr(self.stt, 'stop'):
            try:
                await self.stt.stop()
                self.logger.info("✅ STT закрыт")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка закрытия STT: {e}")
    
    def _detect_devices(self) -> tuple[str, str]:
        """Автоопределение устройств для STT/TTS"""
        # Если устройства заданы явно — используем их
        if self._stt_device and self._tts_device:
            self.logger.info(
                f"Устройства заданы явно: STT={self._stt_device}, TTS={self._tts_device}"
            )
            return self._stt_device, self._tts_device
        
        # Автодетект
        try:
            import torch
            cfg = load_config()
            devices_cfg = cfg.get("devices", {})
            
            stt_device = devices_cfg.get("stt") or devices_cfg.get("audio")
            tts_device = devices_cfg.get("tts") or devices_cfg.get("audio")
            
            if not stt_device or not tts_device:
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    self.logger.info(f"🎮 Обнаружено GPU: {gpu_count} шт.")
                    stt_device = stt_device or "cuda:0"
                    tts_device = tts_device or ("cuda:1" if gpu_count > 1 else "cuda:0")
                else:
                    self.logger.warning("⚠️ CUDA недоступна, используем CPU")
                    stt_device = "cpu"
                    tts_device = "cpu"
            
            return stt_device, tts_device
            
        except ImportError:
            self.logger.warning("⚠️ PyTorch не установлен, используем CPU")
            return "cpu", "cpu"
    
    async def health_check(self) -> bool:
        """Проверка здоровья аудио-подсистем"""
        if not self._running:
            return False
        
        checks = {
            "stt": self.stt is not None,
            "tts": self.tts is not None,
        }
        
        all_ok = all(checks.values())
        if not all_ok:
            self.logger.warning(f"⚠️ Health check failed: {checks}")
        
        return all_ok
    
    # ==================== API: STT ====================
    
    async def listen(self, timeout: float = 30.0) -> str:
        """
        Слушать пользователя с timeout.
        
        Args:
            timeout: максимальное время ожидания речи (сек)
        
        Returns:
            Распознанный текст (или пустая строка при timeout)
        
        Raises:
            RuntimeError: если MasterAudio не запущен
        """
        if not self.stt:
            raise RuntimeError("MasterAudio не запущен (STT недоступен)")
        
        try:
            text = await asyncio.wait_for(
                self.stt.listen(),
                timeout=timeout
            )
            
            if text and text.strip():
                self._listen_count += 1
                self.logger.debug(f"🎤 Распознано: {text[:50]}...")
                return text.strip()
            else:
                return ""
                
        except asyncio.TimeoutError:
            self.logger.debug(f"⏱️ STT timeout ({timeout}s) — пользователь молчит")
            return ""
        
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка распознавания речи: {e}")
            raise
    
    # ==================== API: TTS ====================
    
    async def speak(
        self,
        text: str,
        emotion: Optional[str] = None,
        interrupt: bool = True
    ) -> None:
        """
        Озвучить текст с заданной эмоцией.
        
        Args:
            text: текст для озвучивания
            emotion: эмоция (happy/sad/angry/neutral/...)
            interrupt: прервать текущее воспроизведение
        
        Raises:
            RuntimeError: если MasterAudio не запущен
        """
        if not self.tts:
            raise RuntimeError("MasterAudio не запущен (TTS недоступен)")
        
        if not text or not text.strip():
            self.logger.warning("⚠️ Попытка озвучить пустой текст")
            return
        
        try:
            await self.tts.speak(text, emotion=emotion, interrupt=interrupt)
            self._speak_count += 1
            self.logger.debug(f"🔊 Озвучено: {text[:50]}... [{emotion or 'neutral'}]")
        
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка синтеза речи: {e}")
            raise
    
    async def stop_speaking(self) -> None:
        """Остановить текущее воспроизведение TTS"""
        if self.tts:
            try:
                await self.tts.stop()
                self.logger.debug("🛑 TTS остановлен")
            except Exception as e:
                self.logger.error(f"❌ Ошибка остановки TTS: {e}")
    
    def is_speaking(self) -> bool:
        """Проверить, идёт ли воспроизведение TTS"""
        if self.tts and hasattr(self.tts, '_playing'):
            return self.tts._playing.is_set()
        return False
    
    # ==================== УТИЛИТЫ ====================
    
    def get_stats(self) -> dict:
        """Получить статистику работы аудио-подсистем"""
        return {
            "running": self._running,
            "stt_device": self._stt_device,
            "tts_device": self._tts_device,
            "listen_count": self._listen_count,
            "speak_count": self._speak_count,
            "error_count": self._error_count,
            "is_speaking": self.is_speaking(),
        }
    
    def get_devices(self) -> dict:
        """Получить информацию об используемых устройствах"""
        return {
            "stt": self._stt_device,
            "tts": self._tts_device,
        }
