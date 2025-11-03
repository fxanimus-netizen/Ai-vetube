# -*- coding: utf-8 -*-
"""
masters/avatar_master.py — Аватар-мастер VTuber системы

Управляет:
- OSC коммуникацией (VSeeFace, Unity, Luppet)
- Эмоциями и BlendShapes
- Позами и костями
- Lip-sync сигналами

Версия: 1.0 (2025-11-03)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

from .base import BaseMaster

# Импорты аватар-модулей
from avatar.osc import MultiTargetOSCController

# Импорты эмоций
try:
    from avatar.emotion import BLENDMAP, EMOTIONS
except ImportError:
    BLENDMAP = {}
    EMOTIONS = ["happy", "sad", "angry", "surprised", "neutral"]

# Импорты конфигурации
from core.config import VTuberConfig

logger = logging.getLogger("MasterAvatar")


class MasterAvatar(BaseMaster):
    """
    Аватар-мастер — управление OSC и визуализацией эмоций.
    
    Возможности:
    - Отправка эмоций в аватар (VSeeFace/Unity/Luppet)
    - Управление позами и костями
    - Lip-sync сигналы (говорит/молчит)
    - Pulse-эффекты для эмоций
    
    API:
    - set_emotion() — установить эмоцию
    - pulse_emotion() — короткая вспышка эмоции
    - set_pose() — установить позу кости
    - speak_signal() — сигнал начала/конца речи
    """
    
    def __init__(
        self,
        config: Optional[VTuberConfig] = None,
        enable_unity: bool = True,
        enable_luppet_midi: bool = True,
    ):
        super().__init__("Avatar")
        
        # Конфигурация
        self.config = config or VTuberConfig.load()
        self._enable_unity = enable_unity
        self._enable_luppet_midi = enable_luppet_midi
        
        # Подсистемы (инициализируем при start)
        self.osc: Optional[MultiTargetOSCController] = None
        
        # Статистика
        self._emotion_changes = 0
        self._pose_updates = 0
        self._lipsync_signals = 0
        self._error_count = 0
        
        # Текущая эмоция (для отслеживания)
        self._current_emotion = "neutral"
    
    async def _start_internal(self) -> None:
        """Запуск OSC контроллера"""
        try:
            # Получаем порты из конфига
            luppet_port = getattr(self.config, "osc_luppet_port", 39539)
            vseeface_port = getattr(self.config, "osc_vseeface_port", 39540)
            unity_port = getattr(self.config, "osc_unity_port", 39541)
            
            # Инициализация OSC
            self.osc = MultiTargetOSCController(
                host="127.0.0.1",
                luppet_port=luppet_port,
                vseeface_port=vseeface_port,
                unity_port=unity_port,
                enable_unity=self._enable_unity,
                enable_luppet_midi=self._enable_luppet_midi,
            )
            
            self.logger.info(
                f"✅ OSC контроллер инициализирован: "
                f"Luppet={luppet_port}, VSeeFace={vseeface_port}, Unity={unity_port}"
            )
            
            # Устанавливаем нейтральную эмоцию по умолчанию
            await self.set_emotion("neutral", value=1.0)
            
            self.logger.info("🎭 MasterAvatar полностью инициализирован")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации аватара: {e}", exc_info=True)
            raise
    
    async def _stop_internal(self) -> None:
        """Закрытие OSC контроллера"""
        # Статистика
        self.logger.info(
            f"📊 Статистика аватара: emotions={self._emotion_changes}, "
            f"poses={self._pose_updates}, lipsync={self._lipsync_signals}, "
            f"errors={self._error_count}"
        )
        
        # Сбрасываем все эмоции перед закрытием
        if self.osc:
            try:
                await self.set_emotion("neutral", value=1.0)
                await asyncio.sleep(0.1)
            except Exception:
                pass
        
        # Закрываем OSC
        if self.osc:
            try:
                await self.osc.shutdown()
                self.logger.info("✅ OSC контроллер закрыт")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка закрытия OSC: {e}")
    
    async def health_check(self) -> bool:
        """Проверка здоровья аватар-подсистем"""
        if not self._running:
            return False
        
        checks = {
            "osc": self.osc is not None,
        }
        
        all_ok = all(checks.values())
        if not all_ok:
            self.logger.warning(f"⚠️ Health check failed: {checks}")
        
        return all_ok
    
    # ==================== API: ЭМОЦИИ ====================
    
    async def set_emotion(self, emotion: str, value: float = 1.0) -> None:
        """
        Установить эмоцию аватара.
        
        Args:
            emotion: название эмоции (happy/sad/angry/surprised/neutral)
            value: интенсивность (0.0 - 1.0)
        
        Raises:
            RuntimeError: если MasterAvatar не запущен
        """
        if not self.osc:
            raise RuntimeError("MasterAvatar не запущен (OSC недоступен)")
        
        # Нормализуем название эмоции
        emotion = emotion.lower().strip()
        
        # Проверяем валидность
        if emotion not in EMOTIONS and emotion not in BLENDMAP:
            self.logger.warning(f"⚠️ Неизвестная эмоция '{emotion}', используем neutral")
            emotion = "neutral"
        
        # Ограничиваем value
        value = max(0.0, min(1.0, float(value)))
        
        try:
            # Отправляем в OSC
            await asyncio.to_thread(self.osc.send_emotion, emotion, value)
            
            self._emotion_changes += 1
            self._current_emotion = emotion
            
            self.logger.debug(f"😊 Эмоция установлена: {emotion} ({value:.2f})")
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка установки эмоции: {e}")
            # Не пробрасываем ошибку — аватар не критичен
    
    async def pulse_emotion(
        self,
        emotion: str,
        intensity: float = 1.0,
        duration: float = 1.2
    ) -> None:
        """
        Короткая вспышка эмоции (pulse-эффект).
        
        Args:
            emotion: название эмоции
            intensity: интенсивность
            duration: длительность (сек)
        
        Raises:
            RuntimeError: если MasterAvatar не запущен
        """
        if not self.osc:
            raise RuntimeError("MasterAvatar не запущен (OSC недоступен)")
        
        try:
            # Используем async-версию pulse
            await self.osc.pulse_emotion_async(emotion, intensity, duration)
            
            self._emotion_changes += 1
            self.logger.debug(
                f"✨ Pulse эмоции: {emotion} ({intensity:.2f}, {duration}s)"
            )
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка pulse эмоции: {e}")
    
    def get_current_emotion(self) -> str:
        """Получить текущую эмоцию"""
        return self._current_emotion
    
    # ==================== API: ПОЗЫ ====================
    
    async def set_pose(
        self,
        bone: str,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float]
    ) -> None:
        """
        Установить позу кости.
        
        Args:
            bone: название кости (Head, LeftHand, etc.)
            position: позиция (x, y, z)
            rotation: поворот (rx, ry, rz)
        
        Raises:
            RuntimeError: если MasterAvatar не запущен
        """
        if not self.osc:
            raise RuntimeError("MasterAvatar не запущен (OSC недоступен)")
        
        try:
            # Отправляем позу
            await asyncio.to_thread(
                self.osc.send_pose,
                bone,
                position,
                rotation
            )
            
            self._pose_updates += 1
            self.logger.debug(f"🦴 Поза обновлена: {bone}")
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка установки позы: {e}")
    
    # ==================== API: LIP-SYNC ====================
    
    async def speak_signal(self, active: bool = True) -> None:
        """
        Сигнал начала/конца речи (для lip-sync).
        
        Args:
            active: True - начало речи, False - конец речи
        
        Raises:
            RuntimeError: если MasterAvatar не запущен
        """
        if not self.osc:
            raise RuntimeError("MasterAvatar не запущен (OSC недоступен)")
        
        try:
            # Отправляем сигнал
            await asyncio.to_thread(self.osc.speak_signal, active)
            
            self._lipsync_signals += 1
            self.logger.debug(f"🗣️ Lip-sync: {'ON' if active else 'OFF'}")
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка сигнала lip-sync: {e}")
    
    # ==================== УТИЛИТЫ ====================
    
    def get_stats(self) -> dict:
        """Получить статистику работы аватара"""
        return {
            "running": self._running,
            "current_emotion": self._current_emotion,
            "emotion_changes": self._emotion_changes,
            "pose_updates": self._pose_updates,
            "lipsync_signals": self._lipsync_signals,
            "error_count": self._error_count,
            "unity_enabled": self._enable_unity,
            "luppet_midi_enabled": self._enable_luppet_midi,
        }
    
    def get_supported_emotions(self) -> list:
        """Получить список поддерживаемых эмоций"""
        return list(EMOTIONS)
