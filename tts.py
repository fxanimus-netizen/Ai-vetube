"""
TTS.py — Исправлен конфликт переменных (простая версия)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Optional

import numpy as np
import sounddevice as sd

try:
    from cosyvoice import CosyVoice
except Exception:
    CosyVoice = None

from avatar.osc import MultiTargetOSCController

try:
    from core.config import VTuberConfig
except Exception:
    VTuberConfig = None

logger = logging.getLogger("AudioTTS")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class TTS:
    """Асинхронный TTS без потокового режима"""

    @staticmethod
    def _map_voice(v: str) -> str:
        try:
            key = v.lower()
        except Exception:
            return v
        female_aliases = {"alloy", "sakura", "female", "woman", "f", "alto"}
        male_aliases = {"onyx", "male", "man", "m", "baritone", "bass"}
        if key in female_aliases:
            return "female"
        if key in male_aliases:
            return "male"
        return v

    def __init__(
        self,
        model: str = "cosyvoice-2",
        speaker: str = "female",
        style: str = "soft",
        language: str = "ru",
        device: str = "cuda:0",
        unity_port: int = 39541,
        samplerate: int = 24000,
        blocksize: int = 2048,
        preload: bool = True,
        config: VTuberConfig | None = None,
    ) -> None:
        # Базовые параметры (дефолты из аргументов)
        self.model_name = model
        self.speaker = self._map_voice(speaker)
        self.style = style
        self.language = language
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize

        # ========================================
        # 🔧 ИСПРАВЛЕНО: Упрощённая логика конфига
        # ========================================
        
        # Шаг 1: Если передан конфиг явно — используем его
        if config is not None:
            self._apply_config(config)
        
        # Шаг 2: Если конфиг НЕ передан, но класс VTuberConfig доступен — пробуем загрузить
        elif VTuberConfig is not None:
            try:
                loaded_config = VTuberConfig.load()
                self._apply_config(loaded_config)
            except Exception as e:
                logger.debug(f"Не удалось загрузить VTuberConfig: {e}. Используются дефолты.")
        
        # Шаг 3: Если ничего не получилось — просто используем дефолты из аргументов
        # (они уже установлены выше)

        # Остальная инициализация
        self.avatar = MultiTargetOSCController(unity_port=unity_port)

        self._tts: Optional[CosyVoice] = None
        self._playing = asyncio.Event()
        self._stop_request = asyncio.Event()
        self._model_lock = asyncio.Lock()
        self._exit_stack = AsyncExitStack()

        if preload:
            asyncio.get_event_loop().create_task(self._ensure_model_loaded())

    def _apply_config(self, cfg: VTuberConfig) -> None:
        """
        Применяет настройки из конфига (если они есть).
        Вынесено в отдельный метод для чистоты.
        """
        try:
            # Переопределяем только если в конфиге есть непустые значения
            if hasattr(cfg, "tts_model") and cfg.tts_model:
                self.model_name = cfg.tts_model
            
            if hasattr(cfg, "tts_voice") and cfg.tts_voice:
                self.speaker = self._map_voice(cfg.tts_voice)
            
            logger.debug(f"Конфиг применён: model={self.model_name}, voice={self.speaker}")
        except Exception as e:
            logger.warning(f"Ошибка применения конфига: {e}")

    # -------------------------- ВСПОМОГАТЕЛЬНОЕ --------------------------

    async def _ensure_model_loaded(self) -> None:
        async with self._model_lock:
            if self._tts is not None:
                return
            if CosyVoice is None:
                logger.warning("CosyVoice не установлен. TTS будет недоступен.")
                return

            logger.info(f"Загрузка модели CosyVoice: {self.model_name} ({self.device})...")

            def _load():
                return CosyVoice(model=self.model_name, device=self.device)

            try:
                self._tts = await asyncio.to_thread(_load)
                logger.info("✅ CosyVoice инициализирован")
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации CosyVoice: {e}")
                self._tts = None

    @staticmethod
    def _normalize_audio(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x.astype(np.float32, copy=False)
        m = np.max(np.abs(x)) if x.size else 1.0
        return (x / m).astype(np.float32) if m > 1.0 else x

    # ------------------------------- ПУБЛИЧНОЕ API -------------------------------

    async def speak(self, text: str, emotion: Optional[str] = None, interrupt: bool = True) -> None:
        """Синтез речи (без потокового режима)"""
        await self._ensure_model_loaded()
        if self._tts is None:
            logger.error("TTS не инициализирован (модель не загружена).")
            return

        if interrupt:
            await self.stop()

        style = emotion or self.style
        logger.info(f"🗣️ [{style}] → {text!r}")

        self._stop_request.clear()
        self._playing.set()
        self.avatar.speak_signal(True)

        try:
            def _generate():
                return self._tts.speak(
                    text=text,
                    speaker=self.speaker,
                    style=style,
                    language=self.language,
                    stream=False,
                )

            audio = await asyncio.to_thread(_generate)

            if isinstance(audio, np.ndarray):
                await self._play_array(audio)
            else:
                logger.error("Модель вернула некорректный тип аудио.")
        except Exception as e:
            logger.error(f"Ошибка синтеза или воспроизведения: {e}")
        finally:
            self._playing.clear()
            self.avatar.speak_signal(False)

    async def stop(self) -> None:
        """Остановка текущего воспроизведения с ожиданием завершения"""
        if self._playing.is_set():
            self._stop_request.set()
            try:
                await asyncio.to_thread(sd.stop)
            except Exception:
                pass
            
            # Дожидаемся, пока флаг снимется (таймаут 5 сек)
            for _ in range(50):
                if not self._playing.is_set():
                    break
                await asyncio.sleep(0.1)
            else:
                logger.warning("TTS не остановился за 5 секунд")

    async def aclose(self) -> None:
        """Закрытие и освобождение ресурсов"""
        await self.stop()
        await self._exit_stack.aclose()
        logger.info("TTS закрыт")

    # -------------------------- ВОСПРОИЗВЕДЕНИЕ --------------------------

    async def _play_array(self, audio: np.ndarray) -> None:
        arr = self._normalize_audio(np.asarray(audio))

        def _play_wait():
            sd.play(arr, samplerate=self.samplerate)
            sd.wait()

        await asyncio.to_thread(_play_wait)
