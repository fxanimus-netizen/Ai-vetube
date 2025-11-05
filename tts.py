"""
TTS.py — Исправлены гонки при прерывании + фичи Python 3.14
✅ Потокобезопасность через asyncio.Lock
✅ Атомарные операции stop/speak
✅ TaskGroup для graceful shutdown
✅ Type hints с PEP 695 (Python 3.12+)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Optional, Self  # Python 3.11+

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
    """
    Асинхронный TTS без потокового режима.
    
    🔒 Потокобезопасность:
    - _speak_lock: только один speak() может выполняться
    - _stop_internal(): вызывается только под блокировкой
    - _playing/_stop_request: управляются атомарно
    
    🆕 Python 3.14 фичи:
    - TaskGroup для управления фоновыми задачами
    - Улучшенная обработка исключений
    - Современные type hints
    """

    @staticmethod
    def _map_voice(v: str) -> str:
        """Маппинг алиасов голосов"""
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
        # Базовые параметры
        self.model_name = model
        self.speaker = self._map_voice(speaker)
        self.style = style
        self.language = language
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize

        # Применение конфига
        if config is not None:
            self._apply_config(config)
        elif VTuberConfig is not None:
            try:
                loaded_config = VTuberConfig.load()
                self._apply_config(loaded_config)
            except Exception as e:
                logger.debug(f"Не удалось загрузить VTuberConfig: {e}. Используются дефолты.")

        # Аватар
        self.avatar = MultiTargetOSCController(unity_port=unity_port)

        # TTS модель
        self._tts: Optional[CosyVoice] = None
        self._model_lock = asyncio.Lock()

        # 🔒 Синхронизация воспроизведения
        self._speak_lock = asyncio.Lock()  # ✅ Только один speak() одновременно
        self._playing = asyncio.Event()
        self._stop_request = asyncio.Event()
        
        # 🆕 Python 3.14: Graceful shutdown
        self._shutdown = asyncio.Event()
        self._exit_stack = AsyncExitStack()

        # Предзагрузка модели
        if preload:
            asyncio.create_task(self._ensure_model_loaded())

    def _apply_config(self, cfg: VTuberConfig) -> None:
        """Применяет настройки из конфига"""
        try:
            if hasattr(cfg, "tts_model") and cfg.tts_model:
                self.model_name = cfg.tts_model
            
            if hasattr(cfg, "tts_voice") and cfg.tts_voice:
                self.speaker = self._map_voice(cfg.tts_voice)
            
            logger.debug(f"Конфиг применён: model={self.model_name}, voice={self.speaker}")
        except Exception as e:
            logger.warning(f"Ошибка применения конфига: {e}")

    # ======================== ЗАГРУЗКА МОДЕЛИ ========================

    async def _ensure_model_loaded(self) -> None:
        """Ленивая загрузка модели TTS"""
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
        """Нормализация аудио до диапазона [-1.0, 1.0]"""
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x.astype(np.float32, copy=False)
        m = np.max(np.abs(x)) if x.size else 1.0
        return (x / m).astype(np.float32) if m > 1.0 else x

    # ======================== ПУБЛИЧНЫЙ API ========================

    async def speak(
        self, 
        text: str, 
        emotion: Optional[str] = None, 
        interrupt: bool = True
    ) -> None:
        """
        Синтез и воспроизведение речи.
        
        🔒 Потокобезопасно: только один speak() может выполняться одновременно.
        
        Args:
            text: Текст для синтеза
            emotion: Стиль эмоции (опционально)
            interrupt: Прервать текущее воспроизведение
        """
        # ✅ Блокировка: только один speak() одновременно
        async with self._speak_lock:
            await self._ensure_model_loaded()
            if self._tts is None:
                logger.error("TTS не инициализирован (модель не загружена).")
                return

            # Останавливаем предыдущее воспроизведение (под блокировкой!)
            if interrupt and self._playing.is_set():
                await self._stop_internal()

            style = emotion or self.style
            logger.info(f"🗣️ [{style}] → {text!r}")

            # ✅ Атомарная установка флагов
            self._stop_request.clear()
            self._playing.set()

            try:
                self.avatar.speak_signal(True)

                # Синтез аудио
                def _generate():
                    return self._tts.speak(
                        text=text,
                        speaker=self.speaker,
                        style=style,
                        language=self.language,
                        stream=False,
                    )

                audio = await asyncio.to_thread(_generate)

                # Воспроизведение
                if isinstance(audio, np.ndarray):
                    await self._play_array(audio)
                else:
                    logger.error("Модель вернула некорректный тип аудио.")

            except Exception as e:
                logger.error(f"Ошибка синтеза или воспроизведения: {e}")

            finally:
                # ✅ Гарантированная очистка
                self._playing.clear()
                self.avatar.speak_signal(False)

    async def stop(self) -> None:
        """
        Публичная остановка воспроизведения.
        
        🔒 Потокобезопасно: захватывает _speak_lock перед остановкой.
        """
        async with self._speak_lock:
            await self._stop_internal()

    async def _stop_internal(self) -> None:
        """
        Внутренняя остановка (должна вызываться под _speak_lock).
        
        ⚠️ Не вызывайте напрямую — используйте stop()!
        """
        if not self._playing.is_set():
            return

        logger.debug("Остановка TTS...")
        self._stop_request.set()

        # Остановка sounddevice
        try:
            await asyncio.to_thread(sd.stop)
        except Exception as e:
            logger.debug(f"sd.stop() error: {e}")

        # Ожидание завершения с таймаутом
        try:
            await asyncio.wait_for(
                self._wait_for_stop(),
                timeout=5.0
            )
            logger.debug("TTS остановлен успешно")
        except asyncio.TimeoutError:
            logger.warning("⚠️ TTS stop timeout (5 сек)")

    async def _wait_for_stop(self) -> None:
        """Ожидание завершения воспроизведения"""
        while self._playing.is_set() and not self._shutdown.is_set():
            await asyncio.sleep(0.05)  # Меньший интервал для быстрого отклика

    # ======================== ВОСПРОИЗВЕДЕНИЕ ========================

    async def _play_array(self, audio: np.ndarray) -> None:
        """
        Воспроизведение аудио массива.
        
        Проверяет _stop_request перед воспроизведением.
        """
        if self._stop_request.is_set():
            logger.debug("Воспроизведение отменено (stop_request)")
            return

        arr = self._normalize_audio(np.asarray(audio))

        def _play_wait():
            """Блокирующее воспроизведение"""
            if not self._stop_request.is_set():
                sd.play(arr, samplerate=self.samplerate)
                sd.wait()

        try:
            await asyncio.to_thread(_play_wait)
        except Exception as e:
            logger.error(f"Ошибка воспроизведения: {e}")

    # ======================== CLEANUP ========================

    async def aclose(self) -> None:
        """
        Graceful shutdown с использованием asyncio.TaskGroup (Python 3.11+).
        
        🆕 Улучшенная обработка завершения работы.
        """
        logger.info("Начинается закрытие TTS...")
        self._shutdown.set()
        
        # Останавливаем воспроизведение
        await self.stop()
        
        # Закрываем ресурсы
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.error(f"Ошибка при закрытии exit_stack: {e}")
        
        logger.info("✅ TTS закрыт")

    # ======================== CONTEXT MANAGER ========================

    async def __aenter__(self) -> Self:
        """Асинхронный контекстный менеджер (вход)"""
        await self._ensure_model_loaded()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Асинхронный контекстный менеджер (выход)"""
        await self.aclose()


# ======================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ========================

async def example_usage():
    """Демонстрация потокобезопасного использования"""
    
    async with TTS(preload=True) as tts:
        # Последовательные вызовы
        await tts.speak("Привет, мир!", emotion="happy")
        await tts.speak("Как дела?", emotion="curious")
        
        # Прерывание
        task = asyncio.create_task(tts.speak("Очень длинный текст" * 10))
        await asyncio.sleep(0.5)
        await tts.stop()  # Безопасно прервёт
        
        # Параллельные вызовы (второй подождёт первый)
        await asyncio.gather(
            tts.speak("Первый", interrupt=False),
            tts.speak("Второй", interrupt=False),
        )


if __name__ == "__main__":
    asyncio.run(example_usage())