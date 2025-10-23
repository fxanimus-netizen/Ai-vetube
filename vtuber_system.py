"""
system/vtuber_system_adaptive.py — ядро VTuber AI с адаптацией личности (всё в одном файле)
Базируется на твоём актуальном vtuber_system.py, добавлены:
- встроенный класс AdaptivePersonality (мягкое самообучение без изменения весов)
- инициализация self.adaptive в __init__
- вызов await self.adaptive.analyze_and_update(user_text, reply) в цикле диалога
- сохранены CUDA-настройки и чтение config.json (devices: llm/cuda:0, stt/tts → cuda:1)
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import re
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import sounddevice as sd  # для перехвата sd.PortAudioError

# --- Импорты подсистем (как в твоём проекте) ---
from core.config import VTuberConfig, load_config
from core.mood import MoodManager
from core.memory import HybridMemory
from avatar.personalization import (
    PersonalizationManager,
    apply_personalized_prompt,
    log_after_dialog,
)
from llm.router import HybridOllamaRouter
from llm.ollama_client import OptimizedOllamaClient
from audio.tts import TTS
# prefer local upgraded STT, fallback to original
try:
    from stt import WhisperSTT  # upgraded async VAD+ASR
except Exception:
    from audio.stt import WhisperSTT
from avatar.osc import MultiTargetOSCController

# --- Импорт эмоций (fallback если модуль недоступен) ---
try:
    from avatar.emotion import EmotionType, BLENDMAP
except Exception:
    class EmotionType:
        HAPPY = type("E", (), {"value": "Joy"})
        SAD = type("E", (), {"value": "Sad"})
        ANGRY = type("E", (), {"value": "Angry"})
        SURPRISED = type("E", (), {"value": "Surprised"})
        NEUTRAL = type("E", (), {"value": "Neutral"})
    BLENDMAP = {
        EmotionType.HAPPY: "Joy",
        EmotionType.SAD: "Sad",
        EmotionType.ANGRY: "Angry",
        EmotionType.SURPRISED: "Surprised",
        EmotionType.NEUTRAL: "Neutral",
    }

# --- Настройка логирования ---
logger = logging.getLogger("VTuberSystem")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ======================================================================
#           ВСТРОЕННЫЙ МОДУЛЬ МЯГКОЙ АДАПТАЦИИ ЛИЧНОСТИ (RU)
# ======================================================================
class AdaptivePersonality:
    """
    Мягкая адаптация личности AI без обучения весов модели.
    Анализирует последнюю реплику пользователя и ответ AI,
    обновляет профиль (tone/mood/response_style) в PersonalizationManager.
    """

    def __init__(self, personalization: PersonalizationManager):
        self.personalization = personalization

    async def analyze_and_update(self, user_text: str, model_reply: str) -> None:
        mood = self._detect_mood(user_text, model_reply)
        tone = self._detect_tone(user_text, model_reply)
        style = self._detect_style(user_text, model_reply)

        changed = False
        profile = getattr(self.personalization, "profile", {}) or {}

        if mood and profile.get("mood") != mood:
            profile["mood"] = mood
            changed = True

        if tone and profile.get("tone") != tone:
            profile["tone"] = tone
            changed = True

        if style and profile.get("response_style") != style:
            profile["response_style"] = style
            changed = True

        if changed:
            profile["last_update"] = datetime.now().isoformat(timespec="seconds")
            # сохраняем через персонализацию (у тебя уже есть метод сохранения)
            try:
                self.personalization.profile = profile
                if hasattr(self.personalization, "save_profile"):
                    self.personalization.save_profile()
            except Exception as e:
                logger.warning(f"Не удалось сохранить персонализацию: {e}")
            else:
                logger.info(f"🧠 Personality adapted → mood={profile.get('mood')}, tone={profile.get('tone')}, style={profile.get('response_style')}")

    # --- Простейшие эвристики на ключевых словах (можно заменить на тональный анализ) ---
    def _detect_mood(self, user_text: str, reply: str) -> Optional[str]:
        text = f"{user_text} {reply}".lower()
        if re.search(r"(плохо|грустн|одинок|устал|тоска|печаль|сложно)", text):
            return "supportive"      # поддерживающий
        if re.search(r"(весел|ура|смешн|хаха|рад|класс|супер)", text):
            return "cheerful"        # весёлый
        if re.search(r"(злюсь|раздраж|бесит|ненавижу|злой)", text):
            return "calm"            # спокойный (успокаивающий)
        if re.search(r"(люблю|спасибо|благодарю|благодарен)", text):
            return "empathetic"      # эмпатичный
        return None

    def _detect_tone(self, user_text: str, reply: str) -> Optional[str]:
        text = f"{user_text} {reply}".lower()
        if re.search(r"(коротко|по делу|без воды|не растягивай)", text):
            return "concise"
        if re.search(r"(расскажи подробнее|объясни|поясни|развернуто)", text):
            return "detailed"
        if re.search(r"(шути|анекдот|смешн|игриво)", text):
            return "playful"
        if re.search(r"(медленно|спокойно|тише|не спеши)", text):
            return "calm"
        return None

    def _detect_style(self, user_text: str, reply: str) -> Optional[str]:
        text = f"{user_text} {reply}".lower()
        if re.search(r"(списком|буллеты|буллетами|по пунктам)", text):
            return "bulleted"
        if re.search(r"(пример|примером|аналогия|сравнение)", text):
            return "with_examples"
        return None


# ======================================================================
#                       ОСНОВНОЙ КЛАСС СИСТЕМЫ
# ======================================================================
class RealtimeVTuberSystem:
    def __init__(
        self,
        config: Optional[VTuberConfig] = None,
        enable_unity: bool = True,
        install_signal_handlers: bool = True,
    ):
        """Основной контроллер VTuber-системы.
        install_signal_handlers — если False, не будет ловить Ctrl+C (для Unity, тестов и Jupyter).
        """
        # --- Автоотключение сигналов, если среда не терминальная ---
        if not sys.stdout.isatty():
            install_signal_handlers = False

        self.config = config or VTuberConfig()
        self.mood = MoodManager()
        self.memory: Optional[HybridMemory] = None
        self.personalization = PersonalizationManager()
        self.install_signal_handlers = install_signal_handlers

        # ✅ добавлено: инициализация адаптивного поведения
        self.adaptive = AdaptivePersonality(self.personalization)

        # --- Определение устройств для STT/TTS (через config.json с fallback) ---
        self.stt_device = "cpu"
        self.tts_device = "cpu"
        try:
            import torch
            try:
                # читаем конфиг приоритетно (если задано)
                cfg = load_config()
                devices_cfg = cfg.get("devices", {})
            except Exception:
                devices_cfg = {}

            stt_device = devices_cfg.get("stt") or devices_cfg.get("audio")
            tts_device = devices_cfg.get("tts") or devices_cfg.get("audio")

            # если в конфиге не указано — автодетект
            if not stt_device or not tts_device:
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    logger.info(f"Обнаружено GPU: {gpu_count} шт.")
                    stt_device = stt_device or "cuda:0"
                    tts_device = tts_device or ("cuda:1" if gpu_count > 1 else "cuda:0")
                else:
                    logger.warning("CUDA недоступна, используем CPU")
                    stt_device = "cpu"
                    tts_device = "cpu"
            else:
                logger.info(f"Задано в конфиге: STT={stt_device}, TTS={tts_device}")
        except ImportError:
            logger.warning("PyTorch не установлен, используем CPU")
            stt_device = "cpu"
            tts_device = "cpu"

        self.stt_device = stt_device
        self.tts_device = tts_device

        # --- Мозг (LLM роутер) ---
        try:
            ollama_client = OptimizedOllamaClient()
            self.ollama_client = ollama_client
            self.router = HybridOllamaRouter(
                ollama=ollama_client,
                fast_model=self.config.fast_model,
                smart_model=self.config.smart_model,
            )
            logger.info(f"Router готов: fast={self.config.fast_model}, smart={self.config.smart_model}")
        except Exception as e:
            logger.error(f"Ошибка инициализации LLM роутера: {e}", exc_info=True)
            raise RuntimeError("Не удалось инициализировать LLM систему") from e

        # --- Аудио ---
        try:
            self.tts = TTS(
                model="cosyvoice-2",
                speaker="female",
                style="soft",
                language="ru",
                device=self.tts_device,
            )
            cfg = {}
            try:
                cfg = load_config()
            except Exception:
                cfg = {}
            stt_cfg = cfg.get("stt", {})
            self.stt = WhisperSTT(
                model_size=stt_cfg.get("model_size", "small"),
                device=self.stt_device,
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
        except Exception as e:
            logger.error(f"Ошибка инициализации аудио модулей: {e}", exc_info=True)
            raise RuntimeError("Не удалось инициализировать аудио подсистему") from e

        # --- Аватар (OSC) ---
        self.avatar = MultiTargetOSCController(enable_unity=enable_unity)

        # --- Служебные флаги ---
        self._running = False
        self._stop_lock = asyncio.Lock()
        self._signal_handlers_installed = False

        logger.info(f"✅ RealtimeVTuberSystem инициализирована (STT: {self.stt_device}, TTS: {self.tts_device})")

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            logger.warning("VTuber уже запущен.")
            return

        self.memory = HybridMemory()
        await self.memory.aopen()
        self._running = True

        # --- Установка обработчиков сигналов ---
        if self.install_signal_handlers and not self._signal_handlers_installed:
            loop = asyncio.get_running_loop()

            def _graceful_stop(sig: signal.Signals):
                logger.warning(f"Получен сигнал {sig.name} — начинаем корректное завершение...")
                asyncio.create_task(self.stop())

            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError):
                    loop.add_signal_handler(sig, functools.partial(_graceful_stop, sig))

            self._signal_handlers_installed = True

        logger.info("VTuber готов к работе (нажми Ctrl+C для выхода).")

    # ------------------------------------------------------------------
    async def stop(self) -> None:
        async with self._stop_lock:
            if not self._running:
                return
            self._running = False

            logger.info("Останавливаем VTuber...")

            with contextlib.suppress(Exception):
                await self.tts.stop()
            with contextlib.suppress(Exception):
                await self.avatar.shutdown()
            with contextlib.suppress(Exception):
                if self.memory:
                    await self.memory.aclose()
            with contextlib.suppress(Exception):
                if getattr(self, 'ollama_client', None):
                    await self.ollama_client.close()

            logger.info("VTuber корректно завершён.")

    # ------------------------------------------------------------------
    async def run_dialogue(self) -> None:
        if not self._running:
            logger.error("Система не запущена. Сначала вызови await start().")
            return

        error_count = 0

        while self._running:
            try:
                user_text = await self.stt.listen()
                if not user_text:
                    await asyncio.sleep(0.2)
                    continue
                await self.memory.add_turn("user", user_text)

                context = await self.memory.context(user_text)

                base_prompt = "Ты — виртуальный VTuber-компаньон. Общайся естественно и поддерживай контакт."
                personalized_prompt = await apply_personalized_prompt(base_prompt, username="guest", platform="voice")

                reply, emotion_name = await self.router.generate_reply(
                    user_text, context=context, system_prompt=personalized_prompt
                )

                # ✅ адаптация личности на основе диалога (до TTS)
                await self.adaptive.analyze_and_update(user_text, reply)

                await self.memory.add_turn("assistant", reply)

                await self.tts.speak(reply, emotion=emotion_name)
                await log_after_dialog("guest", user_text, reply, emotion_name)

                if hasattr(self.avatar, "set_emotion"):
                    await self.avatar.set_emotion(emotion_name or "neutral")

                error_count = 0

            except asyncio.CancelledError:
                break
            except sd.PortAudioError as e:
                logger.error(f"Аудио-ошибка: {e}")
                error_count += 1
            except Exception as e:
                logger.exception(f"Ошибка в цикле диалога: {e}")
                error_count += 1

            if error_count > 3:
                logger.warning("⚠️ WhisperSTT временно недоступен, повтор через 3 секунды")
                await asyncio.sleep(3)
                error_count = 0

        logger.info("Выход из цикла диалога.")


# ======================================================================
# Точка входа (ручной запуск)
# ======================================================================
if __name__ == "__main__":
    async def main():
        vtuber = RealtimeVTuberSystem()
        await vtuber.start()
        try:
            await vtuber.run_dialogue()
        finally:
            await vtuber.stop()

    asyncio.run(main())
