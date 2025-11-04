# -*- coding: utf-8 -*-
"""
vtuber_system.py — VTuber System v3.0 FINAL + Duplex Mode

🎉 ФИНАЛЬНАЯ ВЕРСИЯ с полнодуплексным диалогом

Все подсистемы вынесены в мастеры:
- MasterCore: данные и состояние (память, персонализация)
- MasterAudio: звук (STT/TTS)
- MasterLLM: генерация ответов (роутинг fast/smart)
- MasterAvatar: визуализация (OSC, эмоции, позы)

Новое в v3.0:
- ✅ Полнодуплексный режим (параллельная обработка STT/TTS)
- ✅ Возможность перебивать бота
- ✅ Умная система приоритетов
- ✅ Интеллектуальное прерывание

Преимущества:
- Чистый код (главный класс — только оркестрация)
- Легко тестировать (каждый мастер независим)
- Легко расширять (добавить новый мастер)
- Отказоустойчиво (если аватар упал, диалог работает)
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import platform
import signal
import sys
from typing import Optional, Dict

# Импорты мастеров
from masters import MasterCore, MasterAudio, MasterLLM, MasterAvatar

# Импорты конфигурации
from core.config import VTuberConfig

# Импорт полнодуплексного режима
from duplex_dialogue import DuplexDialogueManager, InterruptionStrategy

from memory import HybridMemory

# Для ловли ошибок аудио (опционально)
try:
    import sounddevice as sd
except ImportError:
    sd = None

logger = logging.getLogger(__name__)


class VTuberSystem:
    """
    Главный класс VTuber системы с поддержкой полнодуплексного диалога.
    
    Архитектура:
    - Masters: независимые модули (Core, Audio, LLM, Avatar)
    - DuplexDialogueManager: параллельная обработка (опционально)
    - Signal handlers: graceful shutdown
    
    Режимы работы:
    - Duplex (по умолчанию): параллельная обработка STT/TTS
    - Sequential: старый режим (для совместимости)
    """
    
    def __init__(
        self,
        config: Optional[VTuberConfig] = None,
        enable_unity: bool = True,
        enable_luppet_midi: bool = True,
        install_signal_handlers: bool = True,
        enable_duplex: bool = True,
    ):
        """
        Инициализация VTuber системы.
        
        Args:
            config: конфигурация системы (если None, загружается из config.json)
            enable_unity: включить Unity аватар
            enable_luppet_midi: включить MIDI для Luppet
            install_signal_handlers: установить обработчики Ctrl+C (для терминала)
            enable_duplex: включить полнодуплексный режим диалога (рекомендуется)
        """
        # Автоотключение сигналов для неинтерактивных окружений
        if not sys.stdout.isatty():
            install_signal_handlers = False
        
        self.config = config or VTuberConfig.load()
        self.install_signal_handlers = install_signal_handlers
        
        # ========== ИНИЦИАЛИЗАЦИЯ МАСТЕРОВ ==========
        
        # 1. Ядро (память, персонализация, настроение)
        self.core = MasterCore(config=self.config)
        
        # 2. Аудио (STT, TTS)
        self.audio = MasterAudio(config=self.config)
        
        # 3. LLM (генерация ответов, роутинг)
        self.llm = MasterLLM(config=self.config)
        
        # 4. Аватар (OSC, эмоции, позы)
        self.avatar = MasterAvatar(
            config=self.config,
            enable_unity=enable_unity,
            enable_luppet_midi=enable_luppet_midi
        )
        
        # Служебные флаги
        self._running = False
        self._stop_lock = asyncio.Lock()
        self._signal_handlers_installed = False
        
        # Полнодуплексный режим
        self._enable_duplex = enable_duplex
        self._duplex_manager: Optional[DuplexDialogueManager] = None
        
        logger.info(
            f"✅ VTuber System v3.0 FINAL создана "
            f"(config: {self.config.name}, duplex: {enable_duplex})"
        )
    
    # ------------------------------------------------------------------
    #                          ЖИЗНЕННЫЙ ЦИКЛ
    # ------------------------------------------------------------------
    
    async def start(self) -> None:
        """Запуск всех мастеров"""
        if self._running:
            logger.warning("VTuber уже запущена")
            return
        
        logger.info("🚀 Запуск VTuber System v3.0 FINAL...")
        
        try:
            # Запускаем мастеров в правильном порядке
            # (от самых независимых к самым зависимым)
            
            # 1. Core (фундамент — нужен всем)
            await self.core.start()
            
            # 2. Audio (независим от остальных)
            await self.audio.start()
            
            # 3. LLM (независим от остальных)
            await self.llm.start()
            
            # 4. Avatar (может упасть — не критично)
            try:
                await self.avatar.start()
            except Exception as e:
                logger.warning(
                    f"⚠️ Avatar недоступен (будет работать без визуализации): {e}"
                )
            
            # 5. Инициализация полнодуплексного менеджера
            if self._enable_duplex:
                self._duplex_manager = DuplexDialogueManager(
                    core=self.core,
                    audio=self.audio,
                    llm=self.llm,
                    avatar=self.avatar,
                    interruption_strategy=InterruptionStrategy.SENTENCE,
                    user_silence_timeout=30.0,
                    bot_silence_timeout=2.0,
                )
                
                # Устанавливаем колбэки (опционально)
                self._duplex_manager.on_user_message = self._on_user_message
                self._duplex_manager.on_bot_reply = self._on_bot_reply
                self._duplex_manager.on_interrupt = self._on_interrupt
                
                logger.info("✅ Полнодуплексный режим включён")
            
            # 6. Устанавливаем обработчики сигналов
            if self.install_signal_handlers and not self._signal_handlers_installed:
                self._setup_signal_handlers()
            
            self._running = True
            logger.info(
                "✅ VTuber System v3.0 FINAL готова к работе\n"
                "   📊 Статус мастеров:\n"
                f"      - Core: {'✅' if self.core.is_running() else '❌'}\n"
                f"      - Audio: {'✅' if self.audio.is_running() else '❌'}\n"
                f"      - LLM: {'✅' if self.llm.is_running() else '❌'}\n"
                f"      - Avatar: {'✅' if self.avatar.is_running() else '⚠️ (опционально)'}\n"
                f"      - Duplex: {'✅' if self._enable_duplex else '❌ (sequential mode)'}\n"
                "   🎙️ Нажми Ctrl+C для выхода"
            )
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка запуска: {e}", exc_info=True)
            await self.stop()
            raise RuntimeError("Не удалось запустить VTuber систему") from e
    
    def _setup_signal_handlers(self) -> None:
        """Установка обработчиков сигналов для graceful shutdown"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("Event loop не запущен, signal handlers не установлены")
            return
        
        def _graceful_stop(sig: signal.Signals):
            logger.warning(
                f"Получен сигнал {sig.name} — начинаем корректное завершение..."
            )
            asyncio.create_task(self.stop())
        
        if platform.system() != "Windows":
            # Unix-like systems
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError):
                    loop.add_signal_handler(
                        sig, 
                        functools.partial(_graceful_stop, sig)
                    )
        else:
            # Windows fallback
            def windows_handler(signum, frame):
                asyncio.create_task(self.stop())
            signal.signal(signal.SIGINT, windows_handler)
        
        self._signal_handlers_installed = True
        logger.debug("✅ Signal handlers установлены")
    
    async def stop(self) -> None:
        """Корректная остановка всех мастеров"""
        async with self._stop_lock:
            if not self._running:
                return
            self._running = False
            
            logger.info("🛑 Останавливаем VTuber System v3.0...")
            
            # Останавливаем duplex-менеджер (если был запущен)
            if self._duplex_manager:
                with contextlib.suppress(Exception):
                    await self._duplex_manager.stop()
            
            # Останавливаем мастеров в обратном порядке
            # (от самых зависимых к самым независимым)
            
            # 1. Avatar (может быть уже упал — не критично)
            with contextlib.suppress(Exception):
                await self.avatar.stop()
            
            # 2. LLM
            with contextlib.suppress(Exception):
                await self.llm.stop()
            
            # 3. Audio
            with contextlib.suppress(Exception):
                await self.audio.stop()
            
            # 4. Core (останавливаем последним — сохраняет данные)
            with contextlib.suppress(Exception):
                await self.core.stop()
            
            logger.info("✅ VTuber System v3.0 корректно завершена")
    
    # ------------------------------------------------------------------
    #                       ОСНОВНОЙ ЦИКЛ ДИАЛОГА
    # ------------------------------------------------------------------
    
    async def run_dialogue(self) -> None:
        """
        Основной цикл диалога с использованием всех мастеров.
        
        РЕЖИМЫ:
        - Duplex (enable_duplex=True): параллельная обработка STT/TTS
        - Sequential (enable_duplex=False): последовательная обработка
        
        ПОТОК ДАННЫХ (Duplex):
        1. STT Worker → слушает параллельно
        2. LLM Worker → обрабатывает очередь
        3. TTS Worker → говорит параллельно
        
        ПОТОК ДАННЫХ (Sequential):
        1. Audio.listen() → текст пользователя
        2. Core.add_turn() → сохранить в память
        3. Core.get_context() → получить контекст
        4. Core.get_personalized_prompt() → промпт с персонализацией
        5. LLM.generate_reply() → сгенерировать ответ + эмоцию
        6. Core.add_turn() → сохранить ответ
        7. Audio.speak() + Avatar.set_emotion() → озвучить + показать
        8. Core.adapt_personality() → адаптировать (фон)
        9. Core.update_user_interaction() → статистика (фон)
        """
        if not self._running:
            logger.error("Система не запущена. Сначала вызови await start()")
            return
        
        if self._enable_duplex and self._duplex_manager:
            # ========== ПОЛНОДУПЛЕКСНЫЙ РЕЖИМ ==========
            logger.info("🎙️ Начинаем полнодуплексный диалог...")
            
            try:
                # Запускаем менеджер (он сам создаст worker-задачи)
                await self._duplex_manager.start()
                
                # Просто ждём, пока система работает
                # (все worker-задачи работают в фоне)
                while self._running:
                    await asyncio.sleep(1)
                    
                    # Периодически проверяем здоровье
                    if not self._duplex_manager.is_running():
                        logger.error("❌ Duplex-менеджер упал, перезапускаем...")
                        await self._duplex_manager.stop()
                        await asyncio.sleep(2)
                        await self._duplex_manager.start()
            
            finally:
                # Корректная остановка
                await self._duplex_manager.stop()
                logger.info("✅ Полнодуплексный диалог завершён")
        
        else:
            # ========== ПОСЛЕДОВАТЕЛЬНЫЙ РЕЖИМ (старый) ==========
            await self._run_sequential_dialogue()
    
    async def _run_sequential_dialogue(self) -> None:
        """
        Последовательный режим диалога (для обратной совместимости).
        
        В этом режиме обработка происходит последовательно:
        listen → process → speak → repeat
        """
        error_count = 0
        consecutive_empty_responses = 0
        
        logger.info("🎙️ Начинаем диалог (последовательный режим)...")
        
        while self._running:
            user_text = None
            reply = None
            context = None
            emotion_name = None
            
            try:
                # ========== 1. СЛУШАЕМ ПОЛЬЗОВАТЕЛЯ (Audio) ==========
                try:
                    user_text = await self.audio.listen(timeout=30.0)
                except asyncio.TimeoutError:
                    logger.debug("⏱️ Timeout — пользователь молчит")
                    await asyncio.sleep(0.5)
                    continue
                
                if not user_text or not user_text.strip():
                    await asyncio.sleep(0.2)
                    continue
                
                logger.info(f"👤 Пользователь: {user_text}")
                
                # ========== 2. СОХРАНЯЕМ В ПАМЯТЬ (Core) ==========
                await self.core.add_turn("user", user_text)
                
                # ========== 3. ПОЛУЧАЕМ КОНТЕКСТ (Core) ==========
                context = await self.core.get_context(
                    last_n_turns=10,
                    max_facts=30
                )
                
                # ========== 4. ПЕРСОНАЛИЗАЦИЯ (Core) ==========
                base_prompt = (
                    "Ты — виртуальный VTuber-компаньон. "
                    "Общайся естественно и поддерживай контакт."
                )
                system_prompt = await self.core.get_personalized_prompt(
                    base_prompt,
                    username="guest",
                    platform="voice"
                )
                
                # ========== 5. ГЕНЕРАЦИЯ ОТВЕТА (LLM) ==========
                reply, emotion_name = await self.llm.generate_reply(
                    user_text,
                    context=context,
                    system_prompt=system_prompt
                )
                
                # ========== 6. ВАЛИДАЦИЯ ОТВЕТА ==========
                if not reply or not reply.strip():
                    consecutive_empty_responses += 1
                    logger.warning(
                        f"⚠️ LLM вернул пустой ответ "
                        f"({consecutive_empty_responses}/3)"
                    )
                    
                    if consecutive_empty_responses >= 3:
                        logger.error("❌ LLM не отвечает, используем fallback")
                        reply = (
                            "Извини, у меня технические проблемы. "
                            "Попробуй переформулировать вопрос."
                        )
                        emotion_name = "neutral"
                        consecutive_empty_responses = 0
                    else:
                        await asyncio.sleep(1)
                        continue
                else:
                    consecutive_empty_responses = 0
                
                logger.info(f"🤖 Ответ: {reply[:60]}... [{emotion_name}]")
                
                # ========== 7. ВАЛИДАЦИЯ ЭМОЦИИ ==========
                VALID_EMOTIONS = {
                    "happy", "sad", "angry", "surprised", "neutral", "joy"
                }
                if not emotion_name or emotion_name.lower() not in VALID_EMOTIONS:
                    logger.warning(
                        f"⚠️ Неизвестная эмоция '{emotion_name}', "
                        f"используем neutral"
                    )
                    emotion_name = "neutral"
                
                # ========== 8. СОХРАНЯЕМ ОТВЕТ (Core) ==========
                await self.core.add_turn("assistant", reply)
                
                # ========== 9. ОЗВУЧИВАЕМ + ПОКАЗЫВАЕМ ЭМОЦИЮ ==========
                # (параллельно, не блокируем друг друга)
                
                # TTS (ждём завершения — пользователь должен услышать)
                tts_task = asyncio.create_task(
                    self.audio.speak(reply, emotion=emotion_name)
                )
                
                # Avatar (неблокирующий — может упасть без проблем)
                if self.avatar.is_running():
                    asyncio.create_task(
                        self.avatar.set_emotion(emotion_name, value=1.0)
                    )
                    asyncio.create_task(
                        self.avatar.speak_signal(True)
                    )
                
                # Ждём завершения TTS
                await tts_task
                
                # Отключаем lip-sync
                if self.avatar.is_running():
                    asyncio.create_task(
                        self.avatar.speak_signal(False)
                    )
                
                # ========== 10. ФОНОВЫЕ ЗАДАЧИ (не блокируют диалог) ==========

            asyncio.create_task(
                self.core.adapt_personality(
                    user_text, 
                    reply,
                    username="guest",      
                    platform="voice"
                )
            )

                # Обновление статистики пользователя (уже было корректно)
            asyncio.create_task(
                self.core.update_user_interaction(
                    username="guest",
                    user_message=user_text,
                    bot_response=reply,
                    emotion=emotion_name,
                    platform="voice"
                )
            )

                # Успех — сбрасываем счётчик ошибок
                error_count = 0
                
            except asyncio.CancelledError:
                logger.info("🛑 Диалог отменён (CancelledError)")
                break
            
            except Exception as e:
                error_count += 1
                logger.exception(
                    f"❌ Ошибка в цикле диалога ({error_count}/5): {e}"
                )
                
                if error_count >= 5:
                    logger.error("❌ Слишком много ошибок, пауза 5 секунд...")
                    await asyncio.sleep(5)
                    error_count = 0
                else:
                    await asyncio.sleep(1)
            
            finally:
                # Очистка памяти
                del user_text, reply, context, emotion_name
                if error_count == 0:
                    await asyncio.sleep(0.1)
        
        logger.info("✅ Выход из цикла диалога (последовательный режим)")
    
    # ------------------------------------------------------------------
    #                    КОЛБЭКИ DUPLEX-РЕЖИМА
    # ------------------------------------------------------------------
    
    def _on_user_message(self, text: str):
        """Колбэк: получено сообщение от пользователя"""
        logger.debug(f"📥 User message: {text[:50]}...")
    
    def _on_bot_reply(self, text: str, emotion: str):
        """Колбэк: бот сгенерировал ответ"""
        logger.debug(f"📤 Bot reply: {text[:50]}... [{emotion}]")
    
    def _on_interrupt(self, text: str):
        """Колбэк: пользователь перебил бота"""
        logger.info(f"⚠️ Перебивание: {text}")
        
        # Можно добавить реакцию (например, эмоцию "surprised")
        if self.avatar.is_running():
            asyncio.create_task(
                self.avatar.pulse_emotion("surprised", 0.5, 0.8)
            )
    
    # ------------------------------------------------------------------
    #                    УТИЛИТЫ И СТАТИСТИКА
    # ------------------------------------------------------------------
    
    async def get_system_stats(self) -> dict:
        """Получить детальную статистику работы всей системы"""
        stats = {
            "running": self._running,
            "duplex_enabled": self._enable_duplex,
            "masters": {}
        }
        
        # Собираем статистику по каждому мастеру
        if self.core.is_running():
            stats["masters"]["core"] = self.core.get_stats()
        
        if self.audio.is_running():
            stats["masters"]["audio"] = self.audio.get_stats()
        
        if self.llm.is_running():
            stats["masters"]["llm"] = self.llm.get_stats()
        
        if self.avatar.is_running():
            stats["masters"]["avatar"] = self.avatar.get_stats()
        
        # Duplex статистика
        if self._duplex_manager:
            stats["duplex"] = self._duplex_manager.get_stats()
        
        return stats
    
    async def get_duplex_stats(self) -> dict:
        """Получить статистику полнодуплексного режима"""
        if self._duplex_manager:
            return self._duplex_manager.get_stats()
        return {"enabled": False}
    
    async def health_check(self) -> dict:
        """Проверка здоровья всех мастеров"""
        health = {}
        
        try:
            health["core"] = await self.core.health_check()
        except Exception:
            health["core"] = False
        
        try:
            health["audio"] = await self.audio.health_check()
        except Exception:
            health["audio"] = False
        
        try:
            health["llm"] = await self.llm.health_check()
        except Exception:
            health["llm"] = False
        
        try:
            health["avatar"] = await self.avatar.health_check()
        except Exception:
            health["avatar"] = False
        
        if self._duplex_manager:
            health["duplex"] = self._duplex_manager.is_running()
        
        return health
    
    async def clear_memory(self) -> None:
        """Очистить кратковременную память диалога"""
        if self.core.is_running():
            await self.core.clear_short_term_memory()
            logger.info("🗑️ Кратковременная память очищена")
    
    def is_ready(self) -> bool:
        """Проверить, готова ли система к работе (критичные мастера)"""
        critical_masters = [self.core, self.audio, self.llm]
        return all(m.is_running() for m in critical_masters)
    
    def get_version(self) -> str:
        """Получить версию системы"""
        return "3.0.0 FINAL + Duplex"


# ------------------------------------------------------------------
#                        ТОЧКА ВХОДА
# ------------------------------------------------------------------

async def main():
    """Главная функция для запуска системы"""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)-20s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Создаём систему
    system = VTuberSystem(
        enable_unity=True,
        enable_luppet_midi=True,
        enable_duplex=True,  # ← Включаем полнодуплексный режим
        install_signal_handlers=True,
    )
    
    try:
        # Запускаем систему
        await system.start()
        
        # Запускаем диалог
        await system.run_dialogue()
    
    except KeyboardInterrupt:
        logger.info("Ctrl+C — завершаем работу...")
    
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
    
    finally:
        # Корректно останавливаем
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
