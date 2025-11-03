# -*- coding: utf-8 -*-
"""
duplex_dialogue.py — Полнодуплексный диалог (v1.0)

КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:
1. Параллельная работа STT и TTS (можно перебивать бота)
2. Умная очередь реплик (приоритеты)
3. Интеллектуальное прерывание (не режем на середине мысли)
4. Управление вниманием (детекция перебивания)

АРХИТЕКТУРА:
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ STT Loop    │────▶│ Message Queue│────▶│ LLM Worker  │
│ (слушаем)   │     │ (приоритеты) │     │ (генерация) │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │ TTS Loop     │◀────│ Reply Queue │
                    │ (говорим)    │     │             │
                    └──────────────┘     └─────────────┘
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Dict
from collections import deque

logger = logging.getLogger("DuplexDialogue")


# ==================== ТИПЫ И МОДЕЛИ ====================

class MessagePriority(Enum):
    """Приоритеты сообщений в очереди"""
    URGENT = 1      # Команды типа "стоп", "хватит"
    HIGH = 2        # Перебивание во время речи бота
    NORMAL = 3      # Обычные реплики
    LOW = 4         # Фоновые задачи (адаптация, статистика)


@dataclass(order=True)
class QueuedMessage:
    """Сообщение в очереди обработки"""
    priority: MessagePriority = field(compare=True)
    timestamp: float = field(compare=True)
    text: str = field(compare=False)
    meta: Dict = field(default_factory=dict, compare=False)


class InterruptionStrategy(Enum):
    """Стратегии прерывания речи бота"""
    IMMEDIATE = "immediate"     # Прервать немедленно
    SENTENCE = "sentence"       # Дождаться конца предложения
    NEVER = "never"            # Никогда не прерывать (например, важное сообщение)


# ==================== ПОЛНОДУПЛЕКСНЫЙ МЕНЕДЖЕР ====================

class DuplexDialogueManager:
    """
    Менеджер полнодуплексного диалога с параллельной обработкой.
    
    Возможности:
    - Одновременная работа STT и TTS
    - Умное прерывание (с учётом контекста)
    - Приоритетная очередь сообщений
    - Детекция перебивания пользователем
    - Управление вниманием (attention management)
    
    Использование:
        manager = DuplexDialogueManager(core, audio, llm, avatar)
        await manager.start()
        await manager.run()  # Основной цикл
        await manager.stop()
    """
    
    def __init__(
        self,
        core,  # MasterCore
        audio,  # MasterAudio
        llm,  # MasterLLM
        avatar,  # MasterAvatar
        interruption_strategy: InterruptionStrategy = InterruptionStrategy.SENTENCE,
        user_silence_timeout: float = 30.0,
        bot_silence_timeout: float = 2.0,
    ):
        self.core = core
        self.audio = audio
        self.llm = llm
        self.avatar = avatar
        
        # Конфигурация
        self.interruption_strategy = interruption_strategy
        self.user_silence_timeout = user_silence_timeout
        self.bot_silence_timeout = bot_silence_timeout
        
        # Очереди
        self.input_queue = asyncio.PriorityQueue()  # Входящие сообщения (от STT)
        self.output_queue = asyncio.Queue()         # Исходящие реплики (для TTS)
        
        # Состояние
        self._running = False
        self._bot_speaking = asyncio.Event()        # Бот говорит прямо сейчас
        self._interrupt_requested = asyncio.Event()  # Запрос на прерывание
        self._processing = asyncio.Event()          # LLM обрабатывает запрос
        
        # Задачи (workers)
        self._tasks = []
        
        # Статистика
        self._stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "interruptions": 0,
            "errors": 0,
        }
        
        # Колбэки (для расширения)
        self.on_interrupt: Optional[Callable[[str], None]] = None
        self.on_user_message: Optional[Callable[[str], None]] = None
        self.on_bot_reply: Optional[Callable[[str, str], None]] = None
    
    # ==================== ЖИЗНЕННЫЙ ЦИКЛ ====================
    
    async def start(self) -> None:
        """Запуск всех worker-задач"""
        if self._running:
            logger.warning("DuplexDialogue уже запущен")
            return
        
        logger.info("🚀 Запуск полнодуплексного диалога...")
        
        self._running = True
        
        # Создаём worker-задачи
        self._tasks = [
            asyncio.create_task(self._stt_worker(), name="STT-Worker"),
            asyncio.create_task(self._llm_worker(), name="LLM-Worker"),
            asyncio.create_task(self._tts_worker(), name="TTS-Worker"),
            asyncio.create_task(self._watchdog(), name="Watchdog"),
        ]
        
        logger.info("✅ Полнодуплексный диалог запущен")
    
    async def stop(self) -> None:
        """Остановка всех задач"""
        if not self._running:
            return
        
        logger.info("🛑 Останавливаем полнодуплексный диалог...")
        
        self._running = False
        
        # Отменяем все задачи
        for task in self._tasks:
            task.cancel()
        
        # Ждём завершения
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Очищаем очереди
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"✅ Полнодуплексный диалог остановлен. Статистика: {self._stats}")
    
    # ==================== WORKER-ЗАДАЧИ ====================
    
    async def _stt_worker(self) -> None:
        """
        Worker для распознавания речи.
        Непрерывно слушает микрофон и добавляет сообщения в очередь.
        """
        logger.info("🎤 STT Worker запущен")
        
        consecutive_errors = 0
        
        while self._running:
            try:
                # Слушаем пользователя
                user_text = await self.audio.listen(timeout=self.user_silence_timeout)
                
                if not user_text or not user_text.strip():
                    await asyncio.sleep(0.1)
                    continue
                
                user_text = user_text.strip()
                logger.info(f"👤 Пользователь: {user_text}")
                
                # Увеличиваем счётчик
                self._stats["messages_received"] += 1
                
                # Колбэк (если есть)
                if self.on_user_message:
                    try:
                        self.on_user_message(user_text)
                    except Exception as e:
                        logger.warning(f"Ошибка в on_user_message: {e}")
                
                # Определяем приоритет
                priority = self._classify_message(user_text)
                
                # Если бот говорит И это перебивание — обрабатываем
                if self._bot_speaking.is_set() and priority in (
                    MessagePriority.URGENT,
                    MessagePriority.HIGH
                ):
                    logger.info(f"⚠️ Перебивание! Приоритет: {priority.name}")
                    self._stats["interruptions"] += 1
                    
                    # Запрашиваем прерывание
                    self._interrupt_requested.set()
                    
                    # Колбэк
                    if self.on_interrupt:
                        try:
                            self.on_interrupt(user_text)
                        except Exception:
                            pass
                
                # Добавляем в очередь
                msg = QueuedMessage(
                    priority=priority,
                    timestamp=asyncio.get_event_loop().time(),
                    text=user_text,
                    meta={"interrupted": self._bot_speaking.is_set()}
                )
                
                await self.input_queue.put(msg)
                
                # Успешная обработка — сбрасываем ошибки
                consecutive_errors = 0
                
            except asyncio.CancelledError:
                logger.info("STT Worker отменён")
                break
            
            except asyncio.TimeoutError:
                # Пользователь молчит — это нормально
                await asyncio.sleep(0.1)
            
            except Exception as e:
                consecutive_errors += 1
                self._stats["errors"] += 1
                logger.error(f"❌ Ошибка STT Worker ({consecutive_errors}/5): {e}")
                
                if consecutive_errors >= 5:
                    logger.error("STT Worker: критическая ошибка, пауза 5 сек")
                    await asyncio.sleep(5)
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(1)
    
    async def _llm_worker(self) -> None:
        """
        Worker для генерации ответов.
        Берёт сообщения из очереди и генерирует ответы через LLM.
        """
        logger.info("🧠 LLM Worker запущен")
        
        while self._running:
            try:
                # Ждём сообщение из очереди (с таймаутом)
                try:
                    msg = await asyncio.wait_for(
                        self.input_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                logger.debug(f"📥 Обрабатываем: [{msg.priority.name}] {msg.text[:50]}...")
                
                # Помечаем, что обрабатываем
                self._processing.set()
                
                try:
                    # Сохраняем в память
                    await self.core.add_turn("user", msg.text)
                    
                    # Получаем контекст
                    context = await self.core.get_context(last_n_turns=10, max_facts=30)
                    
                    # Персонализированный промпт
                    base_prompt = (
                        "Ты — виртуальный VTuber-компаньон. "
                        "Общайся естественно и поддерживай контакт."
                    )
                    system_prompt = await self.core.get_personalized_prompt(
                        base_prompt,
                        username="guest",
                        platform="voice"
                    )
                    
                    # Генерируем ответ
                    reply, emotion = await self.llm.generate_reply(
                        msg.text,
                        context=context,
                        system_prompt=system_prompt
                    )
                    
                    if not reply or not reply.strip():
                        logger.warning("LLM вернул пустой ответ")
                        reply = "Извини, не расслышала. Повтори, пожалуйста?"
                        emotion = "neutral"
                    
                    logger.info(f"🤖 Ответ: {reply[:60]}... [{emotion}]")
                    
                    # Сохраняем ответ
                    await self.core.add_turn("assistant", reply)
                    
                    # Отправляем в TTS
                    await self.output_queue.put({
                        "text": reply,
                        "emotion": emotion,
                        "strategy": self._get_interruption_strategy(msg),
                        "original_msg": msg,
                    })
                    
                    # Фоновые задачи (не блокируют)
                    asyncio.create_task(self._background_tasks(msg.text, reply, emotion))
                    
                    # Увеличиваем счётчик
                    self._stats["messages_processed"] += 1
                    
                    # Колбэк
                    if self.on_bot_reply:
                        try:
                            self.on_bot_reply(reply, emotion)
                        except Exception:
                            pass
                
                finally:
                    self._processing.clear()
            
            except asyncio.CancelledError:
                logger.info("LLM Worker отменён")
                break
            
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"❌ Ошибка LLM Worker: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _tts_worker(self) -> None:
        """
        Worker для синтеза речи.
        Берёт ответы из очереди и озвучивает их.
        """
        logger.info("🔊 TTS Worker запущен")
        
        while self._running:
            try:
                # Ждём ответ из очереди
                try:
                    reply_data = await asyncio.wait_for(
                        self.output_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                text = reply_data["text"]
                emotion = reply_data["emotion"]
                strategy = reply_data["strategy"]
                
                logger.debug(f"🔊 Озвучиваем: {text[:50]}... [{emotion}]")
                
                # Помечаем, что говорим
                self._bot_speaking.set()
                self._interrupt_requested.clear()
                
                # Устанавливаем эмоцию в аватаре
                if self.avatar.is_running():
                    asyncio.create_task(self.avatar.set_emotion(emotion, value=1.0))
                    asyncio.create_task(self.avatar.speak_signal(True))
                
                try:
                    # Озвучиваем (с поддержкой прерывания)
                    await self._speak_with_interruption(text, emotion, strategy)
                
                finally:
                    # Снимаем флаг
                    self._bot_speaking.clear()
                    
                    # Отключаем lip-sync
                    if self.avatar.is_running():
                        asyncio.create_task(self.avatar.speak_signal(False))
            
            except asyncio.CancelledError:
                logger.info("TTS Worker отменён")
                break
            
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"❌ Ошибка TTS Worker: {e}")
                self._bot_speaking.clear()
                await asyncio.sleep(1)
    
    async def _watchdog(self) -> None:
        """
        Watchdog для мониторинга состояния системы.
        Периодически проверяет здоровье и выводит статистику.
        """
        logger.info("🐕 Watchdog запущен")
        
        while self._running:
            try:
                await asyncio.sleep(60)  # Раз в минуту
                
                # Проверяем здоровье мастеров
                health = {
                    "core": self.core.is_running(),
                    "audio": self.audio.is_running(),
                    "llm": self.llm.is_running(),
                    "avatar": self.avatar.is_running(),
                }
                
                # Статистика
                logger.info(
                    f"📊 Статистика (1 мин): "
                    f"received={self._stats['messages_received']}, "
                    f"processed={self._stats['messages_processed']}, "
                    f"interruptions={self._stats['interruptions']}, "
                    f"errors={self._stats['errors']}"
                )
                
                # Предупреждения
                if not all(health.values()):
                    unhealthy = [k for k, v in health.items() if not v]
                    logger.warning(f"⚠️ Неисправные мастера: {unhealthy}")
            
            except asyncio.CancelledError:
                logger.info("Watchdog отменён")
                break
            
            except Exception as e:
                logger.error(f"❌ Ошибка Watchdog: {e}")
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def _classify_message(self, text: str) -> MessagePriority:
        """
        Классифицирует сообщение по приоритету.
        
        URGENT: команды типа "стоп", "хватит", "заткнись"
        HIGH: перебивание, когда бот говорит
        NORMAL: обычные реплики
        """
        text_lower = text.lower()
        
        # Ключевые слова для URGENT
        urgent_keywords = [
            "стоп", "хватит", "заткнись", "молчи", "тихо", "прекрати",
            "stop", "shut up", "quiet"
        ]
        
        if any(kw in text_lower for kw in urgent_keywords):
            return MessagePriority.URGENT
        
        # Если бот говорит — HIGH (перебивание)
        if self._bot_speaking.is_set():
            return MessagePriority.HIGH
        
        # Остальное — NORMAL
        return MessagePriority.NORMAL
    
    def _get_interruption_strategy(self, msg: QueuedMessage) -> InterruptionStrategy:
        """Определяет стратегию прерывания в зависимости от приоритета"""
        if msg.priority == MessagePriority.URGENT:
            return InterruptionStrategy.IMMEDIATE
        elif msg.priority == MessagePriority.HIGH:
            return InterruptionStrategy.SENTENCE
        else:
            return self.interruption_strategy
    
    async def _speak_with_interruption(
        self,
        text: str,
        emotion: str,
        strategy: InterruptionStrategy
    ) -> None:
        """
        Озвучивание с поддержкой прерывания.
        
        В зависимости от стратегии:
        - IMMEDIATE: прерываем немедленно при запросе
        - SENTENCE: дожидаемся конца предложения
        - NEVER: игнорируем запросы прерывания
        """
        if strategy == InterruptionStrategy.NEVER:
            # Просто озвучиваем без прерываний
            await self.audio.speak(text, emotion=emotion, interrupt=False)
            return
        
        # Разбиваем на предложения (упрощённо)
        sentences = self._split_sentences(text)
        
        for i, sentence in enumerate(sentences):
            # Проверяем запрос на прерывание
            if self._interrupt_requested.is_set():
                if strategy == InterruptionStrategy.IMMEDIATE:
                    logger.info("⚠️ Немедленное прерывание")
                    await self.audio.stop_speaking()
                    break
                elif strategy == InterruptionStrategy.SENTENCE:
                    if i < len(sentences) - 1:
                        logger.info(f"⚠️ Прерывание после предложения ({i+1}/{len(sentences)})")
                        await self.audio.speak(sentence, emotion=emotion, interrupt=False)
                        break
            
            # Озвучиваем предложение
            await self.audio.speak(sentence, emotion=emotion, interrupt=False)
            
            # Небольшая пауза между предложениями
            if i < len(sentences) - 1:
                await asyncio.sleep(self.bot_silence_timeout)
    
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Разбивает текст на предложения (упрощённо)"""
        import re
        # Разбиваем по . ! ? (с учётом пробелов)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _background_tasks(self, user_text: str, reply: str, emotion: str) -> None:
        """Фоновые задачи, которые не блокируют основной диалог"""
        try:
            # Адаптация личности
            await self.core.adapt_personality(user_text, reply)
            
            # Обновление статистики пользователя
            await self.core.update_user_interaction(
                username="guest",
                user_message=user_text,
                bot_response=reply,
                emotion=emotion,
                platform="voice"
            )
        except Exception as e:
            logger.warning(f"Ошибка в фоновых задачах: {e}")
    
    # ==================== ПУБЛИЧНОЕ API ====================
    
    def get_stats(self) -> dict:
        """Получить статистику работы"""
        return {
            **self._stats,
            "running": self._running,
            "bot_speaking": self._bot_speaking.is_set(),
            "processing": self._processing.is_set(),
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
        }
    
    def is_running(self) -> bool:
        """Проверить, запущен ли менеджер"""
        return self._running


# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

if __name__ == "__main__":
    import sys
    
    # Пример интеграции
    async def demo():
        # Псевдо-мастера для демонстрации
        class DummyMaster:
            def is_running(self): return True
            async def add_turn(self, *args): pass
            async def get_context(self, *args): return {"turns": [], "facts": []}
            async def get_personalized_prompt(self, *args): return "Test prompt"
            async def adapt_personality(self, *args): pass
            async def update_user_interaction(self, *args): pass
        
        class DummyLLM(DummyMaster):
            async def generate_reply(self, *args, **kwargs):
                await asyncio.sleep(0.5)  # Имитация генерации
                return "Test reply", "neutral"
        
        class DummyAudio(DummyMaster):
            async def listen(self, timeout=30):
                await asyncio.sleep(2)
                return "Test message"
            
            async def speak(self, text, **kwargs):
                print(f"🔊 Speaking: {text[:50]}...")
                await asyncio.sleep(1)
            
            async def stop_speaking(self):
                pass
        
        class DummyAvatar(DummyMaster):
            async def set_emotion(self, *args, **kwargs): pass
            async def speak_signal(self, *args): pass
        
        # Создаём менеджер
        core = DummyMaster()
        audio = DummyAudio()
        llm = DummyLLM()
        avatar = DummyAvatar()
        
        manager = DuplexDialogueManager(
            core, audio, llm, avatar,
            interruption_strategy=InterruptionStrategy.SENTENCE
        )
        
        # Запускаем
        await manager.start()
        
        # Работаем 10 секунд
        await asyncio.sleep(10)
        
        # Останавливаем
        await manager.stop()
        
        # Статистика
        print(f"\n📊 Финальная статистика: {manager.get_stats()}")
    
    asyncio.run(demo())
