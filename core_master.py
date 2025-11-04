# -*- coding: utf-8 -*-
"""
masters/core_master.py — Ядро VTuber системы (MasterCore)

Управляет:
- Памятью диалогов (HybridMemory)
- Персонализацией пользователей (PersonalizationManager)
- Адаптацией личности (AdaptivePersonality)
- Настроением бота (MoodManager)
- Конфигурацией системы (VTuberConfig)

Версия: 1.0 (2025-11-03) + PATCH (2025-11-04)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict

from .base import BaseMaster
from core.config import VTuberConfig
from core.memory import HybridMemory
from core.mood import MoodManager
from avatar.personalization import PersonalizationManager

# Импорт адаптивной личности
try:
    from avatar.personalization_adaptive import AdaptivePersonality as AdaptivePersonalityV2
    USE_ADAPTIVE_V2 = True
except ImportError:
    USE_ADAPTIVE_V2 = False
    AdaptivePersonalityV2 = None

logger = logging.getLogger("MasterCore")


class AdaptivePersonalityFallback:
    """Fallback-версия адаптивной личности (встроенная из vtuber_system.py)"""
    
    def __init__(self, personalization: PersonalizationManager):
        self.personalization = personalization
    
    async def analyze_and_update(self, user_text: str, model_reply: str, user_id: str) -> None:
        """Простейшая адаптация на ключевых словах"""
        import re
        from datetime import datetime
        
        if not user_id or not user_id.strip():
            logger.warning("⚠️ Fallback: user_id не передан")
            return
        
        mood = self._detect_mood(user_text, model_reply)
        tone = self._detect_tone(user_text, model_reply)
        style = self._detect_style(user_text, model_reply)
        
        changed = False
        profile = await self.personalization.get_user_profile(user_id)
        
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
            await self.personalization.save_user_profile(user_id, profile)
    
    def _detect_mood(self, user_text: str, reply: str) -> Optional[str]:
        text = f"{user_text} {reply}".lower()
        if re.search(r"(плохо|грустн|одинок|устал|тоска|печаль|сложно)", text):
            return "supportive"
        if re.search(r"(весел|ура|смешн|хаха|рад|класс|супер)", text):
            return "cheerful"
        if re.search(r"(злюсь|раздраж|бесит|ненавижу|злой)", text):
            return "calm"
        if re.search(r"(люблю|спасибо|благодарю|благодарен)", text):
            return "empathetic"
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


class MasterCore(BaseMaster):
    """
    Ядро VTuber системы — управление данными и состоянием.
    
    Обязанности:
    - Память диалогов (кратковременная + долговременная)
    - Профили пользователей и персонализация
    - Адаптация личности на основе диалога
    - Управление настроением и эмоциями
    - Конфигурация системы
    
    API для других мастеров:
    - add_turn() — сохранить реплику
    - get_context() — получить контекст диалога
    - get_personalized_prompt() — промпт с учётом пользователя
    - adapt_personality() — обновить модель поведения
    - update_mood() — изменить настроение
    """
    
    def __init__(self, config: Optional[VTuberConfig] = None):
        super().__init__("Core")
        
        # Конфигурация
        self.config = config or VTuberConfig.load()
        
        # Подсистемы (инициализируем при start)
        self.memory: Optional[HybridMemory] = None
        self.personalization: Optional[PersonalizationManager] = None
        self.adaptive = None  # AdaptivePersonality (v1 или v2)
        self.mood = MoodManager()
        
        # Статистика
        self._turn_count = 0
        self._session_start = None
    
    async def _start_internal(self) -> None:
        """Запуск всех подсистем ядра"""
        import time
        self._session_start = time.time()
        
        # 1. Инициализация памяти
        try:
            self.memory = HybridMemory(embed_dim=512)
            await self.memory.aopen(
                db_path=self.config.memory_db_path,
                embed_dim=512
            )
            self.logger.info(f"✅ HybridMemory открыта: {self.config.memory_db_path}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации памяти: {e}")
            raise
        
        # 2. Инициализация персонализации
        try:
            self.personalization = PersonalizationManager(
                db_path=self.config.personalization_db
            )
            await self.personalization.aopen()
            self.logger.info(f"✅ PersonalizationManager открыт: {self.config.personalization_db}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации персонализации: {e}")
            raise
        
        # 3. Инициализация адаптивной личности
        try:
            if USE_ADAPTIVE_V2 and AdaptivePersonalityV2:
                self.adaptive = AdaptivePersonalityV2(self.personalization)
                self.logger.info("✅ AdaptivePersonality v2 (расширенная)")
            else:
                self.adaptive = AdaptivePersonalityFallback(self.personalization)
                self.logger.info("✅ AdaptivePersonality v1 (fallback)")
        except Exception as e:
            self.logger.warning(f"⚠️ Адаптация личности недоступна: {e}")
            self.adaptive = None
        
        # 4. Настроение (уже инициализировано в __init__)
        self.logger.info("✅ MoodManager готов")
        
        self.logger.info("🧠 MasterCore полностью инициализирован")
    
    async def _stop_internal(self) -> None:
        """Закрытие всех подсистем с сохранением данных"""
        # Сохраняем статистику
        if self._session_start:
            import time
            duration = time.time() - self._session_start
            self.logger.info(
                f"📊 Статистика сессии: {self._turn_count} реплик за {duration:.1f}с"
            )
        
        # Закрываем память
        if self.memory:
            try:
                await self.memory.aclose()
                self.logger.info("✅ HybridMemory закрыта")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка закрытия памяти: {e}")
        
        # Закрываем персонализацию
        if self.personalization:
            try:
                await self.personalization.aclose()
                self.logger.info("✅ PersonalizationManager закрыт")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка закрытия персонализации: {e}")
    
    async def health_check(self) -> bool:
        """Проверка здоровья ядра"""
        if not self._running:
            return False
        
        # Проверяем, что все подсистемы живы
        checks = {
            "memory": self.memory is not None,
            "personalization": self.personalization is not None,
            "mood": self.mood is not None,
        }
        
        all_ok = all(checks.values())
        if not all_ok:
            self.logger.warning(f"⚠️ Health check failed: {checks}")
        
        return all_ok
    
    # ==================== API: ПАМЯТЬ ====================
    
    async def add_turn(self, role: str, text: str, meta: Optional[Dict] = None) -> None:
        """
        Сохранить реплику в память.
        
        Args:
            role: 'user' | 'assistant' | 'system'
            text: текст реплики
            meta: дополнительные метаданные (эмоция, timestamp и т.д.)
        """
        if not self.memory:
            raise RuntimeError("MasterCore не запущен (память недоступна)")
        
        try:
            await self.memory.add_turn(role, text, meta or {})
            self._turn_count += 1
            self.logger.debug(f"💾 Сохранена реплика [{role}]: {text[:50]}...")
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения реплики: {e}")
            raise
    
    async def get_context(
        self, 
        last_n_turns: int = 10, 
        max_facts: int = 30
    ) -> Dict:
        """
        Получить контекст диалога для LLM.
        
        Returns:
            {
                "turns": [{"role": "user", "text": "...", ...}, ...],
                "facts": [{"key": "...", "value": "...", ...}, ...]
            }
        """
        if not self.memory:
            raise RuntimeError("MasterCore не запущен (память недоступна)")
        
        try:
            context = await self.memory.context(last_n_turns, max_facts)
            self.logger.debug(
                f"📖 Контекст: {len(context.get('turns', []))} реплик, "
                f"{len(context.get('facts', []))} фактов"
            )
            return context
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения контекста: {e}")
            return {"turns": [], "facts": []}
    
    async def add_fact(self, key: str, value: str, meta: Optional[Dict] = None) -> None:
        """Сохранить факт в долговременную память"""
        if not self.memory:
            raise RuntimeError("MasterCore не запущен (память недоступна)")
        
        try:
            await self.memory.add_fact(key, value, meta or {})
            self.logger.debug(f"💾 Сохранён факт: {key} = {value[:30]}...")
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения факта: {e}")
    
    async def recall(self, query: str, k: int = 5) -> list:
        """Семантический поиск в памяти"""
        if not self.memory:
            return []
        
        try:
            results = await self.memory.recall(query, k)
            self.logger.debug(f"🔍 Найдено {len(results)} релевантных записей")
            return results
        except Exception as e:
            self.logger.error(f"❌ Ошибка поиска в памяти: {e}")
            return []
    
    # ==================== API: ПЕРСОНАЛИЗАЦИЯ ====================
    
    async def get_personalized_prompt(
        self, 
        base_prompt: str, 
        username: str = "guest",
        platform: str = "voice"
    ) -> str:
        """
        Получить промпт с учётом профиля пользователя.
        
        Args:
            base_prompt: базовый системный промпт
            username: имя пользователя
            platform: платформа (voice/telegram/discord)
        
        Returns:
            Персонализированный промпт с контекстом пользователя
        """
        if not self.personalization:
            self.logger.warning("⚠️ Персонализация недоступна, используем базовый промпт")
            return base_prompt
        
        try:
            user_id = self.personalization.get_user_id(username, platform)
            prompt = await self.personalization.get_personalized_system_prompt(
                user_id, 
                base_prompt
            )
            self.logger.debug(f"🎭 Промпт персонализирован для {username}")
            return prompt
        except Exception as e:
            self.logger.error(f"❌ Ошибка персонализации: {e}")
            return base_prompt
    
    async def update_user_interaction(
        self,
        username: str,
        user_message: str,
        bot_response: str,
        emotion: str = "neutral",
        platform: str = "voice"
    ) -> None:
        """
        Обновить статистику взаимодействия с пользователем.
        
        Args:
            username: имя пользователя
            user_message: сообщение пользователя
            bot_response: ответ бота
            emotion: детектированная эмоция
            platform: платформа общения
        """
        if not self.personalization:
            return
        
        try:
            user_id = self.personalization.get_user_id(username, platform)
            await self.personalization.update_interaction(
                user_id,
                user_message,
                bot_response,
                emotion
            )
            self.logger.debug(f"📊 Обновлена статистика для {username}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка обновления статистики: {e}")
    
    # ==================== API: АДАПТАЦИЯ ЛИЧНОСТИ ====================
    
    async def adapt_personality(
        self, 
        user_text: str, 
        bot_reply: str,
        username: str = "guest",
        platform: str = "voice"
    ) -> None:
        """
        Адаптировать личность на основе диалога с автоматическим определением user_id.
        
        Анализирует:
        - Тональность пользователя
        - Стиль общения
        - Предпочтения в ответах
        
        Args:
            user_text: текст пользователя
            bot_reply: ответ бота
            username: имя пользователя (для получения user_id)
            platform: платформа общения
        """
        if not self.adaptive:
            self.logger.debug("⚠️ Адаптация личности недоступна")
            return
        
        try:
            # ✅ ИСПРАВЛЕНО: Получаем user_id через PersonalizationManager
            user_id = self.personalization.get_user_id(username, platform)
            
            # ✅ Передаем явно в analyze_and_update
            await self.adaptive.analyze_and_update(
                user_text, 
                bot_reply,
                user_id=user_id
            )
            self.logger.debug(f"🧠 Личность адаптирована для {username} ({user_id[:8]}...)")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка адаптации личности: {e}", exc_info=True)
    
    # ==================== API: НАСТРОЕНИЕ ====================
    
    async def update_mood(self, text: str) -> str:
        """
        Обновить настроение на основе текста.
        
        Returns:
            Доминантная эмоция ('happy'/'sad'/'angry'/'neutral'/...)
        """
        try:
            emotion = await self.mood.async_update(text)
            self.logger.debug(f"😊 Настроение обновлено: {emotion}")
            return emotion
        except Exception as e:
            self.logger.error(f"❌ Ошибка обновления настроения: {e}")
            return "neutral"
    
    def get_current_emotion(self) -> str:
        """Получить текущую эмоцию"""
        return self.mood.current_emotion()
    
    def get_mood_state(self) -> Dict[str, float]:
        """Получить детальное состояние настроения"""
        return self.mood.get_state()
    
    async def decay_mood(self) -> None:
        """Применить затухание эмоций (вызывать периодически)"""
        try:
            await self.mood.async_decay()
        except Exception as e:
            self.logger.error(f"❌ Ошибка затухания настроения: {e}")
    
    # ==================== УТИЛИТЫ ====================
    
    async def clear_short_term_memory(self) -> None:
        """Очистить кратковременную память (диалоги)"""
        if self.memory:
            try:
                await self.memory.clear_short_term()
                self._turn_count = 0
                self.logger.info("🗑️ Кратковременная память очищена")
            except Exception as e:
                self.logger.error(f"❌ Ошибка очистки памяти: {e}")
    
    def get_stats(self) -> Dict:
        """Получить статистику работы ядра"""
        import time
        duration = time.time() - self._session_start if self._session_start else 0
        
        return {
            "running": self._running,
            "turn_count": self._turn_count,
            "session_duration_sec": duration,
            "current_emotion": self.get_current_emotion(),
            "mood_state": self.get_mood_state(),
        }
