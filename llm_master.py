# -*- coding: utf-8 -*-
"""
masters/llm_master.py — LLM-мастер VTuber системы

Управляет:
- Генерацией ответов (HybridOllamaRouter)
- Ollama клиентом и кэшем
- Детекцией эмоций из текста
- Роутингом между fast/smart моделями

Версия: 1.0 (2025-11-03)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple, Dict

from .base import BaseMaster

# Импорты LLM-модулей
from llm.router import HybridOllamaRouter
from llm.ollama_client import OptimizedOllamaClient

# Импорты конфигурации
from core.config import VTuberConfig

logger = logging.getLogger("MasterLLM")


class MasterLLM(BaseMaster):
    """
    LLM-мастер — управление генерацией ответов и роутингом моделей.
    
    Возможности:
    - Генерация ответов через Ollama
    - Роутинг между fast/smart моделями
    - Детекция эмоций из текста
    - Кэширование ответов
    
    API:
    - generate_reply() — генерация ответа с эмоцией
    - generate_streaming() — потоковая генерация
    - detect_emotion() — определить эмоцию из текста
    """
    
    def __init__(
        self,
        config: Optional[VTuberConfig] = None,
        fast_model: Optional[str] = None,
        smart_model: Optional[str] = None,
    ):
        super().__init__("LLM")
        
        # Конфигурация
        self.config = config or VTuberConfig.load()
        
        # Модели (можно переопределить)
        self._fast_model = fast_model or self.config.fast_model
        self._smart_model = smart_model or self.config.smart_model
        
        # Подсистемы (инициализируем при start)
        self.ollama_client: Optional[OptimizedOllamaClient] = None
        self.router: Optional[HybridOllamaRouter] = None
        
        # Статистика
        self._generation_count = 0
        self._fast_model_usage = 0
        self._smart_model_usage = 0
        self._error_count = 0
    
    async def _start_internal(self) -> None:
        """Запуск Ollama клиента и роутера"""
        try:
            # 1. Инициализация Ollama клиента
            self.ollama_client = OptimizedOllamaClient()
            self.logger.info("✅ Ollama клиент инициализирован")
            
            # 2. Инициализация роутера
            self.router = HybridOllamaRouter(
                ollama=self.ollama_client,
                fast_model=self._fast_model,
                smart_model=self._smart_model,
            )
            self.logger.info(
                f"✅ LLM роутер готов: fast={self._fast_model}, smart={self._smart_model}"
            )
            
            self.logger.info("🧠 MasterLLM полностью инициализирован")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации LLM: {e}", exc_info=True)
            raise
    
    async def _stop_internal(self) -> None:
        """Закрытие Ollama клиента"""
        # Статистика
        self.logger.info(
            f"📊 Статистика LLM: generations={self._generation_count}, "
            f"fast={self._fast_model_usage}, smart={self._smart_model_usage}, "
            f"errors={self._error_count}"
        )
        
        # Закрываем клиент
        if self.ollama_client:
            try:
                await self.ollama_client.close()
                self.logger.info("✅ Ollama клиент закрыт")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка закрытия Ollama: {e}")
    
    async def health_check(self) -> bool:
        """Проверка здоровья LLM-подсистем"""
        if not self._running:
            return False
        
        checks = {
            "ollama_client": self.ollama_client is not None,
            "router": self.router is not None,
        }
        
        all_ok = all(checks.values())
        if not all_ok:
            self.logger.warning(f"⚠️ Health check failed: {checks}")
        
        return all_ok
    
    # ==================== API: ГЕНЕРАЦИЯ ОТВЕТОВ ====================
    
    async def generate_reply(
        self,
        user_text: str,
        context: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Генерация ответа с автоматической детекцией эмоции.
        
        Args:
            user_text: текст пользователя
            context: контекст диалога (turns + facts)
            system_prompt: системный промпт
        
        Returns:
            (reply_text, emotion_name)
        
        Raises:
            RuntimeError: если MasterLLM не запущен
        """
        if not self.router:
            raise RuntimeError("MasterLLM не запущен (роутер недоступен)")
        
        try:
            # Формируем контекст
            context_str = ""
            if context:
                turns = context.get("turns", [])
                context_str = "\n".join([
                    f"{t['role']}: {t['text']}" for t in turns[-10:]
                ])
                full_prompt = f"{context_str}\n\nuser: {user_text}"
            else:
                full_prompt = user_text
            
            # Системный промпт по умолчанию
            if not system_prompt:
                system_prompt = "Ты — виртуальный VTuber-компаньон. Общайся естественно."
            
            # Генерируем ответ
            reply, emotion = await self.router.generate_reply(
                full_prompt,
                context=context_str,
                system_prompt=system_prompt
            )
            
            self._generation_count += 1
            
            # Определяем, какая модель использовалась (примерно)
            if len(user_text.split()) > 18:
                self._smart_model_usage += 1
            else:
                self._fast_model_usage += 1
            
            self.logger.debug(
                f"🤖 Сгенерирован ответ: {reply[:50]}... [{emotion}]"
            )
            
            return reply, emotion
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка генерации ответа: {e}", exc_info=True)
            raise
    
    async def generate_streaming(
        self,
        user_text: str,
        system_prompt: Optional[str] = None,
    ):
        """
        Потоковая генерация ответа (для TTS в реальном времени).
        
        Args:
            user_text: текст пользователя
            system_prompt: системный промпт
        
        Yields:
            Чанки текста по мере генерации
        
        Raises:
            RuntimeError: если MasterLLM не запущен
        """
        if not self.router:
            raise RuntimeError("MasterLLM не запущен (роутер недоступен)")
        
        try:
            if not system_prompt:
                system_prompt = "Ты — виртуальный VTuber-компаньон."
            
            async for chunk in self.router.ask_streaming(user_text, system_prompt):
                yield chunk
            
            self._generation_count += 1
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка потоковой генерации: {e}")
            raise
    
    # ==================== API: ДЕТЕКЦИЯ ЭМОЦИЙ ====================
    
    def detect_emotion(self, text: str) -> str:
        """
        Детекция эмоции из текста (упрощённая эвристика).
        
        Args:
            text: текст для анализа
        
        Returns:
            Название эмоции (happy/sad/angry/surprised/neutral)
        """
        if not self.router:
            return "neutral"
        
        try:
            # Используем встроенный метод роутера
            emotion = self.router._detect_emotion(text, "")
            self.logger.debug(f"😊 Детектирована эмоция: {emotion}")
            return emotion
        
        except Exception as e:
            self.logger.error(f"❌ Ошибка детекции эмоции: {e}")
            return "neutral"
    
    # ==================== API: ПРЯМОЙ ДОСТУП ====================
    
    async def ask_fast(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Прямой запрос к fast-модели (минуя роутер).
        
        Args:
            prompt: промпт
            system_prompt: системный промпт
        
        Returns:
            Ответ модели
        """
        if not self.ollama_client:
            raise RuntimeError("MasterLLM не запущен")
        
        try:
            response = await self.ollama_client.generate(
                prompt=prompt,
                system=system_prompt,
                params={"model": self._fast_model}
            )
            self._fast_model_usage += 1
            return response
        
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка запроса к fast-модели: {e}")
            raise
    
    async def ask_smart(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Прямой запрос к smart-модели (минуя роутер).
        
        Args:
            prompt: промпт
            system_prompt: системный промпт
        
        Returns:
            Ответ модели
        """
        if not self.ollama_client:
            raise RuntimeError("MasterLLM не запущен")
        
        try:
            response = await self.ollama_client.generate(
                prompt=prompt,
                system=system_prompt,
                params={"model": self._smart_model}
            )
            self._smart_model_usage += 1
            return response
        
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"❌ Ошибка запроса к smart-модели: {e}")
            raise
    
    # ==================== УТИЛИТЫ ====================
    
    def get_stats(self) -> dict:
        """Получить статистику работы LLM"""
        return {
            "running": self._running,
            "fast_model": self._fast_model,
            "smart_model": self._smart_model,
            "generation_count": self._generation_count,
            "fast_usage": self._fast_model_usage,
            "smart_usage": self._smart_model_usage,
            "error_count": self._error_count,
        }
    
    def get_models(self) -> dict:
        """Получить информацию о моделях"""
        return {
            "fast": self._fast_model,
            "smart": self._smart_model,
        }
