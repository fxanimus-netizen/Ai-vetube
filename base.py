# -*- coding: utf-8 -*-
"""
masters/base.py — Базовый класс для всех мастеров
"""

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("BaseMaster")


class BaseMaster(ABC):
    """Базовый класс для всех мастеров системы"""
    
    def __init__(self, name: str):
        self.name = name
        self._running = False
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(f"Master.{name}")
    
    @abstractmethod
    async def _start_internal(self) -> None:
        """Внутренняя логика запуска"""
        pass
    
    @abstractmethod
    async def _stop_internal(self) -> None:
        """Внутренняя логика остановки"""
        pass
    
    async def start(self) -> None:
        """Публичный метод запуска"""
        async with self._lock:
            if self._running:
                self.logger.warning(f"{self.name} уже запущен")
                return
            
            self.logger.info(f"🚀 Запуск {self.name}...")
            try:
                await self._start_internal()
                self._running = True
                self.logger.info(f"✅ {self.name} запущен")
            except Exception as e:
                self.logger.error(f"❌ Ошибка запуска {self.name}: {e}", exc_info=True)
                raise
    
    async def stop(self) -> None:
        """Публичный метод остановки"""
        async with self._lock:
            if not self._running:
                return
            
            self.logger.info(f"🛑 Остановка {self.name}...")
            try:
                await self._stop_internal()
                self._running = False
                self.logger.info(f"✅ {self.name} остановлен")
            except Exception as e:
                self.logger.error(f"⚠️ Ошибка остановки {self.name}: {e}")
    
    async def health_check(self) -> bool:
        """Проверка здоровья мастера"""
        return self._running
    
    def is_running(self) -> bool:
        return self._running