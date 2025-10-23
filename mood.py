"""
core/mood.py — управление эмоциональным состоянием VTuber (v3, улучшенная версия)

Патч от 2025-10-11:
• Добавлены асинхронные методы (async_update, async_decay) для безопасного использования в asyncio.
• Реализовано плавное затухание эмоций по реальному времени (в секундах).
• Добавлена нормализация состояния (сумма эмоций = 1.0).
• Добавлена поддержка колбэков on_change — TTS/Avatar могут подписываться на события смены эмоции.
• Базовый sentiment-анализ через простые ключевые слова (для лёгкости без внешних зависимостей).
"""

import asyncio
import math
import time
import logging
from typing import Dict, Optional, Callable

logger = logging.getLogger("MoodManager")


class MoodManager:
    def __init__(self):
        # Текущее состояние эмоций (0–1)
        self.state: Dict[str, float] = {
            "happy": 0.2,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.0,
            "neutral": 0.8,
        }
        # Коэффициенты затухания (чем меньше, тем быстрее остывает)
        self.decay_rate: Dict[str, float] = {
            "happy": 0.97,
            "sad": 0.95,
            "angry": 0.92,
            "surprised": 0.96,
            "neutral": 0.99,
        }
        self.last_update_time = time.time()
        self.on_change: Optional[Callable[[str], None]] = None

    # ------------------------------ Вспомогательные ------------------------------
    def _normalize(self):
        total = sum(self.state.values()) or 1.0
        for k in self.state:
            self.state[k] = max(0.0, min(1.0, self.state[k] / total))

    def _dominant(self) -> str:
        return max(self.state, key=self.state.get)

    def _apply_decay(self, dt: float):
        for k, rate in self.decay_rate.items():
            self.state[k] *= rate ** dt
        self._normalize()

    # ------------------------------ Основные функции ------------------------------
    def update_mood(self, text: str) -> str:
        """Обновление эмоции по входному тексту (упрощённый sentiment-анализ)."""
        text = text.lower().strip()
        delta = 0.15

        if any(word in text for word in ["спасибо", "классно", "ура", "люблю"]):
            self.state["happy"] += delta
        elif any(word in text for word in ["плохо", "грустно", "печально", "устал"]):
            self.state["sad"] += delta
        elif any(word in text for word in ["злюсь", "ненавижу", "ужас", "в бешенстве"]):
            self.state["angry"] += delta
        elif any(word in text for word in ["вау", "ого", "ничего себе", "серьёзно?"]):
            self.state["surprised"] += delta
        else:
            self.state["neutral"] += 0.05

        self._normalize()
        dom = self._dominant()

        # уведомляем подписчиков
        if self.on_change:
            try:
                self.on_change(dom)
            except Exception as e:
                logger.warning(f"Ошибка в on_change: {e}")

        logger.info(f"Новая эмоция: {dom} (state={self.state})")
        self.last_update_time = time.time()
        return dom

    def decay(self):
        """Затухание эмоций со временем (вызов вручную)."""
        now = time.time()
        dt = now - self.last_update_time
        if dt <= 0:
            return
        self._apply_decay(dt)
        self.last_update_time = now

    # ------------------------------ Асинхронные варианты ------------------------------
    async def async_update(self, text: str) -> str:
        return await asyncio.to_thread(self.update_mood, text)

    async def async_decay(self):
        return await asyncio.to_thread(self.decay)

    # ------------------------------ Вспомогательные методы ------------------------------
    def current_emotion(self) -> str:
        return self._dominant()

    def get_state(self) -> Dict[str, float]:
        return dict(self.state)


# ------------------------------ Тест ------------------------------
if __name__ == "__main__":
    async def _demo():
        mood = MoodManager()

        # Пример подписки (аватар реагирует на смену эмоции)
        mood.on_change = lambda e: print(f"[Avatar] → эмоция: {e}")

        await mood.async_update("Спасибо, я рад тебя видеть!")
        await asyncio.sleep(1)
        await mood.async_update("Мне плохо, я устал.")
        await asyncio.sleep(2)
        await mood.async_decay()
        print("Текущее состояние:", mood.get_state())

    asyncio.run(_demo())
