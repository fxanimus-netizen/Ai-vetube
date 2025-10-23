"""
avatar/emotion.py — Эмоции и маппинг BlendShape для VRM

Здесь:
- перечисление EmotionType (логические эмоции для ИИ/LLM)
- BLENDMAP — соответствие логических эмоций реальным именам BlendShape Clips модели (VRM)
- вспомогательные наборы для моргания и липсинка (фонемы A/I/U/E/O)

Если твоя VRM-модель имеет другие имена клипов, просто поменяй значения в BLENDMAP.
Например, в некоторых моделях "Joy" может называться "Fun" или наоборот.
"""

from enum import Enum

class EmotionType(Enum):
    HAPPY = "happy"          # Радость
    SAD = "sad"              # Грусть
    ANGRY = "angry"          # Злость
    SURPRISED = "surprised"  # Удивление
    NEUTRAL = "neutral"      # Нейтрально
    EXCITED = "excited"      # Азарт/веселье
    PROUD = "proud"          # Гордость
    SCORNFUL = "scornful"    # Презрение
    WORRY = "worry"          # Тревога/волнение

# Маппинг логической эмоции → имя BlendShape Clip в VRM
# ⚠️ Проверь в своей модели: открой VRM в Unity/VRM Viewer/Luppet и посмотри точные имена клипов
BLENDMAP = {
    "happy": "Joy",
    "sad": "Sorrow",
    "angry": "Angry",
    "surprised": "Surprise",
    "neutral": "Neutral",
    "excited": "Fun",
    "proud": "Proud",
    "scornful": "Scornful",
    "worry": "Worry",

    # Дополнительно: моргание и фонемы (если используешь ручное управление)
    "blink": "Blink",
    "blink_left": "Blink_L",
    "blink_right": "Blink_R",

    # Липсинк (фонемы/вискемы)
    "a": "A",
    "i": "I",
    "u": "U",
    "e": "E",
    "o": "O",
}

# Наборы для удобства
PHONEMES = ["a", "i", "u", "e", "o"]      # для липсинка
BLINKS   = ["blink", "blink_left", "blink_right"]
EMOTIONS = [e.value for e in EmotionType] # список "happy", "sad", ...
