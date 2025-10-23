"""
core/config.py — конфигурация VTuber системы (v3)
Объединяет dataclass и JSON-сохранение
"""

import json
import os
from dataclasses import dataclass, asdict

CONFIG_PATH = "config.json"

@dataclass
class VTuberConfig:
    """Конфигурация VTuber системы"""
    name: str = "Aiko"
    personality: str = "friendly and cheerful"

    # LLM
    fast_model: str = "llama3:8b"
    smart_model: str = "mistral:7b"

    # TTS
    tts_model: str = "cosyvoice-2"
    tts_voice: str = "alloy"

    # STT
    stt_model: str = "base"
    audio_duration: float = 3.0

    # OSC
    osc_luppet_port: int = 39539
    osc_vseeface_port: int = 39540
    osc_unity_port: int = 39541

    # Память
    memory_db_path: str = "./.data/memory.sqlite"
    memory_chroma_path: str = "./.data/chroma"
    personalization_db: str = "user_profiles.db"

    # Устройство
    device: str = "cuda"

    @classmethod
    def load(cls, path: str = CONFIG_PATH):
        """Загрузить конфиг из JSON"""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**{**asdict(cls()), **data})
            except Exception as e:
                print(f"⚠️ Ошибка загрузки конфига: {e}")
                cfg = cls()
                cfg.save(path)
                return cfg
        else:
            cfg = cls()
            cfg.save(path)
            return cfg
            
    def save(self, path: str = CONFIG_PATH):
        """Сохранить текущие настройки в JSON"""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"❌ Не удалось сохранить конфиг: {e}")

    def validate(self):
        """Проверка корректности конфигурации"""
        assert isinstance(self.name, str) and self.name.strip(), "name не может быть пустым"
        assert isinstance(self.fast_model, str) and self.fast_model.strip(), "fast_model должен быть непустой строкой"
        assert isinstance(self.smart_model, str) and self.smart_model.strip(), "smart_model должен быть непустой строкой"
        assert isinstance(self.tts_voice, str) and self.tts_voice.strip(), "tts_voice должен быть непустой строкой"
        assert 1024 <= self.osc_unity_port <= 65535, "osc_unity_port должен быть в диапазоне 1024-65535"
        assert 0.5 <= self.audio_duration <= 10.0, "audio_duration должна быть от 0.5 до 10 секунд"

# Альтернативный конструктор для обратной совместимости
def create_default_config():
    """Создает конфиг с настройками по умолчанию (для обратной совместимости)"""
    return VTuberConfig()