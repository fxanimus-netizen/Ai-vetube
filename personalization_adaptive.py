# personalization_adaptive.py
import re
import datetime
from typing import Optional

class AdaptivePersonality:
    def __init__(self, personalization_manager):
        self.pm = personalization_manager
        self.min_confirmations = 3          # сколько подряд совпадений нужно
        self.conf_decay = 0.2               # насколько быстро падает доверие
        self.conf_gain = 0.4                # насколько быстро растёт при подтверждении

    async def analyze_and_update(self, user_text: str, assistant_text: str):
        user_id = getattr(self.pm, "current_user_id", None)
        if not user_id:
            return

        profile = await self.pm.get_user_profile(user_id)
        if not profile:
            return

        mood = self.detect_mood(user_text)
        tone = self.detect_tone(user_text)
        new_style = self.detect_response_style(user_text, assistant_text)

        changes = False

        # --- обычное обновление настроения и тона ---
        if mood and profile.get("mood") != mood:
            profile["mood"] = mood
            changes = True
        if tone and profile.get("tone") != tone:
            profile["tone"] = tone
            changes = True

        # --- саморегуляция стиля ---
        current_style = profile.get("response_style")
        style_meta = profile.get("_style_meta", {"candidate": None, "confidence": 0.0})

        if new_style:
            if new_style == current_style:
                # укрепляем уверенность
                style_meta["confidence"] = min(style_meta["confidence"] + self.conf_gain, 1.0)
            elif new_style == style_meta.get("candidate"):
                # тот же кандидат — добавляем подтверждение
                style_meta["confidence"] += self.conf_gain
                if style_meta["confidence"] >= 1.0:
                    profile["response_style"] = new_style
                    style_meta = {"candidate": None, "confidence": 0.0}
                    changes = True
            else:
                # новый кандидат, сбрасываем счётчик
                style_meta = {"candidate": new_style, "confidence": self.conf_gain}

        else:
            # если сигналов нет — уверенность постепенно спадает
            style_meta["confidence"] = max(style_meta["confidence"] - self.conf_decay, 0.0)

        profile["_style_meta"] = style_meta

        if changes:
            profile["last_update"] = datetime.datetime.utcnow().isoformat()
            await self.pm.save_user_profile(user_id, profile)

    # --- эвристики ниже прежние ---
    def detect_mood(self, text: str) -> Optional[str]:
        text = text.lower()
        if any(w in text for w in ["груст", "печаль", "одинок"]):
            return "sad"
        if any(w in text for w in ["рад", "весел", "ура", "класс"]):
            return "happy"
        if any(w in text for w in ["злюсь", "раздраж", "бесит"]):
            return "angry"
        if any(w in text for w in ["спокойн", "ладно", "окей"]):
            return "calm"
        return None

    def detect_tone(self, text: str) -> Optional[str]:
        text = text.lower()
        if re.search(r"\b(давай|сделай|нужно)\b", text):
            return "directive"
        if any(w in text for w in ["спасибо", "пожалуйста", "будь добр"]):
            return "polite"
        if any(w in text for w in ["ха", "лол", "смешно", "ахаха"]):
            return "playful"
        return None

    def detect_response_style(self, user_text: str, assistant_text: str) -> Optional[str]:
        combined = f"{user_text} {assistant_text}".lower()
        if any(w in combined for w in ["по пунктам", "списком", "структурно"]):
            return "structured"
        if any(w in combined for w in ["коротко", "вкратце", "сжато"]):
            return "concise"
        if any(w in combined for w in ["подробно", "развернуто", "глубже"]):
            return "detailed"
        if any(w in combined for w in ["шути", "юмор", "весело", "прикольно"]):
            return "humorous"
        if any(w in combined for w in ["спокойно", "мягко", "поддержи", "утешь"]):
            return "empathetic"
        if any(w in combined for w in ["эмоционально", "ярко", "живее", "энергично"]):
            return "expressive"
        return None
