
"""
llm/router.py — HybridOllamaRouter
Роутер решает, какой моделью Ollama ответить:
- Saiga/Mistral для быстрых простых вопросов
- DeepSeek-R1 для сложных/логических вопросов
"""

import logging
from .ollama_client import OptimizedOllamaClient

# --- Новые импорты для улучшенного анализа эмоций ---
try:
    import pymorphy2
    from dostoevsky.tokenization import RegexTokenizer
    from dostoevsky.models import FastTextSocialNetworkModel
except ImportError:
    pymorphy2 = None
    FastTextSocialNetworkModel = None

logger = logging.getLogger("Router")


class HybridOllamaRouter:
    def __init__(self, ollama: OptimizedOllamaClient,
                 fast_model="saiga-mistral:7b-lora-custom-q4_K",
                 smart_model="deepseek-r1:14b-q4_K_M"):
        self.ollama = ollama
        self.fast_model = fast_model
        self.smart_model = smart_model

        # --- Инициализация инструментов анализа эмоций ---
        if pymorphy2:
            self.morph = pymorphy2.MorphAnalyzer()
        else:
            self.morph = None

        if FastTextSocialNetworkModel:
            try:
                tokenizer = RegexTokenizer()
                self.sentiment_model = FastTextSocialNetworkModel(tokenizer=tokenizer)
            except Exception as e:
                logger.warning(f"Не удалось инициализировать Dostoevsky: {e}")
                self.sentiment_model = None
        else:
            self.sentiment_model = None

    def _is_complex(self, prompt: str) -> bool:
        """ Определяем, сложный ли вопрос. """
        return len(prompt.split()) > 18 or any(
            kw in prompt.lower() for kw in ["почему", "объясни", "докажи", "как устроено", "план", "шаги"]
        )

    def _is_bad_answer(self, text: str) -> bool:
        """ Проверяем, плохой ли получился ответ (слишком короткий или 'не знаю'). """
        if not text or len(text) < 15:
            return True
        t = text.lower()
        return any(kw in t for kw in ["не знаю", "затрудняюсь", "не могу помочь", "сложно ответить"])

    async def ask(self, prompt: str, system_prompt: str) -> str:
        """ Основной метод: выбрать модель и вернуть ответ. """
        if self._is_complex(prompt):
            r = await self.ollama.generate_response(prompt, system_prompt, self.smart_model, max_tokens=160)
            if r:
                return r

        r = await self.ollama.generate_response(prompt, system_prompt, self.fast_model, max_tokens=80)

        if self._is_bad_answer(r):
            r2 = await self.ollama.generate_response(prompt, system_prompt, self.smart_model, max_tokens=200)
            return r2 or (r or "")

        return r or ""

    # ========== НОВЫЙ МЕТОД — ОБНОВЛЁННЫЙ С АНАЛИЗОМ ЭМОЦИЙ ==========
    async def generate_reply(self, prompt: str, context: str = "", system_prompt: str = "") -> tuple[str, str]:
        """ Генерирует ответ с автоматическим определением эмоции. """
        if context:
            full_prompt = f"{context}\n\nПользователь: {prompt}"
        else:
            full_prompt = prompt

        if not system_prompt:
            system_prompt = "Ты дружелюбный и отзывчивый AI-ассистент."

        reply = await self.ask(full_prompt, system_prompt)
        emotion = self._detect_emotion(reply, prompt)

        logger.debug(f"Ответ: {reply[:60]}... | Эмоция: {emotion}")
        return reply, emotion

    def _lemmatize(self, text: str) -> list[str]:
        """ Приводит слова к основе (если pymorphy2 доступен). """
        if not self.morph:
            return text.lower().split()
        return [self.morph.parse(w)[0].normal_form for w in text.lower().split()]

    def _detect_emotion(self, reply_text: str, user_prompt: str = "") -> str:
        """
        Расширенная детекция эмоций:
        - Dostoevsky для анализа тональности
        - pymorphy2 для лемматизации
        - Ключевые слова как резерв
        """
        text_lower = reply_text.lower()

        # --- 1. Попытка определить эмоцию через модель Dostoevsky ---
        if self.sentiment_model:
            try:
                res = self.sentiment_model.predict([text_lower], k=1)
                sentiment = max(res[0], key=res[0].get)
                if sentiment == "positive":
                    return "happy"
                elif sentiment == "negative":
                    # Проверим наличие гнева/печали по ключам
                    if "злю" in text_lower or "раздраж" in text_lower:
                        return "angry"
                    return "sad"
                elif sentiment == "neutral":
                    return "neutral"
            except Exception as e:
                logger.debug(f"Sentiment model error: {e}")

        # --- 2. Лемматизация для ключевых слов ---
        tokens = self._lemmatize(text_lower)

        # Радость
        if any(w in tokens for w in ["радость", "счастливый", "довольный", "ура", "весёлый", "смешно", "улыбка"]):
            return "happy"

        # Грусть
        if any(w in tokens for w in ["грусть", "печаль", "жаль", "уныние", "расстроенный"]):
            return "sad"

        # Злость
        if any(w in tokens for w in ["злость", "раздражение", "злой", "ярость", "бесит"]):
            return "angry"

        # Удивление
        if any(w in tokens for w in ["удивление", "невероятно", "ого", "воу", "серьёзно"]):
            return "surprised"

        # Волнение / тревога
        if any(w in tokens for w in ["тревога", "волнение", "страх", "опасение", "переживание"]):
            return "worry"

        return "neutral"

    # ========== КОНЕЦ НОВОГО МЕТОДА ==========

    async def ask_streaming(self, prompt: str, system_prompt: str):
        """ Потоковый режим (для стриминга текста и TTS). """
        model = self.smart_model if self._is_complex(prompt) else self.fast_model
        async for chunk in self.ollama.generate_response_streaming(prompt, system_prompt, model):
            yield chunk
