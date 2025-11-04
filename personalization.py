# -*- coding: utf-8 -*-
"""
avatar/personalization.py — async aiosqlite patch
- Replaces sqlite3 + check_same_thread with aiosqlite
- Serializes writes via asyncio.Semaphore(1)
- Keeps original table schema and public behavior (best-effort)
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import aiosqlite

logger = logging.getLogger("Personalization")

class UserTier(Enum):
    NEW = "new"
    REGULAR = "regular"
    FRIEND = "friend"
    BESTIE = "bestie"

class CommunicationStyle(Enum):
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    PLAYFUL = "playful"
    EMPATHETIC = "empathetic"

def _now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

class PersonalizationManager:
    def __init__(self, db_path: str = "user_profiles.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._writer_sem = asyncio.Semaphore(1)

    async def aopen(self) -> "PersonalizationManager":
        self._conn = await aiosqlite.connect(self.db_path, isolation_level=None)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.execute("PRAGMA busy_timeout=15000;")
        await self._ensure_schema()
        return self

    async def aclose(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    def _ensure_open(self) -> aiosqlite.Connection:
        if not self._conn:
            raise RuntimeError("Call await aopen() first")
        return self._conn

    async def _ensure_schema(self) -> None:
        conn = self._ensure_open()
        # users
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                interaction_count INTEGER DEFAULT 0,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                user_tier TEXT DEFAULT 'new',
                communication_style TEXT DEFAULT 'friendly',
                favorite_topics TEXT DEFAULT '[]',
                disliked_topics TEXT DEFAULT '[]',
                emotional_bonds REAL DEFAULT 0.0,
                custom_nickname TEXT,
                personality_traits TEXT DEFAULT '{}'
            )
        """)
        # interaction history
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS interaction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                message_type TEXT NOT NULL, -- 'user'/'assistant'
                content TEXT NOT NULL,
                emotion_detected TEXT,
                response_emotion TEXT,
                satisfaction_score REAL DEFAULT 0.0
            )
        """)
        await conn.commit()

    # ---------------- Utility ----------------
    def get_user_id(self, username: str, platform: str = "voice") -> str:
        return hashlib.sha256(f"{platform}:{username}".encode("utf-8")).hexdigest()

    async def get_user_profile(self, user_id: str) -> Dict:
        """Fetch or create default profile."""
        conn = self._ensure_open()
        cur = await conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        await cur.close()
        if row:
            cols = [c[0] for c in (await conn.execute("PRAGMA table_info(users)")).description] if False else None
            # reconstruct manual mapping
            # row order as created above
            (uid, username, interaction_count, first_seen, last_seen, user_tier,
             communication_style, favorite_topics, disliked_topics,
             emotional_bonds, custom_nickname, personality_traits) = row
            def _loads(s, fallback):
                try:
                    return json.loads(s) if isinstance(s, str) else (s or fallback)
                except Exception:
                    return fallback
            return {
                "user_id": uid,
                "username": username,
                "interaction_count": interaction_count or 0,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "user_tier": user_tier or UserTier.NEW.value,
                "communication_style": communication_style or CommunicationStyle.FRIENDLY.value,
                "favorite_topics": _loads(favorite_topics, []),
                "disliked_topics": _loads(disliked_topics, []),
                "emotional_bonds": float(emotional_bonds or 0.0),
                "custom_nickname": custom_nickname,
                "personality_traits": _loads(personality_traits, {}),
            }
        # create default
        now = _now_str()
        profile = {
            "user_id": user_id,
            "username": user_id[:8],
            "interaction_count": 0,
            "first_seen": now,
            "last_seen": now,
            "user_tier": UserTier.NEW.value,
            "communication_style": CommunicationStyle.FRIENDLY.value,
            "favorite_topics": [],
            "disliked_topics": [],
            "emotional_bonds": 0.0,
            "custom_nickname": None,
            "personality_traits": {},
        }
        async with self._writer_sem:
            await conn.execute("""
                INSERT INTO users (
                    user_id, username, interaction_count, first_seen, last_seen, user_tier,
                    communication_style, favorite_topics, disliked_topics, emotional_bonds,
                    custom_nickname, personality_traits
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile["user_id"], profile["username"], profile["interaction_count"],
                profile["first_seen"], profile["last_seen"], profile["user_tier"],
                profile["communication_style"], json.dumps(profile["favorite_topics"]),
                json.dumps(profile["disliked_topics"]), profile["emotional_bonds"],
                profile["custom_nickname"], json.dumps(profile["personality_traits"]),
            ))
            await conn.commit()
        return profile

    async def _log_interaction(self, user_id: str, user_message: str,
                               bot_response: str, emotion: str, satisfaction: float) -> None:
        conn = self._ensure_open()
        async with self._writer_sem:
            await conn.execute("""
                INSERT INTO interaction_history
                (user_id, timestamp, message_type, content, emotion_detected, response_emotion, satisfaction_score)
                VALUES (?, ?, 'user', ?, ?, NULL, NULL)
            """, (user_id, _now_str(), user_message, emotion))
            await conn.execute("""
                INSERT INTO interaction_history
                (user_id, timestamp, message_type, content, emotion_detected, response_emotion, satisfaction_score)
                VALUES (?, ?, 'assistant', ?, NULL, ?, ?)
            """, (user_id, _now_str(), bot_response, emotion, satisfaction))
            await conn.commit()

    # ---------------- Public High-level ----------------
    async def update_interaction(self, user_id: str,
                                 user_message: str, bot_response: str, emotion: str) -> None:
        profile = await self.get_user_profile(user_id)
        topics = self._extract_topics(user_message)
        traits = self._analyze_personality(user_message)
        satisfaction = self._estimate_satisfaction(user_message, bot_response)
        # Update profile
        conn = self._ensure_open()
        profile["interaction_count"] = int(profile.get("interaction_count", 0)) + 1
        profile["last_seen"] = _now_str()
        # mutate aggregates
        fav = set(profile.get("favorite_topics", [])) | set(topics["likes"])
        dis = set(profile.get("disliked_topics", [])) | set(topics["dislikes"])
        profile["favorite_topics"] = list(fav)
        profile["disliked_topics"] = list(dis)
        merged_traits = dict(profile.get("personality_traits", {}))
        merged_traits.update(traits)
        profile["personality_traits"] = merged_traits
        # bonds heuristic
        delta = 0.1 if satisfaction >= 0 else -0.1
        profile["emotional_bonds"] = max(0.0, min(1.0, float(profile.get("emotional_bonds", 0.0)) + delta))
        async with self._writer_sem:
            await conn.execute("""
                UPDATE users
                SET interaction_count=?, last_seen=?, favorite_topics=?, disliked_topics=?,
                    personality_traits=?, emotional_bonds=?, communication_style=?, user_tier=?
                WHERE user_id=?
            """, (
                profile["interaction_count"], profile["last_seen"],
                json.dumps(profile["favorite_topics"]), json.dumps(profile["disliked_topics"]),
                json.dumps(profile["personality_traits"]), profile["emotional_bonds"],
                self._derive_style(profile), self._tier(profile),
                user_id
            ))
            await conn.commit()
        await self._log_interaction(user_id, user_message, bot_response, emotion, satisfaction)

    async def get_personalized_system_prompt(self, user_id: str, base_prompt: str) -> str:
        profile = await self.get_user_profile(user_id)
        ctx = self._build_personalization_context(profile)
        mods = self._get_style_modifiers(profile)
        return f"""{base_prompt}

# Persona
{ctx}

# Style modifiers
{mods}
"""

    # ---------------- Heuristics (left from original behavior) ----------------
    def _extract_topics(self, message: str) -> Dict[str, List[str]]:
        message = (message or "").lower()
        likes, dislikes = [], []
        for kw in ["music", "anime", "game", "coding", "python", "ai"]:
            if kw in message: likes.append(kw)
        for kw in ["ads", "spam", "lag"]:
            if kw in message: dislikes.append(kw)
        return {"likes": likes, "dislikes": dislikes}

    def _analyze_personality(self, message: str) -> Dict[str, float]:
        message = (message or "").lower()
        traits = {}
        traits["curious"] = 1.0 if "why" in message or "почему" in message else 0.4
        traits["humor"] = 1.0 if "lol" in message or "аха" in message else 0.3
        return traits

    def _estimate_satisfaction(self, user_message: str, bot_response: str) -> float:
        # very naive proxy
        score = 0.2
        for s in ("спасибо", "thanks", "great", "класс"): score += 0.2
        for s in ("плохо", "bad", "ужас"): score -= 0.2
        return max(-1.0, min(1.0, score))

    def _derive_style(self, profile: Dict) -> str:
        bonds = float(profile.get("emotional_bonds", 0.0))
        if bonds > 0.7: return CommunicationStyle.PLAYFUL.value
        if bonds > 0.4: return CommunicationStyle.FRIENDLY.value
        return CommunicationStyle.PROFESSIONAL.value

    def _tier(self, profile: Dict) -> str:
        cnt = int(profile.get("interaction_count", 0))
        bonds = float(profile.get("emotional_bonds", 0.0))
        if cnt >= 30 or bonds > 0.8: return UserTier.BESTIE.value
        if cnt >= 15 or bonds > 0.6: return UserTier.FRIEND.value
        if cnt >= 5 or bonds > 0.3: return UserTier.REGULAR.value
        return UserTier.NEW.value

    def _build_personalization_context(self, profile: Dict) -> str:
        return json.dumps({
            "username": profile.get("username"),
            "nickname": profile.get("custom_nickname"),
            "likes": profile.get("favorite_topics", []),
            "dislikes": profile.get("disliked_topics", []),
            "traits": profile.get("personality_traits", {}),
            "tier": profile.get("user_tier"),
            "style": profile.get("communication_style"),
        }, ensure_ascii=False, indent=2)

    def _get_style_modifiers(self, profile: Dict) -> str:
        style = profile.get("communication_style", CommunicationStyle.FRIENDLY.value)
        if style == CommunicationStyle.PLAYFUL.value:
            return "- Use playful, energetic tone\n- Add light humor when appropriate"
        if style == CommunicationStyle.PROFESSIONAL.value:
            return "- Be concise and clear\n- Keep a calm, helpful tone"
        return "- Be warm and friendly\n- Mirror user's vibe lightly"

    async def save_user_profile(self, user_id: str, profile: Dict) -> None:
        """
        Сохраняет обновлённый профиль пользователя в БД.
        
        Args:
            user_id: идентификатор пользователя
            profile: словарь с данными профиля
        """
        conn = self._ensure_open()
        
        async with self._writer_sem:
            await conn.execute("""
                UPDATE users
                SET username=?, interaction_count=?, last_seen=?, user_tier=?,
                    communication_style=?, favorite_topics=?, disliked_topics=?,
                    emotional_bonds=?, custom_nickname=?, personality_traits=?
                WHERE user_id=?
            """, (
                profile.get("username"),
                profile.get("interaction_count", 0),
                profile.get("last_seen"),
                profile.get("user_tier"),
                profile.get("communication_style"),
                json.dumps(profile.get("favorite_topics", [])),
                json.dumps(profile.get("disliked_topics", [])),
                profile.get("emotional_bonds", 0.0),
                profile.get("custom_nickname"),
                json.dumps(profile.get("personality_traits", {})),
                user_id
            ))
            await conn.commit()
            logger.debug(f"💾 Профиль сохранён: {user_id[:8]}...")


# Singleton helpers from original
_PM_SINGLETON: PersonalizationManager | None = None

def _get_pm_singleton() -> PersonalizationManager:
    global _PM_SINGLETON
    if _PM_SINGLETON is None:
        _PM_SINGLETON = PersonalizationManager()
    return _PM_SINGLETON

async def apply_personalized_prompt(base_prompt: str, username: str, platform: str = "voice") -> str:
    pm = _get_pm_singleton()
    if pm._conn is None:
        await pm.aopen()
    user_id = pm.get_user_id(username, platform=platform)
    return await pm.get_personalized_system_prompt(user_id, base_prompt)

async def log_after_dialog(username: str, user_message: str, bot_response: str, emotion: str = "neutral", platform: str = "voice") -> None:
    pm = _get_pm_singleton()
    if pm._conn is None:
        await pm.aopen()
    user_id = pm.get_user_id(username, platform=platform)
    await pm.update_interaction(user_id, user_message, bot_response, emotion)
