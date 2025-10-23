# -*- coding: utf-8 -*-
"""
llm/ollama_client.py — patched cache with aiosqlite (async, safe)
- Replaces sqlite3 + to_thread with aiosqlite
- Serialized writes via asyncio.Semaphore(1)
"""

import aiohttp
import asyncio
import aiosqlite
import hashlib
import json
import logging
from typing import Optional, AsyncGenerator

from core.config import load_config

logger = logging.getLogger("Ollama")

class AsyncCache:
    def __init__(self, db_path: str = "ollama_cache.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._writer_sem = asyncio.Semaphore(1)
        self._init_done = False

    async def _ensure_init(self) -> None:
        if self._init_done and self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self.db_path, isolation_level=None)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.execute("PRAGMA busy_timeout=5000;")
        await self._conn.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        await self._conn.commit()
        self._init_done = True

    def _make_key(self, prompt: str, system: Optional[str], params: dict) -> str:
        raw = json.dumps({"p": prompt, "s": system or "", "params": params}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get(self, prompt: str, system: Optional[str], params: dict) -> Optional[str]:
        await self._ensure_init()
        key = self._make_key(prompt, system, params)
        cur = await self._conn.execute("SELECT response FROM cache WHERE key=?", (key,))
        row = await cur.fetchone()
        await cur.close()
        return row[0] if row else None

    async def set(self, prompt: str, system: Optional[str], params: dict, response: str) -> None:
        await self._ensure_init()
        key = self._make_key(prompt, system, params)
        async with self._writer_sem:
            await self._conn.execute("INSERT OR REPLACE INTO cache VALUES(?, ?)", (key, response))
            await self._conn.commit()

    async def close(self):
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._init_done = False


# ------------------- Ollama Client (kept same interface) -------------------
class OptimizedOllamaClient:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 120):
        cfg = load_config()
        self.base_url = base_url or cfg.get("ollama", {}).get("base_url", "http://localhost:11434")
        self.model = model or cfg.get("ollama", {}).get("model", "llama3")
        self.timeout = timeout
        self.cache = AsyncCache(cfg.get("ollama", {}).get("cache_path", "ollama_cache.db"))

    async def generate(self, prompt: str, system: Optional[str] = None, params: Optional[dict] = None) -> str:
        params = params or {}
        cached = await self.cache.get(prompt, system, params)
        if cached is not None:
            return cached
        payload = {"model": self.model, "prompt": prompt, "system": system or "", **params}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as sess:
            async with sess.post(f"{self.base_url}/api/generate", json=payload) as r:
                r.raise_for_status()
                data = await r.json()
        text = data.get("response", "")
        await self.cache.set(prompt, system, params, text)
        return text

    async def stream(self, prompt: str, system: Optional[str] = None, params: Optional[dict] = None) -> AsyncGenerator[str, None]:
        params = params or {}
        payload = {"model": self.model, "prompt": prompt, "system": system or "", **params}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as sess:
            async with sess.post(f"{self.base_url}/api/generate", json=payload) as r:
                r.raise_for_status()
                async for line in r.content:
                    if not line:
                        continue
                    try:
                        piece = json.loads(line.decode("utf-8")).get("response", "")
                        if piece:
                            yield piece
                    except Exception:
                        continue

    async def close(self):
        await self.cache.close()
