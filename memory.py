# -*- coding: utf-8 -*-
"""
HybridMemory v2 — с автоматической ротацией и лимитами
+ Кратковременная память (последние N реплик)
+ Среднесрочная память (последние M дней)
+ Долговременная память (факты) — никогда не удаляется
+ Автоматическая очистка старых записей
"""
from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable
import logging

import aiosqlite

try:
    import numpy as np
except Exception:
    np = None

logger = logging.getLogger("HybridMemory")

# ======================== КОНФИГУРАЦИЯ ПАМЯТИ ========================

@dataclass
class MemoryLimits:
    """Лимиты для управления размером памяти"""
    
    # Кратковременная память (turns)
    max_short_term_turns: int = 100  # Максимум реплик в активной памяти
    
    # Среднесрочная память (vectors)
    max_vector_age_days: int = 7  # Удалять векторы старше N дней
    max_vectors: int = 10_000  # Абсолютный лимит векторов
    
    # Авто-очистка
    auto_cleanup_enabled: bool = True
    cleanup_interval_sec: float = 3600.0  # Каждый час
    
    # Сжатие БД
    auto_vacuum_enabled: bool = True
    vacuum_interval_sec: float = 86400.0  # Раз в сутки


# --------------------------- Utils & Embeddings -------------------------------

def _utcnow() -> str:
    """ISO8601 timestamp (UTC)"""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _parse_timestamp(ts: str) -> float:
    """Парсинг ISO8601 в Unix timestamp"""
    try:
        return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return time.time()


class SimpleHashEmbedder:
    """Fast, deterministic embedding for local usage/testing."""
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def __call__(self, texts: Iterable[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            for i, ch in enumerate(t):
                v[i % self.dim] += (ord(ch) % 53) / 53.0
            # L2 normalize
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors"""
    if np is not None:
        va = np.asarray(a, dtype=float)
        vb = np.asarray(b, dtype=float)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
        return float(va.dot(vb) / denom)
    # pure python fallback
    num = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return num / (na * nb)


# ------------------------------ Vector Store ----------------------------------

@dataclass(slots=True)
class VectorRecord:
    id: str
    text: str
    meta: dict
    vec: list[float]


class SQLiteVectorStore:
    """Lightweight vector store inside SQLite (`vectors` table)."""
    def __init__(self, conn: aiosqlite.Connection, *, embed_dim: int = 512) -> None:
        self.conn = conn
        self.embed_dim = embed_dim

    async def ensure_schema(self) -> None:
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
              id   TEXT PRIMARY KEY,
              text TEXT NOT NULL,
              meta TEXT NOT NULL,
              vec  TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        # Индекс для фильтрации по времени
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vectors_created ON vectors(created_at DESC)"
        )
        await self.conn.commit()

    async def add_no_commit(self, rec_id: str, text: str, meta: dict, vec: list[float]) -> None:
        """Add vector without committing (for batch operations)"""
        await self.conn.execute(
            "INSERT OR REPLACE INTO vectors(id, text, meta, vec, created_at) VALUES(?, ?, ?, ?, ?)",
            (rec_id, text, json.dumps(meta, ensure_ascii=False), json.dumps(vec), _utcnow()),
        )

    async def add(self, rec_id: str, text: str, meta: dict, vec: list[float]) -> None:
        """Add vector with immediate commit"""
        await self.add_no_commit(rec_id, text, meta, vec)
        await self.conn.commit()

    async def search(self, query_vec: list[float], k: int = 5) -> list[dict]:
        """Semantic search using cosine similarity"""
        cur = await self.conn.execute("SELECT id, text, meta, vec FROM vectors")
        rows = await cur.fetchall()
        await cur.close()
        
        scored = []
        for rid, text, meta_s, vec_s in rows:
            try:
                v = json.loads(vec_s)
            except Exception:
                continue
            score = _cosine(query_vec, v)
            scored.append((score, rid, text, meta_s))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        out = []
        for score, rid, text, meta_s in scored[:k]:
            try:
                meta = json.loads(meta_s) if isinstance(meta_s, str) else (meta_s or {})
            except Exception:
                meta = {}
            out.append({"id": rid, "text": text, "meta": meta, "score": float(score)})
        
        # Memory cleanup
        del scored
        del rows
        
        return out

    async def cleanup_old(self, max_age_days: int) -> int:
        """Удалить векторы старше N дней"""
        cutoff = time.time() - (max_age_days * 86400)
        cutoff_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(cutoff))
        
        cur = await self.conn.execute(
            "DELETE FROM vectors WHERE created_at < ?",
            (cutoff_str,)
        )
        deleted = cur.rowcount
        await self.conn.commit()
        return deleted

    async def limit_count(self, max_count: int) -> int:
        """Оставить только N самых свежих векторов"""
        cur = await self.conn.execute(
            """
            DELETE FROM vectors WHERE id NOT IN (
                SELECT id FROM vectors ORDER BY created_at DESC LIMIT ?
            )
            """,
            (max_count,)
        )
        deleted = cur.rowcount
        await self.conn.commit()
        return deleted


# ------------------------------- HybridMemory ---------------------------------

class HybridMemory:
    """Production-ready async memory layer with auto-rotation."""
    
    def __init__(
        self,
        embed_dim: int = 512,
        embed_fn: Callable[[Iterable[str]], list[list[float]]] | None = None,
        read_pool_size: int = 3,
        limits: MemoryLimits | None = None,
    ) -> None:
        self.embed_dim = embed_dim
        self.embed_fn = embed_fn or SimpleHashEmbedder(embed_dim)
        self.read_pool_size = read_pool_size
        self.limits = limits or MemoryLimits()
        
        # Write connection (single, serialized)
        self._write_conn: aiosqlite.Connection | None = None
        self._writer_sem = asyncio.Semaphore(1)
        
        # Read connection pool
        self._read_pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        self._read_conns: list[aiosqlite.Connection] = []
        
        self._vstore: SQLiteVectorStore | None = None
        self._db_path: str = ""
        self._closed = False
        
        # Фоновые задачи
        self._cleanup_task: asyncio.Task | None = None
        self._vacuum_task: asyncio.Task | None = None

    # -------- Lifecycle --------

    async def aopen(self, db_path: str = "memory.db", embed_dim: int | None = None) -> "HybridMemory":
        """Open database and initialize connection pool"""
        if embed_dim:
            self.embed_dim = embed_dim
            if isinstance(self.embed_fn, SimpleHashEmbedder):
                self.embed_fn = SimpleHashEmbedder(embed_dim)
        
        self._db_path = db_path
        self._closed = False
        
        # Write connection
        self._write_conn = await aiosqlite.connect(db_path, isolation_level=None)
        await self._setup_pragmas(self._write_conn)
        await self._ensure_schema()
        
        # Vector store (uses write connection)
        self._vstore = SQLiteVectorStore(self._write_conn, embed_dim=self.embed_dim)
        await self._vstore.ensure_schema()
        
        # Read connection pool
        for _ in range(self.read_pool_size):
            conn = await aiosqlite.connect(db_path, isolation_level="DEFERRED")
            await self._setup_pragmas(conn, read_only=True)
            self._read_conns.append(conn)
            await self._read_pool.put(conn)
        
        # Запускаем фоновую очистку
        if self.limits.auto_cleanup_enabled:
            self._cleanup_task = asyncio.create_task(self._auto_cleanup_loop())
            logger.info(f"🧹 Автоочистка включена (каждые {self.limits.cleanup_interval_sec}s)")
        
        if self.limits.auto_vacuum_enabled:
            self._vacuum_task = asyncio.create_task(self._auto_vacuum_loop())
            logger.info(f"🗜️ Автосжатие включено (каждые {self.limits.vacuum_interval_sec}s)")
        
        logger.info(f"✅ HybridMemory v2 opened: {db_path} (pool size: {self.read_pool_size})")
        return self

    async def aclose(self) -> None:
        """Graceful shutdown with WAL checkpoint"""
        if self._closed:
            return
        
        self._closed = True
        
        # Останавливаем фоновые задачи
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
        
        if self._vacuum_task:
            self._vacuum_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._vacuum_task
        
        # Flush pending writes
        if self._write_conn:
            async with self._writer_sem:
                try:
                    await self._write_conn.commit()
                    # Checkpoint WAL to main database
                    await self._write_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                except Exception as e:
                    logger.warning(f"WAL checkpoint failed: {e}")
            
            await self._write_conn.close()
            self._write_conn = None
        
        # Close read pool
        for conn in self._read_conns:
            try:
                await conn.close()
            except Exception:
                pass
        self._read_conns.clear()
        
        logger.info("✅ HybridMemory v2 closed gracefully")

    async def _setup_pragmas(self, conn: aiosqlite.Connection, read_only: bool = False) -> None:
        """Configure SQLite for optimal performance"""
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await conn.execute("PRAGMA busy_timeout=5000;")
        await conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache
        
        if read_only:
            await conn.execute("PRAGMA query_only=ON;")
        else:
            await conn.execute("PRAGMA foreign_keys=ON;")

    # -------- Schema --------

    async def _ensure_schema(self) -> None:
        """Create tables with optimized indexes"""
        assert self._write_conn is not None
        
        # Turns table
        await self._write_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
              id TEXT PRIMARY KEY,
              role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
              text TEXT NOT NULL,
              meta TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        
        # Index for ORDER BY created_at DESC (recent turns)
        await self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at DESC)"
        )
        
        # Facts table
        await self._write_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              meta  TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        
        # Index for ORDER BY updated_at DESC (recent facts)
        await self._write_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_updated ON facts(updated_at DESC)"
        )
        
        await self._write_conn.commit()

    def _ensure_open(self) -> None:
        """Guard: ensure DB is open"""
        if self._closed or self._write_conn is None:
            raise RuntimeError("HybridMemory is closed. Call await aopen().")

    # -------- Read Connection Pool --------

    async def _acquire_reader(self) -> aiosqlite.Connection:
        """Get read connection from pool"""
        return await self._read_pool.get()

    async def _release_reader(self, conn: aiosqlite.Connection) -> None:
        """Return read connection to pool"""
        await self._read_pool.put(conn)

    # -------- Retry Logic --------

    async def _execute_with_retry(
        self, 
        conn: aiosqlite.Connection,
        query: str, 
        params: tuple = (),
        retries: int = 3
    ) -> aiosqlite.Cursor:
        """Execute query with exponential backoff on 'database locked' errors"""
        for attempt in range(retries):
            try:
                return await conn.execute(query, params)
            except aiosqlite.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < retries - 1:
                    wait_time = 0.1 * (2 ** attempt)
                    logger.warning(f"Database locked, retry {attempt+1}/{retries} in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        raise RuntimeError("Should not reach here")

    # -------- Write Operations (Serialized) --------

    async def add_turn(self, role: str, text: str, meta: dict | None = None) -> None:
        """Append a dialogue turn and index it in vectors (atomic transaction)"""
        self._ensure_open()
        
        role = role.lower()
        assert role in {"user", "assistant", "system"}, f"Invalid role: {role}"
        meta = meta or {}
        rec_id = str(uuid.uuid4())
        created_at = _utcnow()
        
        # Single transaction for both turn + vector
        async with self._writer_sem:
            try:
                # Insert turn
                await self._execute_with_retry(
                    self._write_conn,
                    "INSERT INTO turns(id, role, text, meta, created_at) VALUES(?, ?, ?, ?, ?)",
                    (rec_id, role, text, json.dumps(meta, ensure_ascii=False), created_at),
                )
                
                # Index to vectors (no commit yet)
                emb = self.embed_fn([text])[0]
                assert self._vstore is not None
                await self._vstore.add_no_commit(rec_id, text, {"type": "turn", **meta}, emb)
                
                # Single commit for both operations
                await self._write_conn.commit()
                
                # Проверка лимита кратковременной памяти
                await self._enforce_short_term_limit()
                
            except Exception as e:
                logger.error(f"Failed to add turn: {e}")
                await self._write_conn.rollback()
                raise

    async def add_fact(self, key: str, value: str, meta: dict | None = None) -> None:
        """Upsert a key-value fact and index to vectors (atomic transaction)"""
        self._ensure_open()
        
        meta = meta or {}
        updated_at = _utcnow()
        
        async with self._writer_sem:
            try:
                # Upsert fact
                await self._execute_with_retry(
                    self._write_conn,
                    """
                    INSERT INTO facts(key, value, meta, updated_at)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                      value=excluded.value,
                      meta=excluded.meta,
                      updated_at=excluded.updated_at
                    """,
                    (key, value, json.dumps(meta, ensure_ascii=False), updated_at),
                )
                
                # Index to vectors
                emb = self.embed_fn([value])[0]
                assert self._vstore is not None
                await self._vstore.add_no_commit(f"fact:{key}", value, {"type": "fact", **meta}, emb)
                
                # Single commit
                await self._write_conn.commit()
                
            except Exception as e:
                logger.error(f"Failed to add fact '{key}': {e}")
                await self._write_conn.rollback()
                raise

    # -------- Read Operations (Parallel via Pool) --------

    async def recall(self, query: str, k: int = 5) -> list[dict]:
        """Semantic recall from vector store with recent turns boost"""
        self._ensure_open()
        
        # Embed query
        emb = self.embed_fn([query])[0]
        
        # Vector search (uses write connection, but read-only operation)
        assert self._vstore is not None
        candidates = await self._vstore.search(emb, k=k * 2)
        
        # Boost recent turns (parallel read from pool)
        reader = await self._acquire_reader()
        try:
            cur = await self._execute_with_retry(
                reader,
                "SELECT id FROM turns ORDER BY created_at DESC LIMIT 20"
            )
            rows = await cur.fetchall()
            await cur.close()
            last_ids = {row[0] for row in rows}
        finally:
            await self._release_reader(reader)
        
        # Apply boost
        for c in candidates:
            if c["id"] in last_ids:
                c["score"] = float(c.get("score", 0.0)) + 0.05
        
        candidates.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        result = candidates[:k]
        
        # Memory cleanup
        del candidates
        del last_ids
        
        return result

    async def context(self, last_n_turns: int = 20, max_facts: int = 100) -> dict:
        """Return recent dialogue context and current facts (parallel read)"""
        self._ensure_open()
        
        reader = await self._acquire_reader()
        try:
            # Recent turns
            cur = await self._execute_with_retry(
                reader,
                "SELECT role, text, meta, created_at FROM turns ORDER BY created_at DESC LIMIT ?",
                (last_n_turns,),
            )
            turn_rows = await cur.fetchall()
            await cur.close()
            
            turns = []
            for role, text, meta_s, created_at in reversed(turn_rows):
                try:
                    m = json.loads(meta_s) if isinstance(meta_s, str) else (meta_s or {})
                except Exception:
                    m = {}
                turns.append({
                    "role": role, 
                    "text": text, 
                    "meta": m, 
                    "created_at": created_at
                })
            
            # Facts
            cur = await self._execute_with_retry(
                reader,
                "SELECT key, value, meta, updated_at FROM facts ORDER BY updated_at DESC LIMIT ?",
                (max_facts,),
            )
            fact_rows = await cur.fetchall()
            await cur.close()
            
            facts = []
            for key, value, meta_s, updated_at in fact_rows:
                try:
                    m = json.loads(meta_s) if isinstance(meta_s, str) else (meta_s or {})
                except Exception:
                    m = {}
                facts.append({
                    "key": key, 
                    "value": value, 
                    "meta": m, 
                    "updated_at": updated_at
                })
            
            # Cleanup
            del turn_rows
            del fact_rows
            
            return {"turns": turns, "facts": facts}
            
        finally:
            await self._release_reader(reader)

    async def clear_short_term(self) -> None:
        """Clear episodic buffer (turns only, keep facts)"""
        self._ensure_open()
        
        async with self._writer_sem:
            try:
                await self._execute_with_retry(
                    self._write_conn,
                    "DELETE FROM turns"
                )
                await self._write_conn.commit()
                logger.info("✅ Short-term memory cleared")
            except Exception as e:
                logger.error(f"Failed to clear short-term memory: {e}")
                await self._write_conn.rollback()
                raise

    # -------- Auto-cleanup Logic --------

    async def _enforce_short_term_limit(self) -> None:
        """Удалить старые реплики, если превышен лимит"""
        if self.limits.max_short_term_turns <= 0:
            return
        
        try:
            cur = await self._write_conn.execute("SELECT COUNT(*) FROM turns")
            row = await cur.fetchone()
            await cur.close()
            count = row[0] if row else 0
            
            if count > self.limits.max_short_term_turns:
                excess = count - self.limits.max_short_term_turns
                await self._write_conn.execute(
                    """
                    DELETE FROM turns WHERE id IN (
                        SELECT id FROM turns ORDER BY created_at ASC LIMIT ?
                    )
                    """,
                    (excess,)
                )
                await self._write_conn.commit()
                logger.info(f"🧹 Удалено {excess} старых реплик (лимит: {self.limits.max_short_term_turns})")
        except Exception as e:
            logger.error(f"Ошибка очистки turns: {e}")

    async def _auto_cleanup_loop(self) -> None:
        """Фоновая задача автоочистки"""
        while not self._closed:
            try:
                await asyncio.sleep(self.limits.cleanup_interval_sec)
                
                if self._closed:
                    break
                
                logger.info("🧹 Запуск автоочистки памяти...")
                
                # 1. Очистка векторов по возрасту
                deleted_by_age = await self._vstore.cleanup_old(self.limits.max_vector_age_days)
                if deleted_by_age > 0:
                    logger.info(f"   ✅ Удалено {deleted_by_age} векторов старше {self.limits.max_vector_age_days} дней")
                
                # 2. Очистка векторов по количеству
                deleted_by_count = await self._vstore.limit_count(self.limits.max_vectors)
                if deleted_by_count > 0:
                    logger.info(f"   ✅ Удалено {deleted_by_count} векторов (лимит: {self.limits.max_vectors})")
                
                # 3. Очистка старых turns
                await self._enforce_short_term_limit()
                
                logger.info("✅ Автоочистка завершена")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка автоочистки: {e}")

    async def _auto_vacuum_loop(self) -> None:
        """Фоновая задача сжатия БД"""
        while not self._closed:
            try:
                await asyncio.sleep(self.limits.vacuum_interval_sec)
                
                if self._closed:
                    break
                
                logger.info("🗜️ Запуск сжатия БД...")
                
                async with self._writer_sem:
                    await self._write_conn.execute("VACUUM;")
                    await self._write_conn.commit()
                
                logger.info("✅ Сжатие БД завершено")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка сжатия БД: {e}")

    # -------- Statistics --------

    async def get_stats(self) -> dict:
        """Статистика памяти"""
        reader = await self._acquire_reader()
        try:
            # Количество реплик
            cur = await reader.execute("SELECT COUNT(*) FROM turns")
            turn_count = (await cur.fetchone())[0]
            await cur.close()
            
            # Количество фактов
            cur = await reader.execute("SELECT COUNT(*) FROM facts")
            fact_count = (await cur.fetchone())[0]
            await cur.close()
            
            # Количество векторов
            cur = await reader.execute("SELECT COUNT(*) FROM vectors")
            vector_count = (await cur.fetchone())[0]
            await cur.close()
            
            # Размер БД
            import os
            db_size_mb = os.path.getsize(self._db_path) / (1024 * 1024) if os.path.exists(self._db_path) else 0
            
            return {
                "turns": turn_count,
                "facts": fact_count,
                "vectors": vector_count,
                "db_size_mb": round(db_size_mb, 2),
                "limits": {
                    "max_turns": self.limits.max_short_term_turns,
                    "max_vectors": self.limits.max_vectors,
                    "max_vector_age_days": self.limits.max_vector_age_days,
                },
            }
        finally:
            await self._release_reader(reader)


# ----------------------------- Self-test --------------------------------------

if __name__ == "__main__":
    import contextlib
    
    async def _demo():
        import sys
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        
        # Создаём память с лимитами
        limits = MemoryLimits(
            max_short_term_turns=5,  # Только 5 последних реплик
            max_vector_age_days=1,   # Удалять векторы старше 1 дня
            max_vectors=10,          # Максимум 10 векторов
            auto_cleanup_enabled=False,  # Отключаем для теста
        )
        
        mem = HybridMemory(read_pool_size=2, limits=limits)
        await mem.aopen(":memory:")
        
        # Добавляем 10 реплик (должно остаться только 5)
        for i in range(10):
            await mem.add_turn("user", f"Сообщение {i}", {})
        
        # Проверяем статистику
        stats = await mem.get_stats()
        print(f"\n=== Статистика после добавления 10 реплик ===")
        print(f"Turns: {stats['turns']} (ожидается 5)")
        print(f"Vectors: {stats['vectors']}")
        
        # Добавляем факт
        await mem.add_fact("bot_name", "Aiko")
        
        # Проверяем контекст
        ctx = await mem.context(last_n_turns=10)
        print(f"\n=== Контекст ===")
        print(f"Turns: {len(ctx['turns'])}")
        print(f"Facts: {len(ctx['facts'])}")
        
        # Graceful shutdown
        await mem.aclose()
        print("\n✅ Demo completed")

    asyncio.run(_demo())
