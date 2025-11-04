# -*- coding: utf-8 -*-
"""
HybridMemory (async, production-ready) — Python 3.14+
ПАТЧ v1.1: Оптимизация векторного поиска (решение проблемы утечки памяти)

Изменения:
- ✅ Добавлен prefilter_limit для ограничения выборки
- ✅ Использование heapq.nlargest() вместо полной сортировки
- ✅ Добавлен индекс created_at для быстрой фильтрации
- ✅ Явная очистка памяти после поиска

- Uses aiosqlite (fully async, thread-safe)
- Serializes all writes with asyncio.Semaphore(1)
- Connection pool for parallel reads
- WAL mode + optimized indexes
- Retry logic for transient errors
- Graceful shutdown with WAL checkpoint

Public API:
    class HybridMemory:
        async def aopen(self, db_path: str = "memory.db", embed_dim: int = 512): ...
        async def aclose(self) -> None: ...
        async def add_turn(self, role: str, text: str, meta: dict | None = None) -> None: ...
        async def add_fact(self, key: str, value: str, meta: dict | None = None) -> None: ...
        async def recall(self, query: str, k: int = 5) -> list[dict]: ...
        async def context(self, last_n_turns: int = 20, max_facts: int = 100) -> dict: ...
        async def clear_short_term(self) -> None: ...
"""
from __future__ import annotations

import asyncio
import heapq  # ✅ ПАТЧ: Добавлен для оптимизации top-k
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
    np = None  # type: ignore

logger = logging.getLogger("HybridMemory")

# --------------------------- Utils & Embeddings -------------------------------

def _utcnow() -> str:
    """ISO8601 timestamp (UTC)"""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


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
    """
    Lightweight vector store inside SQLite (`vectors` table).
    
    ✅ ПАТЧ v1.1: Оптимизирован метод search() для предотвращения утечки памяти
    - Добавлен prefilter_limit для ограничения выборки
    - Использование heapq.nlargest() вместо полной сортировки
    - Добавлен индекс created_at для быстрой фильтрации по времени
    """
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
              created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
            """
        )
        # Index for faster searches (though we do full scan anyway)
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vectors_id ON vectors(id)"
        )
        # ✅ ПАТЧ: Добавлен индекс для быстрой фильтрации по времени
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vectors_created ON vectors(created_at DESC)"
        )
        await self.conn.commit()

    async def add_no_commit(self, rec_id: str, text: str, meta: dict, vec: list[float]) -> None:
        """Add vector without committing (for batch operations)"""
        await self.conn.execute(
            "INSERT OR REPLACE INTO vectors(id, text, meta, vec) VALUES(?, ?, ?, ?)",
            (rec_id, text, json.dumps(meta, ensure_ascii=False), json.dumps(vec)),
        )

    async def add(self, rec_id: str, text: str, meta: dict, vec: list[float]) -> None:
        """Add vector with immediate commit"""
        await self.add_no_commit(rec_id, text, meta, vec)
        await self.conn.commit()

    async def search(
        self, 
        query_vec: list[float], 
        k: int = 5,
        prefilter_limit: int = 10000  # ✅ ПАТЧ: Ограничение предварительной выборки
    ) -> list[dict]:
        """
        Semantic search using cosine similarity with memory optimization.
        
        ✅ ПАТЧ v1.1: Оптимизирован для предотвращения утечки памяти
        
        Стратегия:
        1. Pre-filtering через SQL LIMIT (берём только N свежих записей)
        2. Вычисление cosine similarity только для отобранных
        3. Использование heapq.nlargest() вместо полной сортировки
        4. Явная очистка памяти после операции
        
        Args:
            query_vec: Вектор запроса
            k: Количество результатов
            prefilter_limit: Максимум записей для обработки (защита от OOM)
        
        Returns:
            Список результатов с полями: id, text, meta, score
        """
        # ✅ ПАТЧ: Используем ORDER BY created_at DESC LIMIT для pre-filtering
        # Это берёт только N свежих записей вместо всех
        cur = await self.conn.execute(
            """
            SELECT id, text, meta, vec 
            FROM vectors 
            ORDER BY created_at DESC 
            LIMIT ?
            """,
            (prefilter_limit,)
        )
        rows = await cur.fetchall()
        await cur.close()
        
        if not rows:
            return []
        
        # ✅ ПАТЧ: Вычисляем similarity только для отобранных записей
        scored = []
        for rid, text, meta_s, vec_s in rows:
            try:
                v = json.loads(vec_s)
            except Exception:
                continue
            score = _cosine(query_vec, v)
            scored.append((score, rid, text, meta_s))
        
        # ✅ ПАТЧ: Используем heapq.nlargest() вместо полной сортировки
        # O(n log k) вместо O(n log n)
        top_k = heapq.nlargest(k, scored, key=lambda x: x[0])
        
        # Формируем результат
        out = []
        for score, rid, text, meta_s in top_k:
            try:
                meta = json.loads(meta_s) if isinstance(meta_s, str) else (meta_s or {})
            except Exception:
                meta = {}
            out.append({"id": rid, "text": text, "meta": meta, "score": float(score)})
        
        # ✅ ПАТЧ: Явная очистка памяти
        del scored
        del rows
        del top_k
        
        return out


# ------------------------------- HybridMemory ---------------------------------

class HybridMemory:
    """Production-ready async memory layer for VTuber pipeline."""
    
    def __init__(
        self,
        embed_dim: int = 512,
        embed_fn: Callable[[Iterable[str]], list[list[float]]] | None = None,
        read_pool_size: int = 3,  # concurrent read connections
    ) -> None:
        self.embed_dim = embed_dim
        self.embed_fn = embed_fn or SimpleHashEmbedder(embed_dim)
        self.read_pool_size = read_pool_size
        
        # Write connection (single, serialized)
        self._write_conn: aiosqlite.Connection | None = None
        self._writer_sem = asyncio.Semaphore(1)
        
        # Read connection pool
        self._read_pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        self._read_conns: list[aiosqlite.Connection] = []
        
        self._vstore: SQLiteVectorStore | None = None
        self._db_path: str = ""
        self._closed = False

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
        
        logger.info(f"✅ HybridMemory opened: {db_path} (pool size: {self.read_pool_size})")
        return self

    async def aclose(self) -> None:
        """Graceful shutdown with WAL checkpoint"""
        if self._closed:
            return
        
        self._closed = True
        
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
        
        logger.info("✅ HybridMemory closed gracefully")

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
                    wait_time = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s
                    logger.warning(f"Database locked, retry {attempt+1}/{retries} in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        raise RuntimeError("Should not reach here")  # pragma: no cover

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
        """
        Semantic recall from vector store with recent turns boost.
        
        ✅ ПАТЧ v1.1: Оптимизирован для работы с большими объёмами данных
        - Ограничение предварительной выборки (prefilter_limit=10000)
        - Эффективная сортировка через heapq
        """
        self._ensure_open()
        
        # Embed query
        emb = self.embed_fn([query])[0]
        
        # ✅ ПАТЧ: Vector search с лимитом предварительной выборки
        assert self._vstore is not None
        candidates = await self._vstore.search(
            emb, 
            k=k * 2,
            prefilter_limit=10000  # ✅ Обрабатываем максимум 10k записей
        )
        
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
        
        # ✅ ПАТЧ: Повторная сортировка только top-k кандидатов (быстро)
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


# ----------------------------- Self-test --------------------------------------

if __name__ == "__main__":
    async def _demo():
        import sys
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        
        mem = HybridMemory(read_pool_size=2)
        await mem.aopen(":memory:")
        
        # Add dialogue
        await mem.add_turn("user", "Привет! Как дела?", {"lang": "ru"})
        await mem.add_fact("bot_name", "Aiko")
        await mem.add_turn("assistant", "Привет! Всё отлично, спасибо!", {})
        await mem.add_turn("user", "Как тебя зовут?", {})
        
        # Test context
        ctx = await mem.context(last_n_turns=10)
        print("\n=== Context ===")
        print(f"Turns: {len(ctx['turns'])}")
        print(f"Facts: {len(ctx['facts'])}")
        
        # Test recall
        results = await mem.recall("имя бота", k=3)
        print("\n=== Recall ===")
        for r in results:
            print(f"[{r['score']:.3f}] {r['text'][:50]}")
        
        # Graceful shutdown
        await mem.aclose()
        print("\n✅ Demo completed")

    asyncio.run(_demo())
