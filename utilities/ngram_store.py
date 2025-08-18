from __future__ import annotations

import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Protocol, Tuple

Row = Tuple[int, int, int, int]


class BaseNGramStore(Protocol):
    def inc(self, ctx_hash: int, aux_id: int, next_event: int, n: int = 1) -> None: ...
    def bulk_inc(self, rows: Iterable[Row]) -> None: ...
    def topk(self, ctx_hash: int, aux_id: int, k: int = 8) -> List[Tuple[int, int]]: ...
    def prune(self, min_count: int) -> None: ...
    def finalize(self) -> None: ...
    def stats(self) -> Dict[str, int]: ...


class MemoryNGramStore:
    """Simple in-memory implementation using nested counters."""

    def __init__(self) -> None:
        self.data: defaultdict[Tuple[int, int], Counter] = defaultdict(Counter)

    def inc(self, ctx_hash: int, aux_id: int, next_event: int, n: int = 1) -> None:
        self.data[(ctx_hash, aux_id)][next_event] += n

    def bulk_inc(self, rows: Iterable[Row]) -> None:
        for ctx_hash, aux_id, next_evt, cnt in rows:
            self.data[(ctx_hash, aux_id)][next_evt] += cnt

    def topk(self, ctx_hash: int, aux_id: int, k: int = 8) -> List[Tuple[int, int]]:
        counter = self.data.get((ctx_hash, aux_id))
        if not counter:
            return []
        return counter.most_common(k)

    def prune(self, min_count: int) -> None:
        for key in list(self.data.keys()):
            counter = self.data[key]
            for evt, cnt in list(counter.items()):
                if cnt < min_count:
                    del counter[evt]
            if not counter:
                del self.data[key]

    def finalize(self) -> None:  # pragma: no cover - nothing to do
        pass

    def stats(self) -> Dict[str, int]:
        rows = sum(len(c) for c in self.data.values())
        return {"rows": rows, "contexts": len(self.data)}

    # Extra helper for training code
    def iter_rows(self) -> Iterable[Row]:
        for (ctx_hash, aux_id), counter in self.data.items():
            for next_evt, cnt in counter.items():
                yield ctx_hash, aux_id, next_evt, cnt


class SQLiteNGramStore:
    """SQLite-backed n-gram store."""

    def __init__(
        self,
        path: Path,
        commit_every: int = 2000,
        *,
        busy_timeout_ms: int = 60000,
        synchronous: str = "NORMAL",
        mmap_mb: int = 64,
        cache_mb: int = 0,
        force_vacuum: bool = False,
        vacuum_threshold_mb: int = 256,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.commit_every = commit_every
        self.busy_timeout_ms = busy_timeout_ms
        self.synchronous = synchronous
        self.mmap_mb = mmap_mb
        self.cache_mb = cache_mb
        self.force_vacuum = force_vacuum
        self.vacuum_threshold_mb = vacuum_threshold_mb
        self.conn = self._open_db()
        self._buf: List[Row] = []

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        sync_mode = os.environ.get("NGDB_SYNCHRONOUS", self.synchronous)
        cur.execute(f"PRAGMA synchronous={sync_mode}")
        cur.execute("PRAGMA temp_store=MEMORY")
        mmap_mb = int(os.environ.get("NGDB_MMAP_MB", str(self.mmap_mb)))
        cur.execute(f"PRAGMA mmap_size={mmap_mb * 1024 * 1024}")
        cache_mb = int(os.environ.get("NGDB_CACHE_MB", str(self.cache_mb)))
        if cache_mb:
            cur.execute(f"PRAGMA cache_size={-cache_mb * 1024}")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ngram (
              ctx_hash INTEGER NOT NULL,
              aux_id   INTEGER NOT NULL,
              next_evt INTEGER NOT NULL,
              count    INTEGER NOT NULL,
              PRIMARY KEY(ctx_hash, aux_id, next_evt)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ngram_ctx ON ngram(ctx_hash, aux_id)")
        conn.commit()
        return conn

    def inc(self, ctx_hash: int, aux_id: int, next_event: int, n: int = 1) -> None:
        self.bulk_inc([(ctx_hash, aux_id, next_event, n)])

    def _flush_pending(self) -> None:
        if not self._buf:
            return
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        cur.executemany(
            """
            INSERT INTO ngram(ctx_hash, aux_id, next_evt, count)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(ctx_hash, aux_id, next_evt)
            DO UPDATE SET count = count + excluded.count
            """,
            self._buf,
        )
        self.conn.commit()
        self._buf.clear()

    def bulk_inc(self, rows: Iterable[Row]) -> None:
        self._buf.extend(list(rows))
        if len(self._buf) >= self.commit_every:
            self._flush_pending()

    def topk(self, ctx_hash: int, aux_id: int, k: int = 8) -> List[Tuple[int, int]]:
        cur = self.conn.execute(
            """
            SELECT next_evt, count FROM ngram
            WHERE ctx_hash = ? AND aux_id = ?
            ORDER BY count DESC LIMIT ?
            """,
            (ctx_hash, aux_id, k),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def prune(self, min_count: int) -> None:
        self.conn.execute("DELETE FROM ngram WHERE count < ?", (min_count,))
        self.conn.commit()

    def finalize(self) -> None:
        if getattr(self, "conn", None) is None:
            return
        self._flush_pending()
        cur = self.conn.cursor()
        try:
            cur.execute("ANALYZE")
            self.conn.commit()
            if self._should_vacuum():
                cur.execute("VACUUM")
        finally:
            self.conn.close()
            self.conn = None

    def close(self) -> None:  # pragma: no cover - trivial wrapper
        self.finalize()

    def stats(self) -> Dict[str, int]:
        cur = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(count),0) FROM ngram")
        rows, total = cur.fetchone()
        return {"rows": rows, "total": total}

    def iter_rows(self) -> Iterable[Row]:
        cur = self.conn.execute("SELECT ctx_hash, aux_id, next_evt, count FROM ngram")
        yield from cur

    # Internal helpers -----------------------------------------------------

    def _should_vacuum(self) -> bool:
        if self.force_vacuum:
            return True
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:  # pragma: no cover - unlikely
            return False
        return size > self.vacuum_threshold_mb * 1024 * 1024
