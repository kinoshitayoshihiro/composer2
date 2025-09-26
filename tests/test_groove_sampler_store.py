from multiprocessing import Process, Queue
from pathlib import Path

from utilities.hash_utils import murmur32
from utilities.ngram_store import SQLiteNGramStore


def _murmur_worker(q: Queue) -> None:
    q.put(murmur32(b"abc"))


def test_sqlite_store_basic(tmp_path: Path) -> None:
    db = tmp_path / "ngrams.db"
    store = SQLiteNGramStore(db, commit_every=1000)
    rows = []
    for i in range(200000):
        rows.append((i % 5, 0, i % 7, 1))
    store.bulk_inc(rows)
    top = store.topk(0, 0, k=3)
    assert top[0][1] >= top[1][1] >= top[2][1]
    store.prune(2)
    stats = store.stats()
    store.finalize()
    assert stats["rows"] <= 35  # 5 ctx * 7 events
    # Ensure database file exists and is not empty
    assert db.stat().st_size > 0


def test_murmur32_stable_across_process() -> None:
    q = Queue()

    p = Process(target=_murmur_worker, args=(q,))
    p.start()
    p.join()
    assert q.get() == murmur32(b"abc")


def test_busy_timeout_and_finalize(tmp_path: Path) -> None:
    db = tmp_path / "ngrams.db"
    store = SQLiteNGramStore(db, commit_every=10, busy_timeout_ms=1234)
    # Busy timeout should be configured
    timeout = store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert timeout == 1234
    store.bulk_inc([(1, 0, 2, 1)])
    store.finalize()  # Should not raise


def test_murmur32_known_value() -> None:
    assert murmur32(b"ctx|mood=melancholic|section=chorus") == 0xF8A8F792


def test_flush_upsert(tmp_path: Path) -> None:
    db = tmp_path / "ngrams.db"
    store = SQLiteNGramStore(db, commit_every=100)
    store.bulk_inc([(1, 2, 3, 1)])
    store.bulk_inc([(1, 2, 3, 1)])
    store._flush_pending()
    assert store.topk(1, 2, 1)[0][1] == 2
    store.finalize()


def test_finalize_vacuum_threshold(tmp_path: Path) -> None:
    db = tmp_path / "ngrams.db"
    store = SQLiteNGramStore(db, commit_every=1, vacuum_threshold_mb=9999)
    store.bulk_inc([(1, 0, 1, 1)])
    traces: list[str] = []
    store.conn.set_trace_callback(traces.append)
    store.finalize()
    assert any("ANALYZE" in t for t in traces)
    assert not any("VACUUM" in t for t in traces)

    db2 = tmp_path / "ngrams2.db"
    store2 = SQLiteNGramStore(db2, commit_every=1, vacuum_threshold_mb=0)
    store2.bulk_inc([(1, 0, 1, 1)])
    traces2: list[str] = []
    store2.conn.set_trace_callback(traces2.append)
    store2.finalize()
    assert any("VACUUM" in t for t in traces2)
