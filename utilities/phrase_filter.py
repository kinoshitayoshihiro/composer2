from __future__ import annotations

from collections.abc import Sequence
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
try:  # optional dependency
    import hdbscan
except ImportError:  # pragma: no cover - optional
    hdbscan = None  # type: ignore


def _phrase_str(events: Sequence[dict]) -> str:
    return " ".join(f"{ev.get('instrument','x')}_{round(ev.get('offset',0)*100)}" for ev in events)


def cluster_phrases(events_list: Sequence[Sequence[dict]], n: int = 4) -> List[bool]:
    """Cluster phrases by 3-gram similarity and return keep mask."""
    if not events_list:
        return []
    texts = [_phrase_str(ev) for ev in events_list]
    vec = CountVectorizer(analyzer="word", ngram_range=(3, 3))
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    if hdbscan is None:
        warnings.warn("hdbscan not installed; skipping phrase clustering", RuntimeWarning)
        return [True] * len(texts)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=n, metric="precomputed")
    labels = clusterer.fit_predict(1 - sim)
    keep: List[bool] = []
    seen: set[int] = set()
    for lab in labels:
        if lab == -1 or lab not in seen:
            keep.append(True)
            if lab != -1:
                seen.add(lab)
        else:
            keep.append(False)
    return keep

__all__ = ["cluster_phrases"]
